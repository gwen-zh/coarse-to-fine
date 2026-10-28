"""Matched tiny final-projection LoRA, with/without original-model retention."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fine_retention import build_head,retention_kl
from coarse_lora import trainable_state
from train_residual_vq import atomic_json


@torch.no_grad()
def validate(model,items):
    return float(np.mean([F.cross_entropy(model(t['h']),t['labels']).item() for t in items]))


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','anchors','anchor_manifest','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--variant',choices=['plain','retained'],required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    projection=torch.load(args.cache/'fine_projection.pt',map_location='cpu',weights_only=True)
    anchor_projection=torch.load(args.anchors/'fine_projection.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(projection[k],anchor_projection[k]) for k in ['weight','bias']):
        raise ValueError('Anchor features and current original head weights differ')
    model=build_head(projection,rank=8)
    manifest=json.loads((args.cache/'manifest.json').read_text())
    train,validation=[],[]
    for row in manifest['targets']:
        if row['dataset']!='corpus':continue
        with np.load(args.cache/row['features']) as a:
            h=torch.from_numpy(a['h'].copy())
        with np.load(row['tokens']) as a:
            labels=torch.from_numpy(a['fine'].copy()).long()
        if h.shape[0]!=len(labels) or not torch.isfinite(h).all():
            raise ValueError('Invalid frozen fine features')
        (train if row['subset']=='train' else validation).append(dict(id=row['id'],h=h,labels=labels))
    anchors=[]
    diagnostic_ids={r['id'] for r in manifest['targets'] if r['dataset']=='diagnostic'}
    for row in json.loads(args.anchor_manifest.read_text())['targets']:
        if row['subset']!='train':continue
        if row['id'] in diagnostic_ids:raise ValueError('Diagnostic sequence used as retention anchor')
        with np.load(args.anchors/(row['id']+'.npz')) as a:
            h=torch.from_numpy(a['sampled_h'].copy())
        if not torch.isfinite(h).all():raise ValueError('Nonfinite anchor')
        anchors.append(h)
    if not train or not validation or not anchors:raise ValueError('Empty subset')
    beta=1. if args.variant=='retained' else 0.
    protocol=dict(version=1,variant=args.variant,rank=8,lr=1e-5,epochs=5,snapshot_epochs=[0,1,3,5],
        beta=beta,anchor_beta=beta,crop=256,anchor_crop=128,seed=0,samples=2,
        cache_manifest_sha256=hashlib.sha256((args.cache/'manifest.json').read_bytes()).hexdigest(),
        anchor_manifest_sha256=hashlib.sha256(args.anchor_manifest.read_bytes()).hexdigest(),
        train_proteins=len(train),validation_proteins=len(validation),anchor_sequences=len(anchors),
        trainable='rank8 LoRA on final fine output projection only; base projection and all upstream parameters frozen',
        data='New corpus sampled original coarse. Old development TRAIN sequences used as sequence-only teacher KL anchors; no anchor labels.',
        loss='fine CE + beta KL(original||adapted) on current inputs + anchor_beta KL on original sequence-only anchors',
        selection='Deferred: new-corpus validation CE subject to original-relative geometry constraints on 24 preselected new validation proteins; epoch0 eligible',
        scope='Development pilot. Diagnostics and holdout are not used for fitting or checkpoint selection.')
    directory=args.output/args.variant
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Retention run protocol changed')
    atomic_json(path,protocol)
    snapshots=directory/'snapshots';snapshots.mkdir(exist_ok=True)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=protocol['lr'],weight_decay=0.)
    rng=torch.Generator().manual_seed(2026)
    history=[]

    def save(epoch,ce):
        torch.save(dict(state=trainable_state(model),epoch=epoch,validation_ce=ce,protocol=protocol),snapshots/f'epoch{epoch}.pt')

    initial=validate(model,validation)
    save(0,initial)
    history.append(dict(epoch=0,validation_ce=initial))
    for epoch in range(1,6):
        losses=[]
        for index in torch.randperm(len(train),generator=rng).tolist():
            t=train[index];n=len(t['h'])
            start=int(torch.randint(max(1,n-256+1),(),generator=rng))
            h,y=t['h'][start:start+256],t['labels'][start:start+256]
            anchor=anchors[int(torch.randint(len(anchors),(),generator=rng))]
            offset=int(torch.randint(max(1,len(anchor)-128+1),(),generator=rng))
            anchor=anchor[offset:offset+128]
            optimizer.zero_grad(set_to_none=True)
            logits=model(h)
            ce=F.cross_entropy(logits,y)
            kl=retention_kl(logits,model.base(h))
            anchor_kl=retention_kl(model(anchor),model.base(anchor))
            loss=ce+beta*kl+beta*anchor_kl
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite retention loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
            optimizer.step()
            losses.append(loss.item())
        ce=validate(model,validation)
        if epoch in protocol['snapshot_epochs']:save(epoch,ce)
        row=dict(epoch=epoch,validation_ce=ce,train_loss=float(np.mean(losses)))
        history.append(row)
        atomic_json(directory/'history.json',history)
        print(json.dumps(row),flush=True)
    atomic_json(directory/'training_summary.json',dict(protocol=protocol,history=history,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    print('FINE_RETENTION_TRAINING_COMPLETE',args.variant,flush=True)


if __name__=='__main__':main()
