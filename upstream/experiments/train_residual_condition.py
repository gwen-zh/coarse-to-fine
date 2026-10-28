"""Train zero-initialized condition injection into frozen pretrained fine tail."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hierarchical_ConfGen'))
from residual_condition import load_tail
from fine_retention import retention_kl
from coarse_lora import trainable_state
from train_residual_vq import atomic_json


def load_item(row,cache,directory,labels=True):
    with np.load(cache/row['features']) as a:
        result=dict(id=row['id'],prefix=torch.from_numpy(a['prefix'].copy())[None],
            skip=torch.from_numpy(a['skip'].copy())[None],teacher_h=torch.from_numpy(a['teacher_h'].copy()),
            coarse=a['coarse'].copy(),baseline_fine=a['fine'].copy())
    with np.load(directory/'mid_predictions'/row['features']) as a:result['condition']=torch.from_numpy(a['condition'].copy())[None]
    if labels:
        with np.load(row['tokens']) as a:result['labels']=torch.from_numpy(a['fine'].copy()).long()
    return result


@torch.no_grad()
def validate(model,items):
    return float(np.mean([F.cross_entropy(model(t['prefix'],t['skip'],t['condition'])[0],t['labels']).item() for t in items]))


@torch.no_grad()
def save_predictions(model,rows,cache,directory,epoch,protocol):
    out=directory/'epochs'/f'epoch{epoch}'
    geometry=out/'geometry_predictions';geometry.mkdir(parents=True,exist_ok=True)
    diagnostics=out/'predictions';diagnostics.mkdir(exist_ok=True)
    shuffled=out/'shuffled_predictions';shuffled.mkdir(exist_ok=True)
    for row in rows:
        item=load_item(row,cache,directory,labels=False)
        fine=model(item['prefix'],item['skip'],item['condition']).argmax(-1)[0].numpy()
        if epoch==0 and not np.array_equal(fine,item['baseline_fine']):raise ValueError('Epoch zero no longer reproduces original fine')
        if row['dataset']=='corpus':
            np.savez_compressed(geometry/(row['id']+'.npz'),tokens=fine[:,None],coarse=item['coarse'])
        else:
            path=f"{row['id']}_sample{row['sample']}.npz"
            np.savez_compressed(diagnostics/path,tokens=fine[:,None],coarse=item['coarse'])
            seed=int(hashlib.sha256((row['id']+str(row['sample'])).encode()).hexdigest()[:8],16)
            rng=torch.Generator().manual_seed(seed)
            order=torch.randperm(item['condition'].shape[1],generator=rng)
            other=model(item['prefix'],item['skip'],item['condition'][:,order]).argmax(-1)[0].numpy()
            np.savez_compressed(shuffled/path,tokens=other[:,None],coarse=item['coarse'])
    atomic_json(out/'summary.json',dict(protocol=protocol,epoch=epoch))


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','run']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['residual','parallel'],required=True)
    args=p.parse_args()
    torch.set_num_threads(16);torch.manual_seed(0)
    directory=args.run/args.mode
    bundle=torch.load(args.cache/'tail.pt',map_location='cpu',weights_only=True)
    model=load_tail(bundle)
    manifest=json.loads((args.cache/'manifest.json').read_text())
    mid=torch.load(directory/'mid.pt',map_location='cpu',weights_only=True)
    if mid['protocol']['cache_manifest_sha256']!=hashlib.sha256((args.cache/'manifest.json').read_bytes()).hexdigest():
        raise ValueError('Mid and fine conditions use different data')
    corpus=[r for r in manifest['targets'] if r['dataset']=='corpus']
    train=[load_item(r,args.cache,directory) for r in corpus if r['subset']=='train']
    validation=[load_item(r,args.cache,directory) for r in corpus if r['subset']=='validation']
    geometry=sorted([r for r in corpus if r['subset']=='validation'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:24]
    diagnostic=[r for r in manifest['targets'] if r['dataset']=='diagnostic']
    if len(geometry)!=24:raise ValueError('Expected fixed 24 geometry validation proteins')
    protocol=dict(version=1,mode=args.mode,epochs=5,snapshot_epochs=[0,1,3,5],lr=1e-4,beta=1.,seed=0,samples=2,
        cache_manifest_sha256=mid['protocol']['cache_manifest_sha256'],mid_selected_epoch=mid['epoch'],
        mid_checkpoint_sha256=hashlib.sha256((directory/'mid.pt').read_bytes()).hexdigest(),
        train_proteins=len(train),validation_proteins=len(validation),geometry_targets=[r['id'] for r in geometry],
        injection='Zero initialized 128->1536 projection before original final4 fine attention blocks; all original weights frozen',
        loss='Native fine CE + KL(original fine || conditioned fine) on same sampled coarse',
        scope='Pilot: cached predicted mid from separately fitted same-data policy, not out-of-fold. One extra original coarse context pass vs 7-call original baseline; residual/parallel have matched inference and parameter cost. No independent test claim.')
    path=directory/'condition_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Condition training protocol changed')
    atomic_json(path,protocol)
    snapshots=directory/'snapshots';snapshots.mkdir(exist_ok=True)
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4,weight_decay=0.)
    rng=torch.Generator().manual_seed(2026)
    history=[]

    def snapshot(epoch,score):
        torch.save(dict(state=trainable_state(model),epoch=epoch,validation_ce=score,protocol=protocol),snapshots/f'epoch{epoch}.pt')
        save_predictions(model,geometry+diagnostic,args.cache,directory,epoch,protocol)

    initial=validate(model,validation);snapshot(0,initial)
    history.append(dict(epoch=0,validation_ce=initial))
    for epoch in range(1,6):
        for index in torch.randperm(len(train),generator=rng).tolist():
            item=train[index]
            optimizer.zero_grad(set_to_none=True)
            logits=model(item['prefix'],item['skip'],item['condition'])[0]
            teacher=F.linear(item['teacher_h'],model.head[-1].weight[:4096],model.head[-1].bias[:4096])
            loss=F.cross_entropy(logits,item['labels'])+retention_kl(logits,teacher)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual condition loss')
            loss.backward();torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.);optimizer.step()
        ce=validate(model,validation)
        if epoch in protocol['snapshot_epochs']:snapshot(epoch,ce)
        row=dict(epoch=epoch,validation_ce=ce);history.append(row)
        atomic_json(directory/'condition_history.json',history)
        print('CONDITION_EPOCH',args.mode,json.dumps(row),flush=True)
    atomic_json(directory/'training_summary.json',dict(protocol=protocol,history=history,
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
    print('RESIDUAL_CONDITION_TRAINING_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
