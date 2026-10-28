"""Matched residual/parallel 512-code predictors on frozen coarse context."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans

from residual_condition import MidPredictor,residual_target
from residual_quantizer import PrefixQuantizer
from train_residual_vq import atomic_json


@torch.no_grad()
def validate(model,items,centers,frequency):
    rows=[]
    for item in items:
        logits=model(item['h'])
        ids=logits.argmax(-1)
        rows.append(dict(target=item['id'],ce=F.cross_entropy(logits,item['labels']).item(),
            accuracy=(ids==item['labels']).float().mean().item(),
            constant_frequency_ce=-frequency[item['labels']].log().mean().item(),
            oracle_mse=F.mse_loss(centers[item['labels']],item['target']).item(),
            predicted_mse=F.mse_loss(centers[ids],item['target']).item()))
    return dict(means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='target'},targets=rows)


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','latents','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['residual','parallel'],required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    manifest=json.loads((args.cache/'manifest.json').read_text())
    coarse_codes=torch.load(args.cache/'coarse_codes.pt',map_location='cpu',weights_only=True)
    train,validation=[],[]
    for item in manifest['targets']:
        if item['dataset']!='corpus':continue
        with np.load(args.cache/item['features']) as a:
            h=torch.from_numpy(a['coarse_h'].copy());coarse=torch.from_numpy(a['coarse'].copy()).long()
        with np.load(args.latents/(item['id']+'.npz')) as a:z=torch.from_numpy(a['latent'].copy())
        target=residual_target(z,coarse,coarse_codes,args.mode)
        (train if item['subset']=='train' else validation).append(dict(id=item['id'],h=h,target=target))
    directory=args.output/args.mode;directory.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,mode=args.mode,codes=512,width=128,epochs=20,lr=3e-4,seed=0,
        cache_manifest_sha256=hashlib.sha256((args.cache/'manifest.json').read_bytes()).hexdigest(),
        train_proteins=len(train),validation_proteins=len(validation),
        target='z minus published embedding of ACTUALLY sampled original coarse' if args.mode=='residual' else 'z',
        selection='Minimum mean new-validation mid CE; initialized epoch0 eligible',
        scope='Only new corpus labels used; all predictions use frozen sequence/coarse context. Training predictions are in-sample, not out-of-fold.')
    path=directory/'mid_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Mid protocol mismatch')
    atomic_json(path,protocol)
    values=torch.cat([t['target'] for t in train])
    quantizer=MiniBatchKMeans(n_clusters=512,batch_size=4096,n_init=3,max_iter=100,random_state=0,reassignment_ratio=.01)
    quantizer.fit(values.numpy())
    centers=torch.from_numpy(quantizer.cluster_centers_.copy())
    for item in train+validation:item['labels']=PrefixQuantizer.nearest(item['target'],centers)
    counts=torch.bincount(torch.cat([t['labels'] for t in train]),minlength=512).float()+1
    frequency=counts/counts.sum()
    model=MidPredictor(width=train[0]['h'].shape[-1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    rng=torch.Generator().manual_seed(2026)
    initial=validate(model,validation,centers,frequency)
    best_score=initial['means']['ce'];best_epoch=0
    best={k:v.detach().clone() for k,v in model.state_dict().items()}
    history=[]
    for epoch in range(1,21):
        for index in torch.randperm(len(train),generator=rng).tolist():
            t=train[index]
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(t['h']),t['labels'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite mid predictor CE')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        result=validate(model,validation,centers,frequency)
        if result['means']['ce']<best_score:
            best_score=result['means']['ce'];best_epoch=epoch
            best={k:v.detach().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch,**result['means']))
        atomic_json(directory/'mid_history.json',history)
        print('MID_EPOCH',args.mode,json.dumps(history[-1]),flush=True)
    model.load_state_dict(best);model.eval()
    torch.save(dict(protocol=protocol,epoch=best_epoch,state=best,centers=centers),directory/'mid.pt')
    predictions=directory/'mid_predictions';predictions.mkdir(exist_ok=True)
    with torch.no_grad():
        for item in manifest['targets']:
            with np.load(args.cache/item['features']) as a:h=torch.from_numpy(a['coarse_h'].copy())
            ids=model(h).argmax(-1)
            np.savez_compressed(predictions/item['features'],mid=ids.numpy(),condition=centers[ids].numpy())
    atomic_json(directory/'mid_summary.json',dict(protocol=protocol,selected_epoch=best_epoch,
        initial=initial,selected=validate(model,validation,centers,frequency),training_code_usage=int((counts>1).sum())))
    print('MID_PREDICTOR_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
