"""Decode fixed new-validation candidates and select without diagnostic peeking.

Runs only tokenizer ESM; fine tail inference was completed in another process.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from fine_retention import select_geometry_safe
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['run','weights']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['residual','parallel','geometry','no_geometry'],required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    directory=args.run/args.mode
    training=json.loads((directory/'training_summary.json').read_text())
    protocol=training['protocol']
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True)
    del state
    decoder.eval().requires_grad_(False)
    candidates=[]
    for epoch in protocol['snapshot_epochs']:
        snapshot=torch.load(directory/'snapshots'/f'epoch{epoch}.pt',map_location='cpu',weights_only=True)
        rows=[]
        for target in protocol['geometry_targets']:
            out=directory/'geometry'/f'epoch{epoch}'/(target+'.json')
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():rows.append(json.loads(out.read_text()));continue
            with np.load(directory/'epochs'/f'epoch{epoch}'/'geometry_predictions'/(target+'.npz')) as a:
                fine=torch.from_numpy(a['tokens'][:,0].copy()).long()
            bb=decode_embedding(decoder,decoder.embed(fine)).numpy()
            row=dict(target=target,length=len(fine),all_finite=bool(np.isfinite(bb).all()),metrics=summarize_backbone(bb))
            atomic_json(out,row);rows.append(row)
        candidate=dict(epoch=epoch,validation_ce=snapshot['validation_ce'],all_finite=all(r['all_finite'] for r in rows),
            ca_validity=float(np.mean([r['metrics'].get('upstream_ca_clash_valid',False) for r in rows])),
            cn_rate=float(np.mean([r['metrics'].get('backbone_cn_outliers',r['length'])/max(1,r['length']-1) for r in rows])),
            ca_clash_rate=float(np.mean([r['metrics'].get('upstream_ca_clashes',r['length'])/r['length'] for r in rows])))
        candidates.append(candidate)
        print('MID_CONDITION_GEOMETRY',args.mode,json.dumps(candidate),flush=True)
    best,eligible=select_geometry_safe(candidates)
    epoch=best['epoch']
    for label,folder in [(args.mode,'predictions'),(args.mode+'_shuffled','shuffled_predictions')]:
        dest=args.run/label;dest.mkdir(exist_ok=True)
        link=dest/'predictions'
        target=(directory/'epochs'/f'epoch{epoch}'/folder).resolve()
        if link.is_symlink():
            if link.resolve()!=target:raise ValueError('Existing selected epoch differs; use new output directory')
        elif link.exists():raise ValueError('Prediction path already exists')
        else:os.symlink(target,link,target_is_directory=True)
        atomic_json(dest/'summary.json',dict(protocol=dict(protocol,shuffled=folder=='shuffled_predictions'),
            selected_epoch=epoch,eligible_epochs=[r['epoch'] for r in eligible],candidates=candidates,
            note='New validation geometry gate and CE selection; original epoch0 eligible. Diagnostic proteins not used for selection.'))
    print('RESIDUAL_CONDITION_SELECTION_COMPLETE',args.mode,epoch,flush=True)


if __name__=='__main__':main()
