"""Select on new validation geometry; diagnostics never select checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from fine_retention import build_head,select_geometry_safe
from coarse_lora import restore_trainable_state
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['cache','weights','run']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--variant',choices=['plain','retained'],required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    projection=torch.load(args.cache/'fine_projection.pt',map_location='cpu',weights_only=True)
    model=build_head(projection)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    weights=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(weights.get('decode_proj',weights),strict=True)
    del weights
    decoder.eval().requires_grad_(False)
    manifest=json.loads((args.cache/'manifest.json').read_text())
    validation=[r for r in manifest['targets'] if r['dataset']=='corpus' and r['subset']=='validation']
    selected=sorted(validation,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:24]
    if len(selected)!=24:raise ValueError('Expected 24 new validation geometry targets')
    directory=args.run/args.variant
    training=json.loads((directory/'training_summary.json').read_text())
    protocol=dict(training['protocol'],geometry_targets=[r['id'] for r in selected],
        geometry_rule='No decrease in CA-valid fraction, no increase in mean per-protein C-N outlier rate or CA clashes/residue; all coordinates finite. Epoch0 always eligible. Among eligible snapshots choose lowest new-validation CE.')
    atomic_json(directory/'selection_protocol.json',protocol)
    candidates=[]
    for epoch in protocol['snapshot_epochs']:
        snapshot=torch.load(directory/'snapshots'/f'epoch{epoch}.pt',map_location='cpu',weights_only=True)
        restore_trainable_state(model,snapshot['state'])
        rows=[]
        for item in selected:
            out=directory/'geometry'/f'epoch{epoch}'/(item['id']+'.json')
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():
                rows.append(json.loads(out.read_text()));continue
            with np.load(args.cache/item['features']) as a:h=torch.from_numpy(a['h'].copy())
            fine=model(h).argmax(-1)
            bb=decode_embedding(decoder,decoder.embed(fine)).numpy()
            metrics=summarize_backbone(bb)
            row=dict(target=item['id'],length=len(fine),all_finite=bool(np.isfinite(bb).all()),metrics=metrics)
            atomic_json(out,row)
            rows.append(row)
        candidate=dict(epoch=epoch,validation_ce=snapshot['validation_ce'],
            all_finite=all(r['all_finite'] for r in rows),
            ca_validity=float(np.mean([r['metrics'].get('upstream_ca_clash_valid',False) for r in rows])),
            ca_clash_rate=float(np.mean([r['metrics'].get('upstream_ca_clashes',r['length'])/r['length'] for r in rows])),
            cn_rate=float(np.mean([r['metrics'].get('backbone_cn_outliers',r['length'])/max(1,r['length']-1) for r in rows])))
        candidates.append(candidate)
        print('RETENTION_GEOMETRY',json.dumps(candidate),flush=True)
    best,eligible=select_geometry_safe(candidates)
    snapshot=torch.load(directory/'snapshots'/f"epoch{best['epoch']}.pt",map_location='cpu',weights_only=True)
    restore_trainable_state(model,snapshot['state'])
    torch.save(snapshot,directory/'best_adapter.pt')
    predictions=directory/'predictions';predictions.mkdir(exist_ok=True)
    for item in manifest['targets']:
        if item['dataset']!='diagnostic':continue
        with np.load(args.cache/item['features']) as a:
            h=torch.from_numpy(a['h'].copy());coarse=a['coarse'].copy()
        fine=model(h).argmax(-1).numpy()
        np.savez_compressed(predictions/f"{item['id']}_sample{item['sample']}.npz",tokens=fine[:,None],coarse=coarse)
    atomic_json(directory/'summary.json',dict(protocol=protocol,selected_epoch=best['epoch'],
        eligible_epochs=[r['epoch'] for r in eligible],candidates=candidates,selected=best,
        note='Geometric criteria are optimized validation proxies, not independent physical validation. Original coarse is unchanged by construction.'))
    print('FINE_RETENTION_SELECTION_COMPLETE',args.variant,best['epoch'],flush=True)


if __name__=='__main__':main()
