"""Tokenizer-only C decoding for every predeclared training snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(16)
    protocol_path=args.run/'protocol.json';protocol=json.loads(protocol_path.read_text());config=protocol['C']
    seed=config['seeds'][args.index//len(config['arms'])];arm=config['arms'][args.index%len(config['arms'])]
    directory=args.run/'C'/f'{arm}_seed{seed}'
    training=json.loads((directory/'training_summary.json').read_text())
    if training['protocol_sha256']!=hashlib.sha256(protocol_path.read_bytes()).hexdigest():raise ValueError('Protocol mismatch')
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    decoded=directory/'decoded';decoded.mkdir(exist_ok=True)
    snapshots=[]
    for point in training['history']:
        step=point['step'];rows=[]
        for name in protocol['geometry_targets']:
            output=decoded/f'step{step}_{name}.json'
            if output.exists():rows.append(json.loads(output.read_text()));continue
            with np.load(directory/'predictions'/f'step{step}_{name}.npz') as a:fine=a['fine'].copy();coarse=a['coarse'].copy()
            with np.load(directory/'predictions'/f'step0_{name}.npz') as a:
                if not np.array_equal(coarse,a['coarse']):raise ValueError('Coarse changed across trajectory')
            bb=decode_embedding(decoder,decoder.embed(torch.from_numpy(fine).long())).numpy()
            if not np.isfinite(bb).all():raise RuntimeError('Nonfinite trajectory decode')
            metrics=summarize_backbone(bb);n=len(fine)
            row=dict(target=name,length=n,cn_rate=metrics['backbone_cn_outliers']/max(1,n-1),
                ca_clashes_per_residue=metrics['upstream_ca_clashes']/n,metrics=metrics)
            atomic_json(output,row);rows.append(row)
        snapshot=dict(step=step,epoch=point['epoch'],validation_ce=point['validation_ce'],train_ce_fixed16=point['train_ce_fixed16'],
            cn_rate=float(np.mean([r['cn_rate'] for r in rows])),
            ca_clashes_per_residue=float(np.mean([r['ca_clashes_per_residue'] for r in rows])),
            ca_validity=float(np.mean([r['metrics']['upstream_ca_clash_valid'] for r in rows])),targets=rows)
        snapshots.append(snapshot)
        print('OBJECTIVE_TRAJECTORY_GEOMETRY',arm,seed,step,snapshot['cn_rate'],snapshot['ca_validity'],flush=True)
    atomic_json(directory/'geometry_summary.json',dict(arm=arm,seed=seed,protocol_sha256=training['protocol_sha256'],
        snapshots=snapshots,no_update_control=snapshots[0],
        note='All planned snapshots, no checkpoint selection. Original no-update control is step0 repeated at every progress.'))


if __name__=='__main__':main()
