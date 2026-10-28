"""Pick four baseline C-N-violating TRAIN proteins for gradient feasibility only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['features','corpus','weights','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(16)
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path=args.features/'manifest.json'
    source=json.loads(manifest_path.read_text())
    projection=torch.load(args.features/'fine_projection.pt',map_location='cpu',weights_only=True)
    corpus={r['id']:r for r in json.loads((args.corpus/'manifest.json').read_text())['targets']}
    candidates=sorted([r for r in source['targets'] if r['dataset']=='corpus' and r['subset']=='train'],
                      key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    protocol=dict(version=1,source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        max_scanned=32,max_probes=4,selection='First4 baseline CN-violating TRAIN proteins in fixed SHA256 ID order among first32 train candidates',
        purpose='Local per-sequence logit optimization feasibility, NOT generator training or an independent improvement benchmark',
        loss='Hard-forward CN squared hinge outside[1.1,1.7]A + mean CA clash squared hinge below3A + teacher KL',
        estimator='Biased straight-through softmax gradient, exact hard-token decoder forward',steps=20,lr=.05,
        arms=['kl_only','geometry_gradient','shuffled_geometry_gradient'])
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Probe protocol changed')
    atomic_json(path,protocol)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    scanned,selected=[],[]
    for row in candidates:
        with np.load(args.features/row['features']) as a:
            h=torch.from_numpy(a['h'].copy());fine=torch.from_numpy(a['fine'].copy()).long()
        logits=F.linear(h,projection['weight'][:4096],projection['bias'][:4096])
        if not torch.equal(logits.argmax(-1),fine):raise ValueError('Original logit replay failed')
        with np.load(args.corpus/corpus[row['id']]['backbone']) as a:reference=a['backbone'][:,1].copy()
        bb=decode_embedding(decoder,decoder.embed(fine)).numpy()
        metrics=summarize_backbone(bb,[reference])
        scanned.append(dict(target=row['id'],metrics=metrics))
        if metrics.get('backbone_cn_outliers',0)>0:
            filename=row['id']+'.npz'
            np.savez_compressed(args.output/filename,logits=logits.numpy(),fine=fine.numpy(),reference_ca=reference)
            selected.append(dict(target=row['id'],input=filename,baseline=metrics))
            print('GEOMETRY_PROBE_SELECTED',row['id'],metrics['backbone_cn_outliers'],flush=True)
        if len(selected)==4:break
    if not selected:raise RuntimeError('No CN-violating examples in fixed training scan')
    atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=selected,scanned=scanned))
    print('GEOMETRY_GRADIENT_PROBE_PREPARED',len(selected),flush=True)


if __name__=='__main__':main()
