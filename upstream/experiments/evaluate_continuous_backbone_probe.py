"""Frozen original decoder evaluation after bounded pretrained FM adaptation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone,aligned_rmsd
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed-index',type=int,required=True);a=p.parse_args()
    run=Path('artifacts/development/continuous_backbone_probe_v1');protocol=json.loads((run/'protocol.json').read_text());seed=protocol['seeds'][a.seed_index];out=run/('seed'+str(seed))
    if json.loads((out/'training.json').read_text())['status']!='complete':raise RuntimeError('Training incomplete')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);weights=torch.load(protocol['decoder'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(weights.get('decode_proj',weights),strict=True);del weights
    decoder.to('cuda').eval().requires_grad_(False)
    def decode(ids):return decode_embedding(decoder,decoder.embed(torch.tensor(ids,device='cuda',dtype=torch.long))).cpu().numpy()
    records=[]
    for item in protocol['targets']:
        name=item['id']
        with np.load(Path(protocol['corpus'])/item['backbone']) as a:reference=a['backbone'].copy()
        with np.load(Path(protocol['tokens'])/(name+'.npz')) as a:truth=a['fine'].copy();true_latent=a['latent'].copy()
        with np.load(out/'step0'/(name+'.npz')) as a:native=decode(a['native_fine'])
        oracle=decode(truth);row=dict(target=name,subset=item['subset'],native=summarize_backbone(native,[reference[:,1]]),oracle=summarize_backbone(oracle,[reference[:,1]]),steps={})
        for step in [0,protocol['steps']]:
            with np.load(out/('step'+str(step))/(name+'.npz')) as a:tokens=a['fine'].copy();latents=a['latent'].copy()
            samples=[];bbs=[]
            for ids,latent in zip(tokens,latents):
                bb=decode(ids);bbs.append(bb)
                if not np.isfinite(bb).all():raise RuntimeError('Nonfinite decoded output')
                try:quality=measurements(bb,reference);error=None
                except ValueError as e:quality=None;error=str(e)
                samples.append(dict(metrics=summarize_backbone(bb,[reference[:,1]]),quality=quality,quality_error=error,latent_mse=float(np.mean((latent-true_latent)**2)),fine_accuracy=float(np.mean(ids==truth)),used_fine_codes=len(np.unique(ids))))
            row['steps'][str(step)]=dict(samples=samples,conditional_spread=aligned_rmsd(bbs[0][:,1],bbs[1][:,1]))
        records.append(row);atomic_json(out/'decoded.json',dict(status='running',records=records))
        print('CONTINUOUS_BACKBONE_DECODED',seed,name,flush=True)
    atomic_json(out/'decoded.json',dict(status='complete',records=records,scope=protocol['scope']))

if __name__=='__main__':main()
