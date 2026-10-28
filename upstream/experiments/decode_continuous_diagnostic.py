"""Decode frozen16/64-step diagnostic outputs through the original outlet."""
import json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone,aligned_rmsd
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    run=Path('artifacts/development/continuous_diagnostic_v1');p=json.loads((run/'protocol.json').read_text());old=json.loads((Path(p['source'])/'protocol.json').read_text())
    if json.loads((run/'flow_audit.json').read_text())['status']!='complete':raise RuntimeError('Flow check incomplete')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);weights=torch.load(old['decoder'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(weights.get('decode_proj',weights),strict=True);del weights;decoder.to('cuda').eval().requires_grad_(False)
    records=[]
    for seed in p['seeds']:
        for target in p['targets']:
            with np.load(Path(old['corpus'])/target['backbone']) as a:ref=a['backbone'].copy()
            with np.load(run/f'seed{seed}'/(target['id']+'.npz')) as a:
                for sample in range(2):
                    bbs={}
                    for steps in [16,64]:
                        ids=a[f'fine_{steps}_{sample}'];bb=decode_embedding(decoder,decoder.embed(torch.tensor(ids,device='cuda',dtype=torch.long))).cpu().numpy()
                        if not np.isfinite(bb).all():raise RuntimeError('Nonfinite decoded diagnostic')
                        bbs[steps]=bb
                        records.append(dict(seed=seed,target=target['id'],subset=target['subset'],sample=sample,steps=steps,length=target['length'],metrics=summarize_backbone(bb,[ref[:,1]])))
                    records[-1]['paired_drift_from_16']=aligned_rmsd(bbs[64][:,1],bbs[16][:,1])
    summaries=[]
    for seed in p['seeds']:
        for subset in ['train','validation']:
            for steps in [16,64]:
                rows=[r for r in records if r['seed']==seed and r['subset']==subset and r['steps']==steps]
                summaries.append(dict(seed=seed,subset=subset,steps=steps,rmsd=float(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in rows])),cn_rate=float(np.mean([r['metrics']['backbone_cn_outliers']/(r['length']-1) for r in rows]))))
    atomic_json(run/'decoded.json',dict(status='complete',records=records,summaries=summaries,scope=p['scope']))
    print('FLOW_DIAGNOSTIC_DECODE_COMPLETE',flush=True)

if __name__=='__main__':main()
