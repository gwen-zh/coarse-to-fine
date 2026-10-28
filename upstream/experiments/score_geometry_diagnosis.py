"""A: frozen fine-head score versus actual hard-decoded backbone geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from likelihood_geometry import sample_tokens,fixed_log_score,spearman
from structural_metrics import summarize_backbone
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(16)
    protocol_path=args.run/'protocol.json';protocol=json.loads(protocol_path.read_text())
    cache=Path(protocol['fine_cache']);corpus=Path(protocol['corpus']);config=protocol['A']
    if not 0<=args.index<config['chunks']:raise ValueError('Invalid chunk')
    if hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest()!=protocol['hashes']['fine_manifest']:
        raise ValueError('Fine manifest changed')
    if hashlib.sha256((cache/'fine_projection.pt').read_bytes()).hexdigest()!=protocol['hashes']['fine_projection']:
        raise ValueError('Fine projection changed')
    rows={r['id']:r for r in json.loads((cache/'manifest.json').read_text())['targets'] if r['dataset']=='corpus'}
    references={r['id']:r for r in json.loads((corpus/'manifest.json').read_text())['targets']}
    projection=torch.load(cache/'fine_projection.pt',map_location='cpu',weights_only=True)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    output=args.run/'A';output.mkdir(exist_ok=True)
    for name in protocol['geometry_targets'][args.index::config['chunks']]:
        path=output/(name+'.json')
        if path.exists():continue
        row=rows[name]
        if row['subset']!='validation':raise ValueError('Unexpected subset')
        with np.load(cache/row['features']) as a:
            logits=F.linear(torch.from_numpy(a['h'].copy()),projection['weight'][:4096],projection['bias'][:4096])
            coarse=a['coarse'].copy();original=torch.from_numpy(a['fine'].copy()).long()
        if not torch.equal(logits.argmax(-1),original):raise ValueError('Original argmax replay failed')
        with np.load(row['tokens']) as a:truth=torch.from_numpy(a['fine'].copy()).long()
        with np.load(corpus/references[name]['backbone']) as a:reference_backbone=a['backbone'].copy()
        reference=reference_backbone[:,1]
        tokens=[original,truth];metadata=[dict(source='argmax',temperature=None,sample=0),dict(source='encoded_reference',temperature=None,sample=0)]
        for temperature in config['temperatures']:
            key=f'{name}:{temperature}:{config["seed"]}'
            seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)
            sampled=sample_tokens(logits,temperature,config['samples_per_temperature'],seed)
            tokens.extend(sampled)
            metadata.extend(dict(source='sampled',temperature=temperature,sample=i) for i in range(len(sampled)))
        tokens=torch.stack(tokens);scores=fixed_log_score(logits,tokens).tolist();records=[]
        for fine,score,meta in zip(tokens,scores,metadata):
            bb=decode_embedding(decoder,decoder.embed(fine)).numpy()
            if not np.isfinite(bb).all():raise RuntimeError('Nonfinite hard decode')
            metrics=summarize_backbone(bb,[reference]);n=len(fine)
            records.append(dict(meta,log_score=score,cn_rate=metrics['backbone_cn_outliers']/max(1,n-1),
                ca_clashes_per_residue=metrics['upstream_ca_clashes']/n,metrics=metrics))
        correlations={}
        for temperature in config['temperatures']:
            group=[r for r in records if r['source']=='sampled' and r['temperature']==temperature]
            correlations[str(temperature)]={metric:spearman([r['log_score'] for r in group],[r[metric] for r in group])
                for metric in ['cn_rate','ca_clashes_per_residue']}
        np.savez_compressed(output/(name+'.npz'),fine=tokens.numpy(),coarse=coarse)
        atomic_json(path,dict(target=name,length=len(original),protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            correlations=correlations,records=records,experimental_reference_metrics=summarize_backbone(reference_backbone)))
        print('FROZEN_SCORE_GEOMETRY_COMPLETE',name,json.dumps(correlations),flush=True)


if __name__=='__main__':main()
