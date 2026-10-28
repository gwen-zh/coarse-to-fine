"""Encode newly screened training structures; never touches benchmark structures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Hierarchical_VQ_VAE'))
from esm.models.new_vqvae import ProjectedEncoder
from esm.models.residual_bridge import load_residual_bridge
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for key in ['corpus','weights','bridges','output']:
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(8)
    manifest_path=args.corpus/'manifest.json'
    corpus=json.loads(manifest_path.read_text())
    if len(corpus['targets'])<200:
        raise RuntimeError('Too few screened structures; inspect curation before training')
    args.output.mkdir(exist_ok=True,parents=True)
    protocol=dict(version=1,source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  residual_bridge_sha256=hashlib.sha256((args.bridges/'residual_bridge.pt').read_bytes()).hexdigest(),
                  parallel_bridge_sha256=hashlib.sha256((args.bridges/'parallel_bridge.pt').read_bytes()).hexdigest(),
                  note='Frozen published encoder and existing development-trained codebooks. New corpus only; '
                       'benchmark structures are not read, encoded or evaluated. Curation filtering is heuristic.')
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Encoded training corpus provenance changed')
    atomic_json(path,protocol)
    encoder=ProjectedEncoder(d_model=1024,n_heads=1,v_heads=128,n_layers=2,d_out=128,n_codes=4096)
    state=torch.load(args.weights/'encoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    encoder.load_state_dict(state.get('encoder_weight',state),strict=True)
    del state
    encoder.eval().requires_grad_(False)
    residual,_=load_residual_bridge(args.bridges/'residual_bridge.pt')
    parallel,_=load_residual_bridge(args.bridges/'parallel_bridge.pt')
    latent=[]
    hook=encoder.pre_vq_proj.register_forward_hook(lambda m,i,o:latent.append(o.detach()))
    rows=[]
    for t in corpus['targets']:
        path=args.output/(t['id']+'.npz')
        if not path.exists():
            with np.load(args.corpus/t['backbone']) as a:
                backbone=torch.from_numpy(a['backbone'].copy())
            assert backbone.shape==(len(t['sequence']),3,3)
            if not torch.isfinite(backbone).all():
                raise ValueError('Nonfinite training backbone')
            latent.clear()
            output=encoder.encode(backbone[None])
            z=latent[0][0]
            if not torch.isfinite(z).all():
                raise ValueError('Nonfinite encoder latent')
            np.savez_compressed(path,latent=z.numpy(),residual_tokens=residual.encode(z).numpy(),
                                parallel_tokens=parallel.encode(z).numpy(),fine=output[3][0].numpy())
        rows.append(dict(id=t['id'],sequence=t['sequence'],length=len(t['sequence']),
                         subset=t['subset'],sequence_hash=t['sequence_hash'],tokens=path.name,
                         source_entry=t['entry'],source_auth_chain=t['auth_chain']))
        if len(rows)%25==0:print('ENCODED',len(rows),flush=True)
    hook.remove()
    atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=rows,
                training_proteins=sum(t['subset']=='train' for t in rows),
                validation_proteins=sum(t['subset']=='validation' for t in rows)))
    print('PDB_TRAINING_ENCODING_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
