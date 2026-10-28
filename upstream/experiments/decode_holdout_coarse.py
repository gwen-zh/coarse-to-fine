"""Decode fixed coarse arms and evaluate the frozen first reference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding,ESMProtein
from esm.models.residual_bridge import load_residual_bridge
from structural_metrics import aligned_rmsd,summarize_backbone
from diagnostic_io import atomic_json
from holdout_confirmation import load_locked


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--index',type=int,required=True);args=parser.parse_args();torch.set_num_threads(16);protocol,digest=load_locked(args.run)
    if not 0<=args.index<protocol['chunks']:raise ValueError('Invalid chunk')
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);checkpoint=torch.load(protocol['inputs']['decoder']['path'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(checkpoint.get('decode_proj',checkpoint),strict=True);del checkpoint;decoder.eval().requires_grad_(False)
    bridge,_=load_residual_bridge(protocol['inputs']['residual_bridge']['path']);output=args.run/'coarse';decoded=output/'decoded';decoded.mkdir(exist_ok=True)
    for target in protocol['targets'][args.index::protocol['chunks']]:
        path=decoded/(target['id']+'.json')
        if path.exists():continue
        protein=ESMProtein.from_pdb(Path('upstream')/target['paths'][0])
        if protein.sequence!=target['sequence']:raise ValueError('Reference sequence mismatch')
        reference=protein.coordinates[:,1].numpy();methods={}
        for arm in ['control','adapted']:
            with np.load(output/'predictions'/(arm+'_'+target['id']+'.npz')) as a:tokens=torch.from_numpy(a['tokens'].copy()).long()
            fine=bridge.fine_tokens(tokens);samples=[];backbones=[]
            for sample in range(protocol['coarse']['samples']):
                bb=decode_embedding(decoder,decoder.embed(fine[sample])).numpy();backbones.append(bb);samples.append(dict(ca_rmsd=aligned_rmsd(bb[:,1],reference),all_finite=bool(np.isfinite(bb).all()),metrics=summarize_backbone(bb)))
            methods[arm]=dict(samples=samples,mean_pairwise_ca_rmsd=aligned_rmsd(backbones[0][:,1],backbones[1][:,1]));np.savez_compressed(decoded/(arm+'_'+target['id']+'.npz'),fine=fine.numpy(),backbone=np.stack(backbones))
        atomic_json(path,dict(target=target['id'],length=target['length'],protocol_sha256=digest,methods=methods));print('HOLDOUT_COARSE_DECODED',target['id'],flush=True)


if __name__=='__main__':main()
