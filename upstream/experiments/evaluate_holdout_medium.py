"""Fixed58 reconstruction confirmation; no fitting or generation package."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tokenizer_diagnostics import ProjectedEncoder,load_weights,ESMProtein
from tokenizer_distillation_control import ProjectedDecoder,decode_table
from quantization_control import assignments
from structural_metrics import aligned_rmsd,summarize_backbone
from diagnostic_io import atomic_json
from holdout_confirmation import load_locked


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--index',type=int,required=True);args=parser.parse_args();torch.set_num_threads(16)
    protocol,digest=load_locked(args.run)
    if not 0<=args.index<protocol['chunks']:raise ValueError('Invalid chunk')
    output=args.run/'medium';output.mkdir(exist_ok=True)
    encoder=ProjectedEncoder(d_model=1024,n_heads=1,v_heads=128,n_layers=2,d_out=128,n_codes=4096)
    load_weights(encoder,Path(protocol['inputs']['encoder']['path']),'encoder_weight');encoder.eval().requires_grad_(False)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    load_weights(decoder,Path(protocol['inputs']['decoder']['path']),'decode_proj');decoder.eval().requires_grad_(False)
    patch=torch.load(protocol['inputs']['medium_patch']['path'],map_location='cpu',weights_only=True)
    if patch['method']!='kmeans512' or patch['train_proteins']!=164:raise ValueError('Unexpected fixed patch')
    latent=[];hook=encoder.pre_vq_proj.register_forward_hook(lambda m,i,o:latent.append(o.detach()))
    for target in protocol['targets'][args.index::protocol['chunks']]:
        path=output/(target['id']+'.json')
        if path.exists():continue
        protein=ESMProtein.from_pdb(Path('upstream')/target['paths'][0])
        if protein.sequence!=target['sequence']:raise ValueError('Reference sequence changed')
        reference=protein.coordinates[:,:3].numpy().copy()
        if not np.isfinite(reference).all():raise ValueError('Previously eligible reference is nonfinite')
        latent.clear();encoded=encoder.encode(protein.coordinates[None]);z=latent[-1][0].numpy();fine=encoded[3][0].numpy();medium=encoded[4][0].numpy()
        with threadpool_limits(limits=1):repaired=assignments(z,patch['encoder_weight'].numpy())
        methods={};arrays=dict(latent=z,original_medium=medium,repaired_medium=repaired,fine=fine,reference=reference)
        for label,ids,table in [('original_medium',medium,decoder.medium_embed.weight),('repaired_medium',repaired,patch['decoder_weight']),('original_fine',fine,decoder.embed.weight)]:
            bb=decode_table(decoder,ids,table).numpy();arrays['backbone_'+label]=bb
            methods[label]=dict(all_finite=bool(np.isfinite(bb).all()),ca_rmsd=aligned_rmsd(bb[:,1],reference[:,1]),metrics=summarize_backbone(bb))
        np.savez_compressed(output/(target['id']+'.npz'),**arrays)
        atomic_json(path,dict(target=target['id'],length=target['length'],protocol_sha256=digest,methods=methods));print('HOLDOUT_MEDIUM_COMPLETE',target['id'],flush=True)
    hook.remove()


if __name__=='__main__':main()
