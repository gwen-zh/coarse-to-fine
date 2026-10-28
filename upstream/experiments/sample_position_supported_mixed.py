"""Development candidate: original mixed weights, train-only fine slot support.

Consumes sequence-only features and random seed. No reference coordinates,
encoded native latents or geometry-based acceptance enter generation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hierarchical_pipeline_io import sha
from mixed_hierarchical_generator import MixedHierarchicalGenerator, HybridTokenizer
from position_supported_fine import sample_supported
RUN=Path('artifacts/mixed_generator_position_support_v1')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-seed',type=int,choices=[2026,2027],required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    deployment=json.loads((RUN/'deployment_manifest.json').read_text())
    paths=deployment['models'][str(args.model_seed)]
    for path,digest in {**deployment['source_sha256'],**paths['sha256']}.items():
        if sha(path)!=digest:raise RuntimeError('Development candidate input changed: '+path)
    saved=torch.load(paths['generator'],map_location='cpu',weights_only=False)
    state=saved['model']
    model=MixedHierarchicalGenerator(state['mean'],state['std'],**saved['model_config'])
    model.load_state_dict(state);model.eval().requires_grad_(False)
    decoder=HybridTokenizer.from_checkpoints(paths['warmup'],paths['tokenizer_final'])
    counts=np.load(paths['counts'])
    with np.load(args.features) as z:h=torch.from_numpy(z['h'].copy()).float()
    if h.ndim!=2 or len(h)==0 or h.shape[1]!=1536 or not torch.isfinite(h).all():
        raise ValueError('Finite Nx1536 AA-only features required')
    with torch.no_grad():
        c=model.sample_coarse(h,args.seed,16)
        f=sample_supported(model.fine,h,model.normalize(c),counts,(args.seed+104729)%(2**32),16)
        bb=decoder.decode(c,f)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,coarse=c.numpy(),fine=f.numpy(),backbone=bb.numpy())
    print('POSITION_SUPPORTED_SEQUENCE_SAMPLE_WRITTEN',args.output,flush=True)


if __name__=='__main__':main()
