"""Sequence-only sampling for either repair fine head and coarse source."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mixed_hierarchical_io import tokenizer
from train_mixed_generator_repair import RUN
from regularized_mixed_generator import RegularizedMixedGenerator


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-seed', type=int, required=True)
    parser.add_argument('--fine-head', choices=['ce','metric'], required=True)
    parser.add_argument('--coarse-source', choices=['original','repaired'], default='repaired')
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args=parser.parse_args()
    p=json.loads((RUN/'protocol.json').read_text())
    parent=json.loads(Path(p['parent_protocol']).read_text())
    root=RUN/f'fine_{args.fine_head}'/f'seed{args.model_seed}'
    if json.loads((root/'complete.json').read_text())['status']!='complete':
        raise RuntimeError('Completed model required')
    decoder=tokenizer(parent,args.model_seed)
    saved=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    state=saved['model']
    model=RegularizedMixedGenerator(state['mean'],state['std'],**saved['model_config'],
                                    codes=decoder.model.fine_book.codes)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    if args.coarse_source=='original':
        from mixed_hierarchical_generator import MixedHierarchicalGenerator
        old=torch.load(Path(p['parent_run'])/'fine'/f'seed{args.model_seed}'/'final.pt',map_location='cpu',weights_only=False)
        original=MixedHierarchicalGenerator(old['model']['mean'],old['model']['std'],**old['model_config'])
        original.load_state_dict(old['model']);original.eval().requires_grad_(False)
        model.coarse_flow=original.coarse_flow
    with np.load(args.features) as z:h=torch.from_numpy(z['h'].copy()).float()
    if h.ndim!=2 or h.shape[1]!=1536 or not torch.isfinite(h).all():
        raise ValueError('AA-only Nx1536 features required')
    with torch.no_grad():
        c,f=model.sample(h,args.seed,p['coarse']['sampling_steps'],p['fine']['sampling_steps'],1.)
        backbone=decoder.decode(c,f)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,coarse=c.numpy(),fine=f.numpy(),backbone=backbone.numpy())
    print('REPAIR_SEQUENCE_SAMPLE_WRITTEN',args.output,flush=True)


if __name__=='__main__':main()
