"""Sequence-only entry: frozen old coarse + selected trained fine alignment arm."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hierarchical_pipeline_io import sha
from mixed_hierarchical_generator import MixedHierarchicalGenerator,HybridTokenizer
from position_aligned_fine import PositionedFineModel,ARMS
RUN=Path('artifacts/position_aligned_fine_v1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-seed',type=int,choices=[2026,2027],required=True);parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=2026);args=parser.parse_args()
    torch.set_num_threads(4)
    p=json.loads((RUN/'protocol.json').read_text());root=RUN/args.arm/f'seed{args.model_seed}'
    if json.loads((root/'complete.json').read_text())['status']!='complete':raise RuntimeError('Complete fine checkpoint required')
    checkpoint=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    if checkpoint['protocol_sha256']!=sha(RUN/'protocol.json'):raise RuntimeError('Fine checkpoint protocol mismatch')
    state=checkpoint['model'];fine=PositionedFineModel(state['mean'],state['std'],state['position_counts'],**checkpoint['model_config']);fine.load_state_dict(state);fine.eval().requires_grad_(False)
    path=p['old_generator'][str(args.model_seed)]
    parent=json.loads(Path(p['parent_protocol']).read_text())
    tok=parent['tokenizers'][str(args.model_seed)]
    for source in [path,*tok.values(),p['counts'][str(args.model_seed)]]:
        if sha(source)!=p['source_sha256'][source]:raise RuntimeError('Frozen deployment input changed '+source)
    old=torch.load(path,map_location='cpu',weights_only=False)
    coarse=MixedHierarchicalGenerator(old['model']['mean'],old['model']['std'],**old['model_config']);coarse.load_state_dict(old['model']);coarse.eval().requires_grad_(False)
    decoder=HybridTokenizer.from_checkpoints(tok['warmup'],tok['final'])
    with np.load(args.features) as z:h=torch.from_numpy(z['h'].copy()).float()
    if h.ndim!=2 or len(h)==0 or h.shape[1]!=1536 or not torch.isfinite(h).all():raise ValueError('Finite AA-only Nx1536 features required')
    with torch.no_grad():c=coarse.sample_coarse(h,args.seed,16);f=fine.sample(h,c,(args.seed+104729)%(2**32),16);bb=decoder.decode(c,f)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,coarse=c.numpy(),fine=f.numpy(),backbone=bb.numpy())
    print('POSITION_ALIGNED_SEQUENCE_SAMPLE_WRITTEN',args.output,flush=True)


if __name__=='__main__':main()
