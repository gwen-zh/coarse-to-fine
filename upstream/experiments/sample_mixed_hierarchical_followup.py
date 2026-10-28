"""Sample trained mixed model from AA-only features, no structure argument."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mixed_hierarchical_io import sha,tokenizer
RUN=Path('artifacts/mixed_hierarchical_followup_v1')
from mixed_hierarchical_generator import MixedHierarchicalGenerator


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model-seed',type=int,required=True);parser.add_argument('--features',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--device',default='cpu');args=parser.parse_args()
 p=json.loads((RUN/'protocol.json').read_text());root=RUN/'fine'/f'seed{args.model_seed}'
 if not (root/'complete.json').exists():raise RuntimeError('Fine generation training is not complete')
 for path in p['tokenizers'][str(args.model_seed)].values():
  if sha(path)!=p['source_sha256'][path]:raise RuntimeError('Tokenizer changed')
 saved=torch.load(root/'final.pt',map_location='cpu',weights_only=False);state=saved['model'];model=MixedHierarchicalGenerator(state['mean'],state['std'],**saved['model_config']).to(args.device);model.load_state_dict(state);model.eval().requires_grad_(False)
 with np.load(args.features) as z:h=torch.from_numpy(z['h'].copy()).float().to(args.device)
 if h.ndim!=2 or h.shape[1]!=1536 or not torch.isfinite(h).all():raise ValueError('Finite N x1536 amino-acid-only features required')
 decoder=tokenizer(p,args.model_seed,args.device)
 with torch.no_grad():c,f=model.sample(h,args.seed,p['coarse']['sampling_steps'],p['fine']['sampling_steps'],p['fine']['temperature']);bb=decoder.decode(c,f)
 args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,backbone=bb.cpu().numpy(),coarse=c.cpu().numpy(),fine=f.cpu().numpy())
 print('MIXED_SEQUENCE_SAMPLE_WRITTEN',args.output,flush=True)


if __name__=='__main__':main()
