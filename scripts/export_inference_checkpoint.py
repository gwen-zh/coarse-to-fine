"""Create local portable snapshots from trusted training checkpoints.

This writes local output only. Model tensors are not added to Git.
"""
import argparse
from pathlib import Path
import torch
from project_model import ROOT
from mixed_hierarchical_generator import HybridTokenizer

p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='kind',required=True)
a=sub.add_parser('tokenizer');a.add_argument('--warmup',type=Path,required=True);a.add_argument('--final',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
a=sub.add_parser('generator');a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
args=p.parse_args();torch.set_num_threads(2)
if args.output.exists():raise FileExistsError('Will not overwrite '+str(args.output))
if args.kind=='tokenizer':
    model=HybridTokenizer.from_checkpoints(args.warmup,args.final).model
    payload=dict(format_version=1,model_config=model.config,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
else:
    state=torch.load(args.source,map_location='cpu',weights_only=False)
    payload=dict(format_version=1,model_config=state['model_config'],model={k:v.detach().cpu().clone() for k,v in state['model'].items()})
args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,args.output);print('Saved local inference snapshot:',args.output)
