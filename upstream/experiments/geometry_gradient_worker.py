"""Tokenizer-only process: differentiate hard-forward geometry with respect to logits."""
import argparse
import json
import sys
from pathlib import Path

import torch

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from geometry_objective import hard_token_embedding,backbone_penalties
from structural_metrics import summarize_backbone


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--exchange',type=Path,required=True)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args();torch.set_num_threads(args.threads)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    print(json.dumps(dict(status='ready')),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('command')=='close':break
        logits=torch.load(args.exchange/'request.pt',map_location='cpu',weights_only=True)['logits'].requires_grad_()
        embedding,ids,_=hard_token_embedding(logits,decoder.embed.weight[:4096])
        bb=decode_embedding(decoder,embedding)
        cn,clash=backbone_penalties(bb)
        gradient=torch.autograd.grad(cn+clash,logits)[0]
        if not torch.isfinite(cn+clash) or not torch.isfinite(gradient).all():
            raise RuntimeError('Nonfinite decoder geometry or gradient')
        if any(p.requires_grad or p.grad is not None for p in decoder.parameters()):
            raise RuntimeError('Decoder parameters must remain frozen')
        torch.save(dict(gradient=gradient,fine=ids.detach(),cn_loss=cn.item(),clash_loss=clash.item(),
            metrics=summarize_backbone(bb.detach().numpy())),args.exchange/'response.pt')
        print(json.dumps(dict(status='complete',request=request['request'])),flush=True)


if __name__=='__main__':main()
