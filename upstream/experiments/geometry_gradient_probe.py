"""Test frozen decoder gradients by optimizing local logits, never model weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from geometry_objective import hard_token_embedding,backbone_penalties,mix_logit_gradients
from structural_metrics import summarize_backbone
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--index',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(16)
    manifest=json.loads((args.run/'manifest.json').read_text())
    if args.index>=len(manifest['targets']):return
    item=manifest['targets'][args.index];protocol=manifest['protocol']
    with np.load(args.run/item['input']) as a:
        initial=torch.from_numpy(a['logits'].copy());original=a['fine'].copy();reference=a['reference_ca'].copy()
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(args.weights/'decoder_checkpoint_110.pth',map_location='cpu',mmap=True,weights_only=False)
    decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    teacher=initial.softmax(-1)
    directory=args.run/item['target'];directory.mkdir(exist_ok=True)
    summaries={}
    for arm in protocol['arms']:
        logits=torch.nn.Parameter(initial.clone())
        optimizer=torch.optim.Adam([logits],lr=protocol['lr'])
        generator=torch.Generator().manual_seed(2026+args.index)
        trace=[];best=None;best_ids=original.copy();best_bb=None;baseline=None;first_gradient_norm=None
        for step in range(protocol['steps']+1):
            optimizer.zero_grad(set_to_none=True)
            embedding,ids,probability=hard_token_embedding(logits,decoder.embed.weight[:4096])
            if step==0 and not np.array_equal(ids.numpy(),original):raise ValueError('Initial IDs differ')
            bb=decode_embedding(decoder,embedding)
            cn,clash=backbone_penalties(bb)
            kl=F.kl_div(logits.log_softmax(-1),teacher,reduction='batchmean')
            if not torch.isfinite(cn+clash+kl):raise RuntimeError('Nonfinite geometry loss')
            metrics=summarize_backbone(bb.detach().numpy(),[reference])
            if baseline is None:baseline=metrics
            row=dict(step=step,cn_loss=cn.item(),clash_loss=clash.item(),kl=kl.item(),
                     objective=(cn+clash+kl).item(),changed_tokens=int((ids.numpy()!=original).sum()),metrics=metrics)
            trace.append(row)
            eligible=(metrics.get('backbone_cn_outliers',999999)<=baseline['backbone_cn_outliers'] and
                      metrics.get('upstream_ca_clashes',999999)<=baseline['upstream_ca_clashes'])
            if eligible and (best is None or (metrics['backbone_cn_outliers'],row['objective'])<
                                          (best['metrics']['backbone_cn_outliers'],best['objective'])):
                best=row;best_ids=ids.numpy().copy();best_bb=bb.detach().numpy().copy()
            if step==protocol['steps']:break
            geometry_gradient=torch.autograd.grad(cn+clash,logits)[0]
            if not torch.isfinite(geometry_gradient).all():raise RuntimeError('Nonfinite decoder-to-logit gradient')
            if step==0:
                if geometry_gradient.norm()==0:raise RuntimeError('No geometry gradient reaches logits')
                first_gradient_norm=geometry_gradient.norm().item()
                print('FULL_DECODER_TO_LOGIT_GRADIENT_VERIFIED',item['target'],arm,geometry_gradient.norm().item(),flush=True)
            order=torch.randperm(logits.shape[-1],generator=generator) if arm=='shuffled_geometry_gradient' else None
            logits.grad=mix_logit_gradients(geometry_gradient,probability.detach(),teacher,
                                          geometry_weight=0. if arm=='kl_only' else 1.,permutation=order)
            torch.nn.utils.clip_grad_norm_([logits],1.)
            optimizer.step()
            if step%5==0:print('GEOMETRY_PROBE_STEP',item['target'],arm,step,row['cn_loss'],row['metrics']['backbone_cn_outliers'],flush=True)
        atomic_json(directory/(arm+'.json'),dict(protocol=protocol,baseline=baseline,best=best,final=trace[-1],trace=trace,first_gradient_norm=first_gradient_norm))
        np.savez_compressed(directory/(arm+'.npz'),fine=best_ids,backbone=best_bb)
        summaries[arm]=dict(best=best,final=trace[-1])
    if any(v.grad is not None or v.requires_grad for v in decoder.parameters()):raise RuntimeError('Frozen decoder weights changed training status')
    atomic_json(directory/'summary.json',dict(target=item['target'],arms=summaries,
        note='Local geometry-guided token optimization on selected TRAIN examples, not trained generator or independent benchmark; biased surrogate gradients.'))
    print('GEOMETRY_GRADIENT_PROBE_COMPLETE',item['target'],flush=True)


if __name__=='__main__':main()
