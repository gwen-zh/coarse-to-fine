"""Stage-wise checkpoint intervention on already sampled native coarse tokens.

Recompute all four combinations on the same device. The adapted checkpoint has
shared LoRA weights: swapping a stage means using that checkpoint for that stage,
not claiming independent coarse-only/fine-only training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_native_generator import (load_state_dict_from_lightning_ckpt, EsmSequenceTokenizer,
    tokenize_sequence, pad_coarse, restore_trainable_state, atomic_json)
from native_generator import install_native_lora, raw_fine_logits

VARIANTS = ('base_base', 'adapt_base', 'base_adapt', 'adapt_adapt')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['source','tokens','checkpoint','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=16)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    saved=torch.load(args.source/'best_adapter.pt',map_location='cpu',weights_only=True)
    cfg=saved['protocol']
    token_hash=hashlib.sha256((args.tokens/'manifest.json').read_bytes()).hexdigest()
    if token_hash!=cfg['diagnostic_manifest_sha256']:
        raise ValueError('Diagnostic tokens differ from source experiment')
    protocol=dict(version=1,samples=cfg['samples'],device=args.device,threads=args.threads,
        source_adapter_sha256=hashlib.sha256((args.source/'best_adapter.pt').read_bytes()).hexdigest(),
        diagnostic_manifest_sha256=token_hash,source_protocol=cfg,
        note='Stage-wise inference interventions on a jointly adapted checkpoint, not separately trained '
             'coarse-only/fine-only models. All four fine predictions recomputed on the same device. '
             'Cached coarse plans are from the original paired GPU samples. No reference enters inference.')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Cross experiment protocol changed')
    atomic_json(path,protocol)
    model=load_state_dict_from_lightning_ckpt(args.checkpoint,device=args.device)
    install_native_lora(model,cfg['last_blocks'],cfg['rank'])
    model.eval()
    tokenizer=EsmSequenceTokenizer()
    selected=[t for t in json.loads((args.tokens/'manifest.json').read_text())['targets'] if t['subset']=='validation']
    discrepancies=[]
    for fine_source in ['base','adapt']:
        if fine_source=='adapt':
            restore_trainable_state(model,saved['state'])
        for coarse_source,source_label in [('base','baseline'),('adapt','adapted')]:
            label=f'{coarse_source}_{fine_source}'
            directory=args.output/label
            predictions=directory/'predictions'
            predictions.mkdir(parents=True,exist_ok=True)
            for t in selected:
                sequence=tokenize_sequence(t['sequence'],tokenizer).to(args.device)[None]
                for sample in range(cfg['samples']):
                    original=args.source/source_label/'predictions'/f"{t['id']}_sample{sample}.npz"
                    with np.load(original) as a:
                        coarse=torch.from_numpy(a['coarse'].copy()).long().to(args.device)[None]
                        expected=a['tokens'][:,0].copy()
                    if coarse.shape!=(1,t['length']) or torch.any((coarse<0)|(coarse>=32)):
                        raise ValueError('Invalid cached coarse')
                    fine=raw_fine_logits(model,sequence,pad_coarse(coarse)).argmax(-1)[0].cpu().numpy()
                    np.savez_compressed(predictions/f"{t['id']}_sample{sample}.npz",
                                        tokens=fine[:,None],coarse=coarse[0].cpu().numpy())
                    if coarse_source==fine_source:
                        discrepancies.append(dict(variant=label,target=t['id'],sample=sample,
                                                  changed_fine_tokens=int((fine!=expected).sum()),length=len(fine)))
                print('CROSS_PREDICTED',label,t['id'],flush=True)
            atomic_json(directory/'summary.json',dict(protocol=dict(protocol,variant=label),proteins=len(selected)))
    atomic_json(args.output/'device_replay_diagnostic.json',dict(
        note='Compare CPU-recomputed diagonal controls against cached GPU fine IDs. '
             'All causal comparisons use newly recomputed same-device controls.',
        changed_tokens=sum(r['changed_fine_tokens'] for r in discrepancies),targets=discrepancies))
    print('NATIVE_STAGE_CROSS_COMPLETE',flush=True)


if __name__=='__main__':main()
