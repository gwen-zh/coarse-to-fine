"""Frozen-head features on original-policy coarse; no backbone retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_native_generator import (load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,
    tokenize_sequence,pad_coarse,atomic_json)
from native_generator import raw_fine_logits


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['tokens','diagnostic_tokens','source','checkpoint','output']:
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(16)
    source=json.loads((args.source/'protocol.json').read_text())
    train_hash=hashlib.sha256((args.tokens/'manifest.json').read_bytes()).hexdigest()
    diag_hash=hashlib.sha256((args.diagnostic_tokens/'manifest.json').read_bytes()).hexdigest()
    if train_hash!=source['training_manifest_sha256'] or diag_hash!=source['diagnostic_manifest_sha256']:
        raise ValueError('Original-policy prefix provenance mismatch')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=1,training_manifest_sha256=train_hash,diagnostic_manifest_sha256=diag_hash,
        source_protocol_sha256=hashlib.sha256((args.source/'protocol.json').read_bytes()).hexdigest(),
        features='FP32 input to original fine final linear; no teacher/reference coarse in inference features',
        device='cpu',threads=16)
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Feature cache protocol mismatch')
    atomic_json(path,protocol)
    model=load_state_dict_from_lightning_ckpt(args.checkpoint,device='cpu')
    model.eval().requires_grad_(False)
    head=model.net.output_heads.fine_structure_head[-1]
    projection=dict(weight=head.weight.detach().cpu(),bias=head.bias.detach().cpu())
    torch.save(projection,args.output/'fine_projection.pt')
    captured=[]
    hook=head.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()))
    tokenizer=EsmSequenceTokenizer()
    rows=[]
    for dataset,manifest in [('corpus',args.tokens),('diagnostic',args.diagnostic_tokens)]:
        for item in json.loads((manifest/'manifest.json').read_text())['targets']:
            n=item['length']
            samples=1 if dataset=='corpus' else 2
            for sample in range(samples):
                out=args.output/f"{dataset}_{item['id']}_{sample}.npz"
                if not out.exists():
                    prefix=(args.source/'initial_prefixes'/f"{item['id']}.npz" if dataset=='corpus' else
                            args.source/'baseline/predictions'/f"{item['id']}_sample{sample}.npz")
                    with np.load(prefix) as a:
                        coarse=torch.from_numpy(a['coarse'].copy()).long()[None]
                        expected=a['tokens'][:,0].copy() if dataset=='diagnostic' else None
                    if coarse.shape!=(1,n) or torch.any((coarse<0)|(coarse>=32)):
                        raise ValueError('Invalid original-policy coarse')
                    captured.clear()
                    seq=tokenize_sequence(item['sequence'],tokenizer)[None]
                    logits=raw_fine_logits(model,seq,pad_coarse(coarse))
                    h=captured[-1][0,1:-1].float().clone()
                    reconstructed=F.linear(h,head.weight[:4096],head.bias[:4096]).argmax(-1)
                    if not torch.equal(reconstructed,logits.argmax(-1)[0]):
                        raise ValueError('Feature cache does not reproduce fine logits')
                    if expected is not None and not np.array_equal(reconstructed.numpy(),expected):
                        raise ValueError('Diagnostic prefix replay differs from original baseline')
                    np.savez_compressed(out,h=h.numpy(),coarse=coarse[0].numpy(),fine=reconstructed.numpy())
                rows.append(dict(id=item['id'],subset=item['subset'],dataset=dataset,length=n,sample=sample,
                                 features=out.name,tokens=str((manifest/item['tokens']).resolve())))
            if len(rows)%25==0:
                print('RETENTION_FEATURES',len(rows),flush=True)
    hook.remove()
    atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=rows))
    print('FINE_RETENTION_FEATURES_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
