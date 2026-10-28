"""Cache coarse context and original fine prefix/tail boundary, with replay guards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from train_native_generator import (load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,
    tokenize_sequence,pad_coarse,raw_coarse_logits,atomic_json)
from native_generator import raw_fine_logits
from residual_condition import load_tail


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ['features','source','checkpoint','encoder','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(16)
    manifest=json.loads((args.features/'manifest.json').read_text())
    protocol=dict(version=1,tail_blocks=4,
        source_manifest_sha256=hashlib.sha256((args.features/'manifest.json').read_bytes()).hexdigest(),
        source_protocol_sha256=hashlib.sha256((args.source/'protocol.json').read_bytes()).hexdigest(),
        features='Original sampled coarse context, original fine input to block44, original fine skip, original final-linear input',
        inference_cost='One additional original coarse-context forward beyond original 5-step sampler; residual and parallel share cost. Not yet matched against optimized equal-compute baseline.')
    if manifest['protocol']['source_protocol_sha256']!=protocol['source_protocol_sha256']:
        raise ValueError('Original feature/prefix policies differ')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Residual cache protocol changed')
    atomic_json(path,protocol)
    model=load_state_dict_from_lightning_ckpt(args.checkpoint,device='cpu')
    model.eval().requires_grad_(False)
    stack=model.net.transformer
    config=dict(width=model.net.d_model,heads=stack.blocks[-1].n_heads,blocks=4,
                scaling_factor=stack.blocks[-1].scaling_factor)
    bundle=dict(config=config,blocks=torch.nn.ModuleList(list(stack.blocks)[-4:]).state_dict(),
                norm=stack.norm.state_dict(),head=model.net.output_heads.fine_structure_head.state_dict())
    torch.save(bundle,args.output/'tail.pt')
    state=torch.load(args.encoder,map_location='cpu',mmap=True,weights_only=False)
    codes=state.get('encoder_weight',state)['small_codebook.embeddings'].clone()
    torch.save(codes,args.output/'coarse_codes.pt')
    del state
    tail=load_tail(bundle)
    captures={}
    hooks=[stack.register_forward_pre_hook(lambda m,i:captures.update(skip=i[0].detach().clone())),
           stack.blocks[-4].register_forward_pre_hook(lambda m,i:captures.update(prefix=i[0].detach().clone())),
           model.net.output_heads.coarse_structure_head[-1].register_forward_pre_hook(lambda m,i:captures.update(coarse_h=i[0].detach().clone()))]
    tokenizer=EsmSequenceTokenizer()
    sequences={}
    for dataset,path in [('corpus',args.source.parents[1]/'pdb_training_v2/native_tokens/manifest.json'),
                         ('diagnostic',args.source.parent/'native_tokens/manifest.json')]:
        sequences[dataset]={r['id']:r['sequence'] for r in json.loads(path.read_text())['targets']}
    rows=[]
    for index,item in enumerate(manifest['targets']):
        out=args.output/item['features']
        if not out.exists():
            with np.load(args.features/item['features']) as a:
                coarse=torch.from_numpy(a['coarse'].copy()).long()[None]
                teacher_h=a['h'].copy();expected=a['fine'].copy()
            seq=tokenize_sequence(sequences[item['dataset']][item['id']],tokenizer)[None]
            sigma=model.noise(torch.full((1,1),1e-5))[0]
            raw_coarse_logits(model,seq,pad_coarse(coarse),sigma)
            coarse_h=captures['coarse_h'][0,1:-1].clone()
            fine=raw_fine_logits(model,seq,pad_coarse(coarse))
            prefix,skip=captures['prefix'],captures['skip']
            replay=tail(prefix,skip)
            if not torch.allclose(replay,fine,atol=1e-5,rtol=1e-5):raise ValueError('Frozen fine tail logits do not replay')
            if not np.array_equal(replay.argmax(-1)[0].numpy(),expected):raise ValueError('Frozen fine tail token replay differs')
            zero=tail(prefix,skip,torch.zeros(1,coarse.shape[1],128))
            if not torch.equal(zero,replay):raise ValueError('Zero initialized condition changes fine output')
            if index==0:print('FULL_SCALE_FINE_TAIL_AND_ZERO_CONDITION_REPLAY_PASSED',flush=True)
            np.savez_compressed(out,coarse_h=coarse_h.numpy(),prefix=prefix[0].numpy(),skip=skip[0].numpy(),
                teacher_h=teacher_h,coarse=coarse[0].numpy(),fine=expected)
        rows.append(item)
        if (index+1)%25==0:print('RESIDUAL_CONDITION_CACHE',index+1,flush=True)
    for h in hooks:h.remove()
    atomic_json(args.output/'manifest.json',dict(protocol=protocol,targets=rows))
    print('RESIDUAL_CONDITION_CACHE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
