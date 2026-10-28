"""Sequence-only paired coarse inference then fixed residual refinement."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_pretrained_coarse import (load_state_dict_from_lightning_ckpt,initialize_new_coarse_semantics,install_coarse_lora,EsmSequenceTokenizer,tokenize_sequence,sample_coarse,generation_seed)
from coarse_lora import trainable_state,restore_trainable_state
from depth_generator import DepthGenerator,sequence_ids
from diagnostic_io import atomic_json
from holdout_confirmation import load_locked


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--index',type=int,required=True);parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args();torch.set_num_threads(16);torch.manual_seed(0)
    protocol,digest=load_locked(args.run)
    if not 0<=args.index<protocol['chunks']:raise ValueError('Invalid chunk')
    targets=protocol['targets'][args.index::protocol['chunks']]
    if args.preflight_only:
        manifest=json.loads(Path(protocol['inputs']['manifest']['path']).read_text())
        targets=[min((r for r in manifest['all_groups'] if r['split']=='development'),key=lambda r:(r['length'],r['sequence_hash']))]
    output=args.run/('preflight' if args.preflight_only else 'coarse');features=output/'features';predictions=output/'predictions';features.mkdir(parents=True,exist_ok=True);predictions.mkdir(exist_ok=True)
    model=load_state_dict_from_lightning_ckpt(Path(protocol['inputs']['generator']['path']),device=torch.device('cpu'));model.eval();tokenizer=EsmSequenceTokenizer();head=model.net.output_heads.fine_structure_head[-1]
    captured=[];hook=head.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()))
    for target in targets:
        path=features/(target['id']+'.npz')
        if path.exists():continue
        sequence=tokenize_sequence(target['sequence'],tokenizer)[None];n=target['length'];seed=(protocol['coarse']['context_seed']+int(target['sequence_hash'][:8],16))%(2**32)
        torch.manual_seed(seed);np.random.seed(seed);captured.clear();fine,coarse=model.ddpm_sample(sequence,lengths=torch.tensor([n]),num_steps=5,return_coarse=True);h=captured[-1][0,1:n+1].numpy().copy()
        if not torch.equal(F.linear(torch.from_numpy(h),head.weight[:4096],head.bias[:4096]).argmax(-1),fine[0,1:n+1]):raise ValueError('Original context fine replay failed')
        np.savez_compressed(path,sampled_h=h,native_fine=fine[0,1:n+1].numpy(),native_coarse=coarse[0,1:n+1].numpy());print('HOLDOUT_SEQUENCE_CONTEXT_COMPLETE',target['id'],flush=True)
    hook.remove();captured.clear();bundle=torch.load(protocol['inputs']['residual_bridge']['path'],map_location='cpu',weights_only=True)
    encoder=torch.load(protocol['inputs']['encoder']['path'],map_location='cpu',weights_only=False,mmap=True);old=encoder.get('encoder_weight',encoder)['small_codebook.embeddings'].clone();del encoder
    initialization=json.loads(Path(protocol['inputs']['semantic_initialization']['path']).read_text());initialize_new_coarse_semantics(model,bundle['centers'][0],old,initialization=initialization);del old,bundle
    adapter=torch.load(protocol['inputs']['coarse_adapter']['path'],map_location='cpu',weights_only=True)
    if adapter['epoch']!=3 or adapter['initialization']!=initialization:raise ValueError('Fixed adapter initialization mismatch')
    install_coarse_lora(model,adapter['protocol']['last_blocks'],adapter['protocol']['rank']);model.eval();initial=trainable_state(model)
    if set(initial)!=set(adapter['state']):raise ValueError('Incomplete adapted state')
    if any(torch.count_nonzero(v) for key,v in initial.items() if key.endswith('.b')):raise ValueError('Initial control LoRA must be zero')
    refiner=torch.load(protocol['inputs']['refiner']['path'],map_location='cpu',weights_only=True);cfg=refiner['protocol']
    if cfg['token_manifest_sha256']!=protocol['inputs']['source_token_manifest']['sha256']:raise ValueError('Refiner semantics mismatch')
    small=DepthGenerator(sizes=cfg['sizes'],width=cfg['width'],layers=cfg['layers']);small.load_state_dict(refiner['state_dict'],strict=True);small.eval().requires_grad_(False)
    for arm,state in [('control',initial),('adapted',adapter['state'])]:
        restore_trainable_state(model,state)
        parameters=dict(model.named_parameters())
        for name,value in state.items():
            if not torch.equal(parameters[name],value):raise ValueError('State restore differs')
        for target in targets:
            path=predictions/(arm+'_'+target['id']+'.npz')
            if path.exists():continue
            seq=tokenize_sequence(target['sequence'],tokenizer)[None]
            with np.load(features/(target['id']+'.npz')) as a:h=torch.from_numpy(a['sampled_h'].copy())[None]
            coarse=[];tokens=[]
            for sample in range(protocol['coarse']['samples']):
                plan=sample_coarse(model,seq,generation_seed(target['id'],sample),steps=protocol['coarse']['steps']);result,calls=small.refine_coarse(h,sequence_ids(target['sequence'])[None],plan)
                if not torch.equal(result[...,0],plan):raise ValueError('Refiner changed coarse')
                coarse.append(plan[0].numpy());tokens.append(result[0].numpy())
            np.savez_compressed(path,coarse=np.stack(coarse),tokens=np.stack(tokens));print('HOLDOUT_COARSE_SAMPLED',arm,target['id'],flush=True)
    atomic_json(output/f'sampling_chunk{args.index}.json',dict(protocol_sha256=digest,targets=[r['id'] for r in targets],device='cpu',dtype='float32',samples=2))


if __name__=='__main__':main()
