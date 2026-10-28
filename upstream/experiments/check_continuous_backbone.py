"""Real pretrained-weight interface audit; no fitting or quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from train_pretrained_coarse import (EsmSequenceTokenizer, tokenize_sequence,
                                    load_state_dict_from_lightning_ckpt)
from train_native_generator import pad_coarse
from native_generator import raw_fine_logits
from continuous_backbone import ContinuousBackbone
from train_residual_vq import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    protocol=json.loads((args.run/'protocol.json').read_text())
    for path,expected in protocol['sha256'].items():
        if digest(path)!=expected:raise RuntimeError('Changed frozen input: '+path)
    torch.set_num_threads(2);torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    started=perf_counter();device=torch.device('cuda')
    model=load_state_dict_from_lightning_ckpt(Path(protocol['checkpoint']),device=device).eval()
    manifest=json.loads(Path(protocol['manifest']).read_text())['targets']
    tokenizer=EsmSequenceTokenizer();items=[]
    with np.load(protocol['calibration']) as a:
        mean=torch.tensor(a['mean'],device=device,dtype=torch.float32)
        rms=torch.tensor(a['centered_rms'],device=device,dtype=torch.float32)
    if rms<=0:raise RuntimeError('Invalid normalization')
    for target in protocol['targets']:
        entry=next(x for x in manifest if x['id']==target)
        if entry['subset']!='train':raise RuntimeError('Interface audit must use training only')
        seq=tokenize_sequence(entry['sequence'],tokenizer).to(device)[None]
        with np.load(Path(protocol['prefixes'])/(target+'.npz')) as a:
            coarse=pad_coarse(torch.from_numpy(a['coarse'].copy()).long().to(device)[None])
        with np.load(Path(protocol['latents'])/(target+'.npz')) as a:
            latent=torch.tensor(a['latent'],device=device,dtype=torch.float32)[None]
        normalized=(latent-mean)/rms
        roundtrip=(normalized*rms+mean-latent).abs().max().item()
        if roundtrip>1e-5:raise RuntimeError('Normalization roundtrip failed')
        with torch.no_grad():baseline=raw_fine_logits(model,seq,coarse).clone()
        items.append((target,seq,coarse,normalized,baseline,roundtrip))
    adapter=ContinuousBackbone(model.net,last_blocks=4,rank=4).eval();rows=[]
    for target,seq,coarse,truth,baseline,roundtrip in items:
        state=torch.randn_like(truth);t=torch.tensor([.5],device=device)
        with torch.no_grad():
            disabled=adapter(seq,coarse,state,t,enabled=False)['native_logits']
            if not torch.equal(disabled,baseline):raise RuntimeError('Disabled native logits not bit-exact')
            first=adapter(seq,coarse,state,t)['hidden']
            changed=adapter(seq,coarse,state+.1*torch.randn_like(state),t)['hidden']
            state_effect=(first-changed).square().mean().sqrt().item()
            time_effect=(first-adapter(seq,coarse,state,t+.1)['hidden']).square().mean().sqrt().item()
            delta=adapter.input_adapter(state,t)
            special_zero=bool((delta[:,[0,-1]]==0).all())
        adapter.zero_grad(set_to_none=True);state=state.detach().requires_grad_(True)
        output=adapter(seq,coarse,state,t)
        # A nonzero diagnostic readout tests trunk connectivity independently of the zero velocity head.
        probe=(output['hidden']*torch.randn_like(output['hidden'])).mean()
        probe.backward()
        grads=dict(state=float(state.grad.norm()),input_projection=float(adapter.input_adapter.state.weight.grad.norm()),
                   time_projection=float(adapter.input_adapter.time[-1].weight.grad.norm()),
                   lora_b=[float(x.b.grad.norm()) if x.b.grad is not None else 0. for x in adapter.scoped_loras])
        if not all(np.isfinite(x) and x>0 for x in [grads['state'],grads['input_projection'],grads['time_projection'],*grads['lora_b']]):
            raise RuntimeError('Missing or nonfinite gradient: '+str(grads))
        if state_effect<=0 or time_effect<=0 or not special_zero:raise RuntimeError('Injection audit failed')
        rows.append(dict(target=target,length=truth.shape[1],disabled_logits_exact=True,normalization_max_error=roundtrip,
                         state_hidden_rms=state_effect,time_hidden_rms=time_effect,special_token_injection_zero=special_zero,diagnostic_gradient_norms=grads))
        del output,probe,first,changed,disabled
        adapter.zero_grad(set_to_none=True)
    # Prove adapters cannot leak into native calls even when LoRA B is no longer zero.
    with torch.no_grad():
        for layer in adapter.scoped_loras:layer.b.normal_(std=.001)
        for target,seq,coarse,state,baseline,_ in items:
            outside=raw_fine_logits(model,seq,coarse)
            disabled=adapter(seq,coarse,state,torch.tensor([.5],device=device),enabled=False)['native_logits']
            if not torch.equal(outside,baseline) or not torch.equal(disabled,baseline):raise RuntimeError('LoRA scope leaked into native path')
        for layer in adapter.scoped_loras:layer.b.zero_()
    result=dict(status='passed',purpose='Interface/gradient audit only; no optimizer updates, no generation improvement claim',
                protocol_sha256=digest(args.run/'protocol.json'),rows=rows,native_after_nonzero_lora_exact=True,
                trainable_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
                gpu=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=perf_counter()-started)
    atomic_json(args.run/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
