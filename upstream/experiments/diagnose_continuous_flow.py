"""Frozen checkpoint diagnosis: teacher interpolation and16/64 Euler sensitivity."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from train_pretrained_coarse import EsmSequenceTokenizer,tokenize_sequence,load_state_dict_from_lightning_ckpt
from train_native_generator import pad_coarse
from continuous_backbone import ContinuousBackbone
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    run=Path('artifacts/development/continuous_diagnostic_v1');p=json.loads((run/'protocol.json').read_text());source=Path(p['source']);old=json.loads((source/'protocol.json').read_text())
    for path,digest in p['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed input '+path)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');model=load_state_dict_from_lightning_ckpt(Path(old['generator']),device=device).eval();adapter=ContinuousBackbone(model.net,last_blocks=4,rank=4).eval();tokenizer=EsmSequenceTokenizer()
    with np.load(old['calibration']) as a:mean=torch.tensor(a['mean'],device=device,dtype=torch.float32);scale=float(a['centered_rms']);codes=torch.tensor(a['fine_codes'],device=device,dtype=torch.float32)
    def snap(z):return (z.square().sum(-1,keepdim=True)-2*z@codes.T+codes.square().sum(-1)).argmin(-1)
    trainable={k:v for k,v in adapter.named_parameters() if v.requires_grad};records=[]
    for seed in p['seeds']:
        checkpoint=torch.load(source/f'seed{seed}/final_adapter.pt',map_location=device,weights_only=False)
        if checkpoint['protocol_sha256']!=hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest():raise RuntimeError('Checkpoint protocol mismatch')
        if set(checkpoint['state'])!=set(trainable):raise RuntimeError('Missing or extra adapter parameters')
        for k,v in trainable.items():v.copy_(checkpoint['state'][k])
        directory=run/f'seed{seed}';directory.mkdir(exist_ok=True)
        for target in p['targets']:
            name=target['id'];seq=tokenize_sequence(target['sequence'],tokenizer).to(device)[None]
            with np.load(Path(old['prefixes'])/(name+'.npz')) as a:coarse=pad_coarse(torch.tensor(a['coarse'],device=device,dtype=torch.long)[None])
            with np.load(Path(old['tokens'])/(name+'.npz')) as a:z=(torch.tensor(a['latent'],device=device,dtype=torch.float32)[None]-mean)/scale
            teacher=[];rng=torch.Generator(device=device).manual_seed(int(hashlib.sha256(('probe-eval:'+name).encode()).hexdigest()[:8],16));noise=torch.randn(z.shape,device=device,generator=rng)
            for t0 in p['teacher_times']:
                state=(1-t0)*noise+t0*z;t=torch.tensor([t0],device=device);v=adapter(seq,coarse,state,t)['velocity'];estimate=state+(1-t0)*v
                teacher.append(dict(t=t0,velocity_mse=float((v-(z-noise)).square().mean()),endpoint_mse=float((estimate-z).square().mean()),state_mse=float((state-z).square().mean())))
            arrays={};samples=[]
            for sample in range(p['samples_per_coarse']):
                for steps in p['ode_steps']:
                    rng=torch.Generator(device=device).manual_seed(int(hashlib.sha256(('probe-sample:'+name+':'+str(sample)).encode()).hexdigest()[:8],16));state=torch.randn(z.shape,device=device,generator=rng)
                    for k in range(steps):state=state+adapter(seq,coarse,state,torch.tensor([k/steps],device=device))['velocity']/steps
                    latent=(state*scale+mean)[0].cpu().numpy();ids=snap(state*scale+mean)[0].cpu().numpy()
                    if not np.isfinite(latent).all():raise RuntimeError('Nonfinite rollout')
                    if steps==16:
                        with np.load(source/f'seed{seed}/step200'/(name+'.npz')) as a:cached=a['latent'][sample];cached_ids=a['fine'][sample]
                        maxerror=float(abs(cached-latent).max())
                        if not np.allclose(cached,latent,rtol=1e-5,atol=1e-4) or not np.array_equal(ids,cached_ids):raise RuntimeError('Cached16-step replay mismatch')
                    else:maxerror=None
                    arrays[f'latent_{steps}_{sample}']=latent;arrays[f'fine_{steps}_{sample}']=ids
                    samples.append(dict(sample=sample,steps=steps,normalized_endpoint_mse=float((state-z).square().mean()),cached16_max_error=maxerror))
            np.savez_compressed(directory/(name+'.npz'),**arrays)
            records.append(dict(seed=seed,target=name,subset=target['subset'],teacher=teacher,samples=samples))
            atomic_json(run/'flow_audit.json',dict(status='running',records=records));print('FLOW_DIAGNOSTIC_TARGET',seed,name,flush=True)
    atomic_json(run/'flow_audit.json',dict(status='complete',records=records,scope=p['scope']))

if __name__=='__main__':main()
