"""Bounded two-seed FM learnability probe on eight training chains."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from train_pretrained_coarse import EsmSequenceTokenizer,tokenize_sequence,load_state_dict_from_lightning_ckpt
from train_native_generator import pad_coarse
from native_generator import raw_fine_logits
from continuous_backbone import ContinuousBackbone
from diagnostic_io import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed-index',type=int,required=True);args=parser.parse_args()
    run=Path('artifacts/development/continuous_backbone_probe_v1');protocol=json.loads((run/'protocol.json').read_text());seed=protocol['seeds'][args.seed_index]
    for p,h in protocol['sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Changed frozen input: '+p)
    if json.loads(Path(protocol['interface_result']).read_text())['status']!='passed':raise RuntimeError('Interface prerequisite failed')
    torch.set_num_threads(2);torch.manual_seed(seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda');out=run/('seed'+str(seed));out.mkdir(exist_ok=True)
    if (out/'history.json').exists():raise RuntimeError('Existing run; explicit resume audit required')
    model=load_state_dict_from_lightning_ckpt(Path(protocol['generator']),device=device).eval();tokenizer=EsmSequenceTokenizer()
    with np.load(protocol['calibration']) as a:
        mean=torch.tensor(a['mean'],device=device,dtype=torch.float32);scale=float(a['centered_rms']);codes=torch.tensor(a['fine_codes'],device=device,dtype=torch.float32)
    code_norm=codes.square().sum(-1)
    def snap(x):return (x.square().sum(-1,keepdim=True)-2*x@codes.T+code_norm).argmin(-1)
    items=[]
    for row in protocol['targets']:
        name=row['id'];seq=tokenize_sequence(row['sequence'],tokenizer).to(device)[None]
        with np.load(Path(protocol['prefixes'])/(name+'.npz')) as a:coarse=pad_coarse(torch.tensor(a['coarse'],device=device,dtype=torch.long)[None])
        with np.load(Path(protocol['tokens'])/(name+'.npz')) as a:
            latent=torch.tensor(a['latent'],device=device,dtype=torch.float32)[None];truth=torch.tensor(a['fine'],device=device,dtype=torch.long)
        if not torch.equal(snap(latent)[0],truth):raise RuntimeError('Original nearest-fine outlet mismatch')
        with torch.no_grad():native=raw_fine_logits(model,seq,coarse).argmax(-1)[0].cpu().numpy()
        items.append(dict(row=row,sequence=seq,coarse=coarse,z=(latent-mean)/scale,native=native))
    adapter=ContinuousBackbone(model.net,last_blocks=4,rank=4).eval();parameters=[p for p in adapter.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=protocol['lr'],weight_decay=.01)
    train=[x for x in items if x['row']['subset']=='train'];history=[];started=time.perf_counter()
    @torch.no_grad()
    def objective(item):
        rng=torch.Generator(device=device).manual_seed(int(hashlib.sha256(('probe-eval:'+item['row']['id']).encode()).hexdigest()[:8],16))
        noise=torch.randn(item['z'].shape,device=device,generator=rng);values=[]
        for value in [.1,.35,.65,.9]:
            t=torch.tensor([value],device=device);state=(1-value)*noise+value*item['z']
            pred=adapter(item['sequence'],item['coarse'],state,t)['velocity'];values.append(float((pred-(item['z']-noise)).square().mean()))
        return float(np.mean(values))
    @torch.no_grad()
    def snapshot(step):
        directory=out/('step'+str(step));directory.mkdir(exist_ok=True)
        scores=[]
        for item in items:
            name=item['row']['id'];predictions=[];tokens=[]
            for sample in range(2):
                rng=torch.Generator(device=device).manual_seed(int(hashlib.sha256(('probe-sample:'+name+':'+str(sample)).encode()).hexdigest()[:8],16))
                state=torch.randn(item['z'].shape,device=device,generator=rng)
                for k in range(protocol['ode_steps']):
                    t=torch.tensor([k/protocol['ode_steps']],device=device)
                    state=state+adapter(item['sequence'],item['coarse'],state,t)['velocity']/protocol['ode_steps']
                latent=state*scale+mean
                if not torch.isfinite(latent).all():raise RuntimeError('Nonfinite sampled latent')
                predictions.append(latent[0].cpu().numpy());tokens.append(snap(latent)[0].cpu().numpy())
            np.savez_compressed(directory/(name+'.npz'),latent=np.stack(predictions),fine=np.stack(tokens),native_fine=item['native'])
            scores.append(dict(target=name,subset=item['row']['subset'],fixed_noise_objective=objective(item)))
        history.append(dict(step=step,elapsed_seconds=time.perf_counter()-started,objectives=scores));atomic_json(out/'history.json',history)
        print('CONTINUOUS_BACKBONE_SNAPSHOT',seed,step,flush=True)
    snapshot(0)
    for step in range(1,protocol['steps']+1):
        item=train[(step-1)%len(train)];noise=torch.randn_like(item['z']);t=torch.rand(1,device=device)
        state=(1-t[:,None,None])*noise+t[:,None,None]*item['z'];target=item['z']-noise
        optimizer.zero_grad(set_to_none=True);pred=adapter(item['sequence'],item['coarse'],state,t)['velocity'];loss=(pred-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or norm<=0:raise RuntimeError('Invalid gradient')
        optimizer.step()
        if step%25==0:print('CONTINUOUS_BACKBONE_TRAIN',seed,step,float(loss),flush=True)
    snapshot(protocol['steps'])
    state={k:p.detach().cpu() for k,p in adapter.named_parameters() if p.requires_grad}
    torch.save(dict(state=state,seed=seed,step=protocol['steps'],protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()),out/'final_adapter.pt')
    atomic_json(out/'training.json',dict(status='complete',seed=seed,steps=protocol['steps'],trainable_parameters=sum(p.numel() for p in parameters),seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope=protocol['scope']))

if __name__=='__main__':main()
