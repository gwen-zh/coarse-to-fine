"""Fixed-budget matched representation experiment, actual hard fine decoding."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from train_multiscale_tokenizer import decode_checkpointed,metrics,verify_sources
from multiscale_tokenizer import reconstruction_loss
from multiscale_discrete import MultiScaleDiscrete,geometry_inputs,hard_embeddings,retained_positions,nominal_rate
from diagnostic_io import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train'],required=True);parser.add_argument('--index',type=int,default=0);a=parser.parse_args()
    run=Path('artifacts/development/multiscale_discrete_v1');p=json.loads((run/'protocol.json').read_text());verify_sources(p)
    if a.stage=='train' and json.loads((run/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke prerequisite failed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda');started=time.perf_counter()
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(p['weights'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state;decoder.to(device).eval().requires_grad_(False)
    codes=torch.load(p['codes'],map_location='cpu',weights_only=True)['fine_codes'].float().to(device);code_scale=codes.std(0).clamp_min(1e-5)
    data=[]
    for row in p['targets']:
        with np.load(row['tokens']) as x:ids=torch.tensor(x['fine'],dtype=torch.long)
        with np.load(row['reference']) as x:bb=torch.tensor(x['backbone'],dtype=torch.float32)
        intrinsic,pair,masks=geometry_inputs(bb);data.append(dict(row=row,ids=ids,bb=bb,intrinsic=intrinsic,pair=pair,masks=masks))
    train=[x for x in data if x['row']['subset']=='train'];val=[x for x in data if x['row']['subset']=='validation']
    def move(x):return {k:({s:t.to(device) for s,t in v.items()} if k=='masks' else v.to(device) if torch.is_tensor(v) else v) for k,v in x.items()}
    configurations=[(p['seeds'][0],m) for m in p['arms']] if a.stage=='smoke' else [(p['seeds'][a.index//3],p['arms'][a.index%3])]
    checks={}
    for seed,mode in configurations:
        torch.manual_seed(seed);np.random.seed(seed);model=MultiScaleDiscrete(mode).to(device)
        directory=run/(('smoke_'+mode) if a.stage=='smoke' else f'{mode}_{seed}');directory.mkdir(exist_ok=True)
        if a.stage=='train' and (directory/'training.json').exists():raise RuntimeError('Existing training run; no silent overwrite')
        with torch.no_grad():
            for i,b in enumerate(model.branches):
                values=[]
                for item in train[:8]:
                    x=move(item);values.append(b.encode(x['intrinsic'],x['pair'],x['masks'][model.scales[i]]))
                b.vq.initialize(torch.cat(values));del values
        optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=1e-4);rng=np.random.default_rng(seed);trace=[];gradient=None
        epochs=1 if a.stage=='smoke' else p['epochs']
        for epoch in range(epochs):
            model.train();order=[0,1] if a.stage=='smoke' else rng.permutation(len(train)).tolist()
            for step,index in enumerate(order):
                x=move(train[index]);keep=retained_positions(len(x['ids']),device);visible=torch.where(keep,x['ids'],0);optimizer.zero_grad(set_to_none=True)
                q,coarse_ids,commit=model.encode(x['intrinsic'],x['pair'],x['masks']);logits=model.complete(visible,keep,q,codes)
                embedding,output_ids=hard_embeddings(logits,visible,keep,decoder.embed.weight[:4096])
                if not torch.equal(embedding.detach(),decoder.embed(output_ids)):raise RuntimeError('Non-discrete decoder forward')
                if a.stage=='smoke' and step==0:
                    with torch.no_grad():baseline=decode_embedding(decoder,embedding.detach())
                    check=decode_checkpointed(decoder,embedding);torch.testing.assert_close(check.detach(),baseline,atol=2e-5,rtol=2e-5);prediction=check
                else:prediction=decode_checkpointed(decoder,embedding)
                geometry,_=reconstruction_loss(prediction,x['bb']);ce=F.cross_entropy(logits[~keep],x['ids'][~keep])
                expected=logits[~keep].softmax(-1)@codes;latent_loss=((expected-codes[x['ids'][~keep]])/code_scale).square().mean()
                loss=ce+p['latent_weight']*latent_loss+p['geometry_weight']*geometry+commit
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                if a.stage=='smoke' and step==1:
                    params=[b.input.weight for b in model.branches];g=torch.autograd.grad(geometry,params,retain_graph=True,allow_unused=True);gradient=[float(v.norm()) if v is not None else 0. for v in g]
                    if not all(np.isfinite(v) and v>0 for v in gradient):raise RuntimeError('Missing real decoder gradient to a scale branch')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                optimizer.step();trace.append(dict(epoch=epoch+1,step=step,ce=float(ce),latent_loss=float(latent_loss),geometry=float(geometry),loss=float(loss)))
                if step%32==0:print('MULTISCALE_DISCRETE_TRAIN',seed,mode,epoch+1,step,flush=True)
            atomic_json(directory/'progress.json',dict(epoch=epoch+1,updates=len(trace),elapsed_seconds=time.perf_counter()-started))
        if any(v.grad is not None or v.requires_grad for v in decoder.parameters()):raise RuntimeError('Decoder changed')
        count=sum(v.numel() for v in model.parameters());atomic_json(directory/'training.json',dict(status='complete',trace=trace,seed=seed,mode=mode,parameters=count,decoder_branch_gradient_norms=gradient,peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started))
        if a.stage=='smoke':checks[mode]=dict(status='passed',parameters=count,decoder_branch_gradient_norms=gradient);continue
        torch.save(dict(model=model.state_dict(),mode=mode,seed=seed,protocol_sha256=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()),directory/'final.pt')
        model.eval();rows=[];counts=torch.zeros(3,4,dtype=torch.long)
        with torch.no_grad():
            for item in val:
                x=move(item);keep=retained_positions(len(x['ids']),device);visible=torch.where(keep,x['ids'],0);q,ids,_=model.encode(x['intrinsic'],x['pair'],x['masks']);n=len(keep)
                for b in range(3):counts[b]+=torch.bincount(ids[:,b].cpu(),minlength=4)
                features={'intact':q};rng_t=torch.Generator(device=device).manual_seed(2026);perm=torch.randperm(n,device=device,generator=rng_t);features['shuffled_all']=q[perm]
                if mode=='mixed':
                    for b in range(3):
                        v=q.clone();v[:,16*b:16*(b+1)]=q[perm,16*b:16*(b+1)];features[f'shuffled_branch{b}']=v
                baseline=decode_embedding(decoder,decoder.embed(x['ids']));row=dict(target=x['row']['target'],length=n,bits_per_residue=nominal_rate(n),baseline=metrics(baseline,x['bb']),methods={});arrays={'baseline':baseline.cpu().numpy(),'coarse_ids':ids.cpu().numpy()}
                for name,coarse in features.items():
                    logits=model.complete(visible,keep,coarse,codes);embedding,fine=hard_embeddings(logits,visible,keep,decoder.embed.weight[:4096]);bb=decode_embedding(decoder,embedding)
                    row['methods'][name]=metrics(bb,x['bb']);row['methods'][name]['hidden_accuracy']=float((fine[~keep]==x['ids'][~keep]).float().mean());arrays[name]=bb.cpu().numpy();arrays[name+'_fine']=fine.cpu().numpy()
                rows.append(row);np.savez_compressed(directory/(row['target']+'.npz'),**arrays);print('MULTISCALE_DISCRETE_EVAL',mode,seed,row['target'],flush=True)
        atomic_json(directory/'evaluation.json',dict(status='complete',mode=mode,seed=seed,parameters=count,targets=rows,code_counts=counts.tolist(),scope=p['scope']))
    if a.stage=='smoke':
        if len(set(x['parameters'] for x in checks.values()))!=1:raise RuntimeError('Parameter mismatch')
        atomic_json(run/'smoke.json',dict(status='passed',arms=checks))

if __name__=='__main__':main()
