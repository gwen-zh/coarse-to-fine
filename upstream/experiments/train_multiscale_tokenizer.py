"""Real frozen-decoder optimization for a bounded two-scale representation probe."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from evaluate_residual_vq import ProjectedDecoder, decode_embedding
from multiscale_tokenizer import TwoScaleTokenizer, pair_geometry, reconstruction_loss
from structural_metrics import aligned_rmsd, summarize_backbone
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


def decode_checkpointed(decoder, embedding):
    x=torch.cat((decoder.embed.weight[4098:4099],embedding,decoder.embed.weight[4097:4098]))[None]
    n=x.shape[1]; ids=torch.zeros((1,n),device=x.device,dtype=torch.long);mask=torch.zeros((1,n,n),device=x.device)
    for block in decoder.decoder_stack.blocks:
        def call(value, block=block): return block(value,ids,None,None,ids,mask)
        x=checkpoint(call,x,use_reentrant=False)
    x=decoder.decoder_stack.norm(x)
    _,bb=decoder.affine_output_projection(x,affine=None,affine_mask=torch.zeros((1,n),device=x.device,dtype=torch.bool))
    return bb[0,1:-1]


def verify_sources(protocol):
    for p,h in protocol['code_sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Code hash mismatch: '+p)
    for row in protocol['targets']:
        for key in ('tokens','reference'):
            if hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()!=row[key+'_sha256']:raise RuntimeError('Input hash mismatch')


def load_data(protocol,device,codes,decoder):
    data=[]
    for row in protocol['targets']:
        with np.load(row['tokens']) as z: ids=torch.as_tensor(z['fine'].copy(),device=device).long(); latent=torch.as_tensor(z['latent'].copy(),device=device)
        if not torch.equal(torch.cdist(latent.float(),codes.float()).argmin(-1),ids):raise RuntimeError('Fine IDs do not match original codebook')
        with np.load(row['reference']) as z: bb=torch.as_tensor(z['backbone'].copy(),device=device)
        pair,mask=pair_geometry(bb)
        data.append(dict(row=row,bb=bb,pair=pair,mask=mask,z=codes[ids].detach(),embedding=decoder.embed(ids).detach()))
    return data


def metrics(bb,reference):
    x,y=bb.detach().cpu().numpy(),reference.detach().cpu().numpy()
    if not np.isfinite(x).all():raise RuntimeError('Nonfinite decoded structure')
    raw=summarize_backbone(x); n=len(x); i,j=np.triu_indices(n,12)
    distances=lambda a:np.linalg.norm(a[i,1]-a[j,1],axis=-1)
    pred,true=distances(x),distances(y)
    return dict(ca_rmsd=aligned_rmsd(x[:,1],y[:,1]),cn_rate=raw['backbone_cn_outliers']/max(n-1,1),
                ca_clashes_per_residue=raw['upstream_ca_clashes']/n,
                long_range_distance_mae=float(np.abs(pred-true).mean()) if len(i) else None,
                quality=measurements(x,y))


@torch.no_grad()
def evaluate(model,decoder,data,scale,directory,epoch,ablations):
    model.eval();rows=[];hist=torch.zeros(32,dtype=torch.long)
    for item in data:
        z,ids,_=model.quantizer(model.encode(item['z'],item['pair'],item['mask']))
        hist+=torch.bincount(ids.cpu(),minlength=32)
        original=decode_embedding(decoder,item['embedding'])
        row=dict(target=item['row']['target'],methods={'original':metrics(original,item['bb'])})
        arrays={'original':original.cpu().numpy(),'coarse_ids':ids.cpu().numpy()}
        features={'full':z}
        if ablations:
            g=torch.Generator().manual_seed(20261026)
            permutation=torch.randperm(len(z),generator=g).to(z.device)
            features.update(zero_coarse=torch.zeros_like(z),shuffled_coarse=z[permutation])
        for name,coarse in features.items():
            emb=model.fuse(item['z'],item['embedding'],coarse,scale)
            bb=decode_embedding(decoder,emb)
            row['methods'][name]=metrics(bb,item['bb']); arrays[name]=bb.cpu().numpy()
        rows.append(row)
        if ablations:np.savez_compressed(directory/(row['target']+'.npz'),**arrays)
    methods={}
    for key in rows[0]['methods']:
        methods[key]={metric:float(np.mean([r['methods'][key][metric] for r in rows])) for metric in ('ca_rmsd','cn_rate','ca_clashes_per_residue','long_range_distance_mae')}
    probs=hist.float()/hist.sum();p=probs[probs>0]
    result=dict(status='complete',epoch=epoch,mode=model.mode,proteins=len(rows),methods=methods,code_usage=int((hist>0).sum()),code_perplexity=float(torch.exp(-(p*p.log()).sum())),code_counts=hist.tolist(),targets=rows)
    atomic_json(directory/('evaluation.json' if ablations else f'epoch{epoch}_evaluation.json'),result)
    print('EVALUATION',model.mode,epoch,json.dumps(methods),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['smoke','train'],required=True);p.add_argument('--arm',choices=['local','global'],default='global');a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());verify_sources(protocol)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU required for real decoder probe')
    device=torch.device('cuda');started=time.monotonic()
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol['weights'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.to(device).eval().requires_grad_(False)
    codes=torch.load(protocol['codes'],map_location='cpu',weights_only=True)['fine_codes'].float().to(device)
    scale=decoder.embed.weight[:4096].std(0).detach()
    # Keep dataset CPU-backed to avoid retaining N^2 tensors for all proteins on GPU.
    data=load_data(protocol,torch.device('cpu'),codes.cpu(),decoder.cpu());decoder.to(device)
    def move(item):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in item.items()}
    training=[d for d in data if d['row']['subset']=='train'];validation=[d for d in data if d['row']['subset']=='validation']
    arms=protocol['arms'] if a.stage=='smoke' else [a.arm];smokes={}
    for arm in arms:
        torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
        model=TwoScaleTokenizer(arm).to(device)
        with torch.no_grad():
            initial=[model.encode(d['z'],d['pair'],d['mask']) for d in map(move,training[:8])]
            model.quantizer.initialize(torch.cat(initial));del initial
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=1e-4)
        directory=a.run/(('smoke_'+arm) if a.stage=='smoke' else arm);directory.mkdir(exist_ok=True)
        item=move(training[0]);model.eval()
        with torch.no_grad():
            emb,_,_=model(item['z'],item['embedding'],item['pair'],item['mask'],scale)
            if not torch.equal(emb,item['embedding']):raise RuntimeError('Epoch0 not original fine embedding')
            original=decode_embedding(decoder,emb)
        emb=emb.detach().requires_grad_();check=decode_checkpointed(decoder,emb)
        torch.testing.assert_close(check,original,atol=2e-5,rtol=2e-5)
        check.square().mean().backward()
        if emb.grad is None or not torch.isfinite(emb.grad).all() or emb.grad.norm()==0:raise RuntimeError('Decoder gradient missing')
        del emb,check,original,item
        schedule=[]
        rng=np.random.default_rng(protocol['seed'])
        epochs=1 if a.stage=='smoke' else protocol['epochs']
        trace=[];gradient_norm=None
        for epoch in range(1,epochs+1):
            order=[0,1] if a.stage=='smoke' else rng.permutation(len(training)).tolist()
            model.train();losses=[];epoch_start=time.monotonic()
            for step,index in enumerate(order):
                item=move(training[index]);optimizer.zero_grad(set_to_none=True)
                embedding,ids,commit=model(item['z'],item['embedding'],item['pair'],item['mask'],scale)
                prediction=decode_checkpointed(decoder,embedding)
                reconstruction,parts=reconstruction_loss(prediction,item['bb']);loss=reconstruction+commit
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
                # On step2 verify reconstruction gradients, separately from VQ commitment.
                if epoch==1 and step==1:
                    grads=torch.autograd.grad(reconstruction,tuple(model.input.parameters()),retain_graph=True,allow_unused=True)
                    gradient_norm=float(sum(g.square().sum() for g in grads if g is not None).sqrt())
                    if not np.isfinite(gradient_norm) or gradient_norm<=0:raise RuntimeError('No geometry gradient to new encoder')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite model gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss));trace.append(dict(epoch=epoch,step=step,target=item['row']['target'],loss=float(loss),reconstruction=float(reconstruction),commitment=float(commit)))
                if step%16==0:print('TRAIN',arm,epoch,step,'loss',float(loss),'elapsed',time.monotonic()-started,flush=True)
                del embedding,prediction,reconstruction,commit,loss,parts,item
            atomic_json(directory/'progress.json',dict(epoch=epoch,mean_loss=float(np.mean(losses)),epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started))
            if a.stage=='train':
                torch.save(dict(model=model.state_dict(),epoch=epoch,mode=arm,protocol_sha256=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest()),directory/'final.pt')
        if any(p.requires_grad or p.grad is not None for p in decoder.parameters()):raise RuntimeError('Decoder not frozen')
        atomic_json(directory/'training.json',dict(trace=trace,encoder_reconstruction_gradient_norm=gradient_norm,parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name()))
        if a.stage=='smoke':
            smokes[arm]=dict(status='passed',encoder_reconstruction_gradient_norm=gradient_norm,parameters=sum(p.numel() for p in model.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated())
        else:
            evaluate(model,decoder,list(map(move,validation)),scale,directory,epochs,True)
            atomic_json(directory/'complete.json',dict(status='complete',elapsed_seconds=time.monotonic()-started,scope=protocol['scope']))
        del model,optimizer
    if a.stage=='smoke':atomic_json(a.run/'smoke.json',dict(status='passed',arms=smokes,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
