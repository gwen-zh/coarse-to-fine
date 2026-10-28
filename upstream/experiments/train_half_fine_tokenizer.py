"""Half-fine reconstruction, no-coarse/local/global matched development arms."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from train_multiscale_tokenizer import decode_checkpointed,load_data,verify_sources,metrics
from multiscale_tokenizer import reconstruction_loss
from half_fine_tokenizer import HalfFineTokenizer,retained_positions,sanitize_fine,hidden_embedding_loss
from diagnostic_io import atomic_json


@torch.no_grad()
def evaluate(model,decoder,data,scale,mean_embedding,directory):
    model.eval();rows=[];hist=torch.zeros(32,dtype=torch.long)
    for item in data:
        keep=retained_positions(len(item['z']),item['z'].device)
        coarse,ids,_=model.coarse(item['z'],item['pair'],item['mask'])
        _,base=sanitize_fine(item['z'],item['embedding'],keep,mean_embedding)
        embeddings={'original':item['embedding'],'half_untrained':base,
            'trained':model.fusion(item['z'],item['embedding'],keep,coarse,scale,mean_embedding)}
        if ids is not None:
            hist+=torch.bincount(ids.cpu(),minlength=32)
            generator=torch.Generator().manual_seed(20261026)
            permutation=torch.randperm(len(coarse),generator=generator).to(coarse.device)
            embeddings['zero_coarse']=model.fusion(item['z'],item['embedding'],keep,torch.zeros_like(coarse),scale,mean_embedding)
            embeddings['shuffled_coarse']=model.fusion(item['z'],item['embedding'],keep,coarse[permutation],scale,mean_embedding)
        row=dict(target=item['row']['target'],length=len(keep),retained=int(keep.sum()),
                 bits_per_residue=(12*int(keep.sum())+(5*len(keep) if ids is not None else 0))/len(keep),methods={})
        arrays={'retained':keep.cpu().numpy()}
        if ids is not None:arrays['coarse_ids']=ids.cpu().numpy()
        for name,embedding in embeddings.items():
            bb=decode_embedding(decoder,embedding)
            row['methods'][name]=metrics(bb,item['bb']);arrays[name]=bb.cpu().numpy()
            row['methods'][name]['hidden_embedding_mse']=float(hidden_embedding_loss(embedding,item['embedding'],keep,scale))
        rows.append(row);np.savez_compressed(directory/(row['target']+'.npz'),**arrays)
    methods={}
    for name in rows[0]['methods']:
        methods[name]={metric:float(np.mean([r['methods'][name][metric] for r in rows])) for metric in ('ca_rmsd','cn_rate','ca_clashes_per_residue','long_range_distance_mae','hidden_embedding_mse')}
    p=hist.float()/max(int(hist.sum()),1);p=p[p>0]
    result=dict(status='complete',mode=model.mode,proteins=len(rows),methods=methods,code_usage=int((hist>0).sum()),code_perplexity=float(torch.exp(-(p*p.log()).sum())) if len(p) else None,code_counts=hist.tolist(),targets=rows)
    atomic_json(directory/'evaluation.json',result)
    print('EVALUATION',model.mode,json.dumps(methods),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['smoke','train'],required=True);p.add_argument('--arm',choices=['none','local','global'],default='global');a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());verify_sources(protocol)
    for path,expected in protocol['source_sha256'].items():
        digest=hashlib.sha256()
        with Path(path).open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=expected:raise RuntimeError('Frozen weight hash mismatch')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    started=time.monotonic();device=torch.device('cuda')
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol['weights'],map_location='cpu',mmap=True,weights_only=False);decoder.load_state_dict(state.get('decode_proj',state),strict=True);del state
    decoder.eval().requires_grad_(False)
    codes=torch.load(protocol['codes'],map_location='cpu',weights_only=True)['fine_codes'].float()
    scale=decoder.embed.weight[:4096].std(0).detach().to(device);mean_embedding=decoder.embed.weight[:4096].mean(0).detach().to(device)
    data=load_data(protocol,torch.device('cpu'),codes,decoder);decoder.to(device)
    def move(item):return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in item.items()}
    training=[d for d in data if d['row']['subset']=='train'];validation=[d for d in data if d['row']['subset']=='validation']
    arms=protocol['arms'] if a.stage=='smoke' else [a.arm];smokes={}
    for arm in arms:
        torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
        model=HalfFineTokenizer(arm).to(device)
        if arm!='none':
            with torch.no_grad():
                features=[model.encode(d['z'],d['pair'],d['mask']) for d in map(move,training[:8])]
                model.quantizer.initialize(torch.cat(features));del features
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=protocol['lr'],weight_decay=1e-4)
        directory=a.run/(('smoke_'+arm) if a.stage=='smoke' else arm);directory.mkdir(exist_ok=True)
        item=move(training[0]);model.eval()
        with torch.no_grad():
            embedding,_,_=model(item['z'],item['embedding'],item['pair'],item['mask'],scale,mean_embedding)
            keep=retained_positions(len(embedding),device)
            _,base=sanitize_fine(item['z'],item['embedding'],keep,mean_embedding)
            if not torch.equal(embedding,base):raise RuntimeError('Initial hidden replacement mismatch')
            baseline=decode_embedding(decoder,embedding)
        embedding=embedding.detach().requires_grad_();check=decode_checkpointed(decoder,embedding)
        torch.testing.assert_close(check,baseline,atol=2e-5,rtol=2e-5)
        check.square().mean().backward()
        if not torch.isfinite(embedding.grad).all() or embedding.grad.norm()==0:raise RuntimeError('Decoder gradient missing')
        del item,embedding,base,baseline,check
        rng=np.random.default_rng(protocol['seed']);trace=[];gradient_norm=None
        epochs=1 if a.stage=='smoke' else protocol['epochs']
        for epoch in range(1,epochs+1):
            order=[0,1] if a.stage=='smoke' else rng.permutation(len(training)).tolist()
            model.train();epoch_start=time.monotonic();records=[]
            for step,index in enumerate(order):
                item=move(training[index]);keep=retained_positions(len(item['z']),device);optimizer.zero_grad(set_to_none=True)
                embedding,ids,commit=model(item['z'],item['embedding'],item['pair'],item['mask'],scale,mean_embedding)
                prediction=decode_checkpointed(decoder,embedding)
                reconstruction,parts=reconstruction_loss(prediction,item['bb'])
                distillation=hidden_embedding_loss(embedding,item['embedding'],keep,scale)
                loss=protocol['geometry_weight']*reconstruction+distillation+commit
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                if epoch==1 and step==1:
                    target=tuple(model.fusion.input.parameters()) if arm=='none' else tuple(model.input.parameters())
                    grads=torch.autograd.grad(reconstruction,target,retain_graph=True,allow_unused=True)
                    gradient_norm=float(sum(g.square().sum() for g in grads if g is not None).sqrt())
                    if not np.isfinite(gradient_norm) or gradient_norm<=0:raise RuntimeError('No real geometry gradient to adapter')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                row=dict(epoch=epoch,step=step,target=item['row']['target'],loss=float(loss),reconstruction=float(reconstruction),hidden_embedding_mse=float(distillation),commitment=float(commit))
                records.append(row);trace.append(row)
                if step%32==0:print('TRAIN',arm,epoch,step,'loss',row['loss'],'reconstruction',row['reconstruction'],'elapsed',time.monotonic()-started,flush=True)
                del item,embedding,prediction,reconstruction,distillation,commit,loss,parts
            atomic_json(directory/'progress.json',dict(epoch=epoch,mean_loss=float(np.mean([r['loss'] for r in records])),mean_reconstruction=float(np.mean([r['reconstruction'] for r in records])),epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started))
            if a.stage=='train':torch.save(dict(model=model.state_dict(),mode=arm,epoch=epoch,protocol_sha256=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest()),directory/'final.pt')
        if any(p.requires_grad or p.grad is not None for p in decoder.parameters()):raise RuntimeError('Decoder unfrozen')
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        atomic_json(directory/'training.json',dict(trace=trace,real_reconstruction_gradient_norm=gradient_norm,trainable_parameters=parameters,peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-started,device=torch.cuda.get_device_name()))
        if a.stage=='smoke':smokes[arm]=dict(status='passed',real_reconstruction_gradient_norm=gradient_norm,trainable_parameters=parameters,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        else:
            evaluate(model,decoder,list(map(move,validation)),scale,mean_embedding,directory)
            atomic_json(directory/'complete.json',dict(status='complete',elapsed_seconds=time.monotonic()-started,scope=protocol['scope']))
        del optimizer,model
    if a.stage=='smoke':atomic_json(a.run/'smoke.json',dict(status='passed',arms=smokes,elapsed_seconds=time.monotonic()-started))

if __name__=='__main__':main()
