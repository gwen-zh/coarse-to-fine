"""Paired fine-only backbone-LoRA CE versus CE+structural-latent alignment."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_pretrained_coarse import load_state_dict_from_lightning_ckpt,EsmSequenceTokenizer,tokenize_sequence,sample_coarse,generation_seed,trainable_state,restore_trainable_state
from train_native_generator import pad_coarse
from native_generator import raw_fine_logits,choose_fine_prefix
from repa_alignment import install_repa_adapters,alignment_loss
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['ce','repa'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(1);torch.manual_seed(20261026);torch.backends.cuda.matmul.allow_tf32=False
    a.run.mkdir(exist_ok=True,parents=True);out=a.run/(('smoke_' if a.smoke else '')+a.arm);out.mkdir(exist_ok=True);device=torch.device('cuda')
    protocol=json.loads((a.run/'protocol.json').read_text())
    for file,digest in protocol['code_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    model=load_state_dict_from_lightning_ckpt(Path(protocol['checkpoint']),device=device);model.eval();tokenizer=EsmSequenceTokenizer()
    rows=json.loads(Path(protocol['tokens'],'manifest.json').read_text())['targets'];names=protocol['geometry_targets'];train=[r for r in rows if r['subset']=='train'];val=[r for r in rows if r['subset']=='validation'];assert len(train)==899 and len(val)==101
    if a.smoke:train=train[:4];val=val[:2];names=[val[0]['id']]
    items={}
    for r in train+val:
        with np.load(Path(protocol['tokens'])/r['tokens']) as z:c=z['coarse'].copy();f=z['fine'].copy()
        with np.load(Path(protocol['latent_tokens'])/r['tokens']) as z:latent=z['latent'].copy();assert np.array_equal(z['fine'],f)
        with np.load(Path(protocol['prefixes'])/(r['id']+'.npz')) as z:sampled=z['coarse'].copy()
        assert len(c)==len(f)==len(latent)==len(sampled)==r['length']
        items[r['id']]=dict(row=r,sequence=tokenize_sequence(r['sequence'],tokenizer).to(device)[None],coarse=torch.from_numpy(c).long().to(device)[None],fine=torch.from_numpy(f).long().to(device)[None],latent=torch.from_numpy(latent).to(device)[None],sampled=torch.from_numpy(sampled).long().to(device)[None])
    # Draw original coarse before installing adapters; matched seeds for both arms.
    samples=1 if a.smoke else 8;originals={}
    for name in names:
        item=items[name];draws=[]
        for sample in range(samples):
            with torch.no_grad():
                coarse=sample_coarse(model,item['sequence'],generation_seed(name,sample))
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):fine=raw_fine_logits(model,item['sequence'],pad_coarse(coarse)).argmax(-1)
            draws.append((coarse.detach(),fine.cpu().numpy()[0]));print('REPA_ORIGINAL_DRAW',a.arm,name,sample,flush=True)
        originals[name]=draws
    # Reset initialization identically after all coarse sampling resets the RNG.
    torch.manual_seed(20261026);projector,configuration=install_repa_adapters(model);model.eval();capture={}
    handle=model.net.transformer.blocks[configuration['alignment_block_zero_based']].register_forward_hook(lambda module,args,output:capture.update(h=output[:,1:-1]))
    with np.load(protocol['calibration']) as z:mean=torch.from_numpy(z['mean'].astype(np.float32)).to(device);scale=float(z['centered_rms'])
    parameters=[v for v in model.parameters() if v.requires_grad]+list(projector.parameters());optimizer=torch.optim.AdamW(parameters,lr=protocol['lr']);coefficient=0. if a.arm=='ce' else protocol['alignment_weight'];rng=torch.Generator().manual_seed(20261026)
    def forward(item,prefix,seq=None,z=None,labels=None):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
            logits=raw_fine_logits(model,item['sequence'] if seq is None else seq,pad_coarse(prefix));prediction=projector(capture.pop('h'))
        ce=F.cross_entropy(logits.float().flatten(0,1),(item['fine'] if labels is None else labels).flatten());align=alignment_loss(prediction,item['latent'] if z is None else z,mean,scale)
        return ce,align
    def validate():
        values=[]
        with torch.no_grad():
            for r in val:
                item=items[r['id']];ce,align=forward(item,item['sampled']);values.append((ce.item(),align.item()))
        return dict(ce=float(np.mean([v[0] for v in values])),alignment=float(np.mean([v[1] for v in values])))
    # Verify alignment itself reaches upstream LoRA; head-only CE gradients do not establish this.
    item=items[train[0]['id']];ce,align=forward(item,item['coarse']);layer=configuration['alignment_block_zero_based'];upstream=model.net.transformer.blocks[layer].attn.out_proj.b
    gradient=torch.autograd.grad(align,upstream,retain_graph=False)[0];norm=float(gradient.float().norm());assert np.isfinite(norm) and norm>0
    atomic_json(out/'gradient_check.json',dict(alignment_to_upstream_lora_norm=norm,configuration=configuration));del gradient,ce,align
    initial=validate();best=initial['ce'];best_epoch=0;best_state=trainable_state(model);history=[dict(epoch=0,**initial)];atomic_json(out/'initial.json',dict(validation=initial,configuration=configuration,trainable_parameters=sum(v.numel() for v in parameters),alignment_weight=coefficient))
    def save_best():torch.save(dict(state=best_state,epoch=best_epoch,configuration=configuration,arm=a.arm,protocol=protocol),out/'best_adapter.pt')
    save_best();started=time.perf_counter();steps=0
    for epoch in range(1,(1 if a.smoke else protocol['epochs'])+1):
        losses=[]
        for index in torch.randperm(len(train),generator=rng).tolist():
            item=items[train[index]['id']];n=item['coarse'].shape[1];start=int(torch.randint(max(1,n-256+1),(),generator=rng));end=min(n,start+256);seq=torch.cat([item['sequence'][:,:1],item['sequence'][:,1+start:1+end],item['sequence'][:,-1:]],1);prefix=choose_fine_prefix(item['coarse'][:,start:end],item['sampled'][:,start:end],.5,rng)
            optimizer.zero_grad(set_to_none=True);ce,align=forward(item,prefix,seq,item['latent'][:,start:end],item['fine'][:,start:end]);loss=ce+coefficient*align
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite REPA objective')
            loss.backward();gradnorm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(gradnorm) and gradnorm>0;optimizer.step();steps+=1;losses.append([ce.item(),align.item()])
            if a.smoke and steps>=2:break
        result=validate();history.append(dict(epoch=epoch,steps=steps,**result,train_ce=float(np.mean([x[0] for x in losses])),train_alignment=float(np.mean([x[1] for x in losses])),elapsed_seconds=time.perf_counter()-started));atomic_json(out/'history.json',history)
        if result['ce']<best:best=result['ce'];best_epoch=epoch;best_state=trainable_state(model);save_best()
        print('REPA_EPOCH',a.arm,json.dumps(history[-1]),flush=True)
    restore_trainable_state(model,best_state);prediction_dir=out/'predictions';prediction_dir.mkdir(exist_ok=True)
    for name,draws in originals.items():
        for sample,(coarse,native_fine) in enumerate(draws):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):fine=raw_fine_logits(model,items[name]['sequence'],pad_coarse(coarse)).argmax(-1)
            capture.clear();np.savez_compressed(prediction_dir/f'{name}_sample{sample}.npz',coarse=coarse.cpu().numpy()[0],fine=fine.cpu().numpy()[0],native_fine=native_fine)
    handle.remove();atomic_json(out/'summary.json',dict(status='smoke_passed' if a.smoke else 'complete',arm=a.arm,steps=steps,selected_epoch=best_epoch,initial_validation=initial,best_validation_ce=best,train_seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),samples_per_protein=samples,geometry_targets=names,configuration=configuration,scope=protocol['scope']))


if __name__=='__main__':main()
