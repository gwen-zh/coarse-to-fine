"""Full-corpus jointly trained rigid two-level tokenizer, then immutable token export."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_rigid_vq import HierarchicalRigidVQ,tokenizer_loss
from hierarchical_pipeline_io import RUN,protocol,load_items,save_checkpoint,restore_checkpoint,reconstruction_evaluation,sha


@torch.no_grad()
def initialize_books(model,train,device,cap,iterations):
 model.eval();coarse=[];fine=[]
 for item in train[:cap]:
  c,f,_=model.encode_latents(item['bb'].to(device));coarse.append(c);fine.append(f)
 model.coarse_book.initialize(torch.cat(coarse),iterations);model.fine_book.initialize(torch.cat(fine),iterations)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 p=protocol();config=p['tokenizer'];seed=p['seeds'][args.index];torch.set_num_threads(config['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed);device=torch.device(args.device)
 items=load_items(p);train=[x for x in items if x['row']['subset']=='train'];model=HierarchicalRigidVQ(**p['tokenizer_model']).to(device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4)
 out=RUN/('smoke_tokenizer' if args.smoke else 'tokenizer')/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():raise RuntimeError('Already complete')
 if args.smoke:
  started=time.perf_counter();losses=[]
  for i in range(32):
   model.train();optimizer.zero_grad(set_to_none=True);loss,_=tokenizer_loss(model(train[i]['bb'].to(device)),p['weights']);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss))
  initialize_books(model,train,device,32,2)
  for i in range(16):
   model.train();optimizer.zero_grad(set_to_none=True);loss,_=tokenizer_loss(model(train[i]['bb'].to(device),True),p['weights']);loss.backward()
   if not torch.isfinite(loss):raise RuntimeError('Smoke nonfinite')
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  model.eval()
  with torch.no_grad():
   x=train[0]['bb'].to(device);o=model(x,True);c,f=model.tokens(x);torch.testing.assert_close(o['backbone'],model.decode_tokens(c,f)['backbone'],atol=2e-5,rtol=2e-5)
  state_path=out/'checkpoint.pt';save_checkpoint(state_path,model,optimizer,rng,model_config=model.config,seed=seed,protocol_sha256=sha(RUN/'protocol.json'))
  reloaded=HierarchicalRigidVQ(**model.config).to(device);reloaded.load_state_dict(torch.load(state_path,map_location=device,weights_only=False)['model']);reloaded.eval()
  torch.testing.assert_close(model(x,True)['backbone'],reloaded(x,True)['backbone'],atol=0,rtol=0)
  result=dict(status='passed',steps=48,seconds=time.perf_counter()-started,parameters=sum(v.numel() for v in model.parameters()),token_only_decode=True,checkpoint_replay_exact=True,losses_finite=bool(np.isfinite(losses).all()))
  atomic_json(RUN/'tokenizer_smoke.json',result);print('HIERARCHICAL_TOKENIZER_SMOKE',json.dumps(result),flush=True);return
 if json.loads((RUN/'tokenizer_smoke.json').read_text())['status']!='passed':raise RuntimeError('Real-data smoke required')
 started=time.perf_counter();start_phase=0;start_epoch=0;updates=0;trace=[]
 if (out/'last.pt').exists():
  saved=restore_checkpoint(out/'last.pt',model,optimizer,rng,device)
  if saved['protocol_sha256']!=sha(RUN/'protocol.json'):raise RuntimeError('Resume protocol differs')
  start_phase=saved['phase'];start_epoch=saved['epoch'];updates=saved['updates']
  if (out/'trace.json').exists():trace=json.loads((out/'trace.json').read_text())
 for phase,epochs in enumerate([config['warmup_epochs'],config['quantized_epochs']]):
  if phase<start_phase:continue
  if phase==1 and not model.coarse_book.initialized:
   initialize_books(model,train,device,config['initialization_proteins'],config['kmeans_iterations']);print('HIERARCHICAL_BOOKS_INITIALIZED',seed,flush=True)
  first=start_epoch if phase==start_phase else 0
  for epoch in range(first,epochs):
   lr=config['min_lr']+.5*(config['lr']-config['min_lr'])*(1+math.cos(math.pi*epoch/max(1,epochs-1)))
   for group in optimizer.param_groups:group['lr']=lr
   model.train();terms_epoch=[];epoch_start=time.perf_counter()
   for i,index in enumerate(rng.permutation(len(train))):
    x=train[index]['bb'].to(device);optimizer.zero_grad(set_to_none=True)
    fraction=min(1.,(epoch+1)/config['quantization_ramp_epochs']) if phase else 1.
    output=model(x,quantized=bool(phase),quantization_fraction=fraction);loss,terms=tokenizer_loss(output,p['weights'])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite full tokenizer loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite tokenizer gradient')
    optimizer.step();updates+=1;terms_epoch.append({k:float(v.detach()) for k,v in terms.items()})
    if i%100==0:atomic_json(out/'progress.json',dict(stage='quantized' if phase else 'continuous_warmup',epoch=epoch+1,epochs=epochs,step_in_epoch=i+1,updates=updates,loss=float(loss),elapsed_seconds=time.perf_counter()-started))
   record=dict(phase=phase,epoch=epoch+1,updates=updates,lr=lr,seconds=time.perf_counter()-epoch_start,terms={k:float(np.mean([r[k] for r in terms_epoch])) for k in terms_epoch[0]});trace.append(record);atomic_json(out/'trace.json',trace)
   save_checkpoint(out/'last.pt',model,optimizer,rng,model_config=model.config,phase=phase,epoch=epoch+1,updates=updates,seed=seed,protocol_sha256=sha(RUN/'protocol.json'))
   print('HIERARCHICAL_TOKENIZER_EPOCH',seed,json.dumps(record),flush=True)
   if (epoch+1)%config['evaluate_every']==0 or epoch+1==epochs:
    evaluation=reconstruction_evaluation(model,items,p,device,bool(phase),out/f'phase{phase}_epoch{epoch+1}')
    print('HIERARCHICAL_TOKENIZER_EVAL',seed,phase,epoch+1,json.dumps(evaluation['groups']['validation']['metrics']),flush=True)
  if phase==0:
   # Save fixed continuous endpoint as a separate comparison, never select the best epoch.
   save_checkpoint(out/'continuous.pt',model,optimizer,rng,model_config=model.config,phase=1,epoch=0,updates=updates,seed=seed,protocol_sha256=sha(RUN/'protocol.json'))
  start_epoch=0
 final=reconstruction_evaluation(model,items,p,device,True,out/'final_evaluation')
 train_metrics=final['groups']['train'];val=final['groups']['validation'];codebooks=val['codebooks']
 gate=dict(all_finite=True,centroid_contract=final['max_centroid_error']<=2e-5,rigid_contract=final['max_rigid_error']<=2e-4,
           coarse_used=codebooks[0]['used']>=8,fine_used=codebooks[1]['used']>=16,
           validation_rmsd_below3A=val['metrics']['ca_rmsd']<=3.,validation_cn_below10pct=val['metrics']['cn_rate']<=.10,
           fine_branch_used=val['zero_fine_mse']>=1.05*val['fine_mse'],coarse_branch_used=val['zero_coarse_rmse']>=1.05*val['coarse_rmse'])
 save_checkpoint(out/'final.pt',model,optimizer,rng,model_config=model.config,phase=2,epoch=0,updates=updates,seed=seed,protocol_sha256=sha(RUN/'protocol.json'))
 model.eval();token_dir=out/'tokens';token_dir.mkdir(exist_ok=True);token_rows=[]
 for item in items:
  with torch.no_grad():c,f=model.tokens(item['bb'].to(device))
  path=token_dir/(item['row']['id']+'.npz');np.savez_compressed(path,coarse=c.cpu().numpy(),fine=f.cpu().numpy());token_rows.append(dict(id=item['row']['id'],subset=item['row']['subset'],sha256=sha(path)))
 atomic_json(out/'token_manifest.json',dict(status='complete',targets=token_rows,tokenizer_sha256=sha(out/'final.pt')))
 result=dict(status='complete',seed=seed,updates=updates,gate=gate,ready_for_generator=all(gate.values()),validation=val,train_probe=train_metrics,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,checkpoint_sha256=sha(out/'final.pt'))
 atomic_json(out/'complete.json',result);print('HIERARCHICAL_TOKENIZER_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
