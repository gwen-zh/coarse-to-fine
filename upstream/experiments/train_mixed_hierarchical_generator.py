"""Ordered mixed generator: technical smoke, first5coarseepochs,fullcoarse,fine."""
import argparse,json,math,time,copy,shutil
import numpy as np
import torch
from diagnostic_io import atomic_json
from mixed_hierarchical_io import RUN,protocol,data,model,tokenizer,sha,save_checkpoint,restore_checkpoint,coarse_evaluation,fine_ce,generation_evaluation


def configure_stage(generator,stage):
 generator.coarse_flow.requires_grad_(stage=='coarse');generator.fine.requires_grad_(stage=='fine')
 generator.eval()
 if stage=='coarse':generator.coarse_flow.train()
 else:generator.fine.train()


def update(generator,decoder,item,optimizer,p,stage,device,rng,epoch):
 configure_stage(generator,stage);optimizer.zero_grad(set_to_none=True);h=item['features'].to(device);target=item['coarse'].to(device)
 sampled=False
 if stage=='coarse':loss,terms=generator.coarse_loss(h,target,decoder,p['coarse']['geometry_weight'])
 else:
  if epoch>=p['fine']['generated_coarse_after_epoch'] and rng.random()<p['fine']['generated_coarse_probability']:
   generator.coarse_flow.eval();target=generator.sample_coarse(h,int(rng.integers(2**31)),p['fine']['training_coarse_steps']);sampled=True
  loss=generator.fine.denoise_loss(h,generator.normalize(target),item['fine'].to(device));terms=dict(ce=loss.detach())
 if not torch.isfinite(loss):raise RuntimeError('Nonfinite mixed '+stage+' loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(generator.coarse_flow.parameters() if stage=='coarse' else generator.fine.parameters(),1.)
 if not torch.isfinite(norm):raise RuntimeError('Nonfinite mixed '+stage+' gradient')
 optimizer.step()
 return dict(loss=float(loss.detach()),terms={k:float(v) for k,v in terms.items()},sampled_coarse=sampled)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','pilot','coarse','fine'],required=True);parser.add_argument('--index',type=int,default=0);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
 p=protocol();seed=p['seeds'][args.index];torch.set_num_threads(p['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed);device=torch.device(args.device)
 items,stats=data(p,seed);train=[x for x in items if x['row']['subset']=='train'];generator=model(p,stats,device);decoder=tokenizer(p,seed,device)
 manifest_sha=sha(RUN/'labels'/f'seed{seed}'/'manifest.json');protocol_sha=sha(RUN/'protocol.json')
 stage='coarse' if args.stage in ['pilot','coarse'] else 'fine';out=RUN/stage/f'seed{seed}';pilot=RUN/'pilot'/f'seed{seed}'
 out.mkdir(parents=True,exist_ok=True);pilot.mkdir(parents=True,exist_ok=True)
 if args.stage=='smoke':
  smoke=RUN/'smoke'/f'seed{seed}';smoke.mkdir(parents=True,exist_ok=True);results=[];started=time.perf_counter()
  for part in ['coarse','fine']:
   configure_stage(generator,part);parameters=generator.coarse_flow.parameters() if part=='coarse' else generator.fine.parameters();optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=1e-4)
   for i in range(12):update(generator,decoder,train[i%len(train)],optimizer,p,part,device,rng,0)
   path=smoke/(part+'.pt');save_checkpoint(path,generator,optimizer,rng,model_config=generator.config)
   expected=update(generator,decoder,train[0],optimizer,p,part,device,rng,0);state={k:v.clone() for k,v in generator.state_dict().items()};optim_state=copy.deepcopy(optimizer.state_dict());torch_state=torch.get_rng_state().clone()
   restored=model(p,stats,device);configure_stage(restored,part);opt2=torch.optim.AdamW(restored.coarse_flow.parameters() if part=='coarse' else restored.fine.parameters(),lr=3e-4,weight_decay=1e-4);rng2=np.random.default_rng(99)
   restore_checkpoint(path,restored,opt2,rng2,device);actual=update(restored,decoder,train[0],opt2,p,part,device,rng2,0)
   if expected!=actual or not all(torch.equal(v,restored.state_dict()[k]) for k,v in state.items()) or not torch.equal(torch_state,torch.get_rng_state()):raise RuntimeError('Mixed checkpoint next-update replay failed')
   for key,values in optim_state['state'].items():
    for name,value in values.items():
     if not torch.equal(value,opt2.state_dict()['state'][key][name]):raise RuntimeError('Adam replay failed')
   results.append(dict(stage=part,next_update_model_rng_adam_exact=True))
  generator.eval();item=train[0]
  with torch.no_grad():
   c,f=generator.sample(item['features'].to(device),seed,coarse_steps=3,fine_steps=3);bb=decoder.decode(c,f)
   if not torch.isfinite(bb).all():raise RuntimeError('Nonfinite smoke sample')
  if any(v.grad is not None or v.requires_grad for v in decoder.parameters()):raise RuntimeError('Tokenizer was trained')
  result=dict(status='passed',seconds=time.perf_counter()-started,parameters=sum(v.numel() for v in generator.parameters()),stages=results,sequence_only_sampling=True,backbone_finite=True)
  atomic_json(RUN/'smoke.json',result);print('MIXED_GENERATOR_SMOKE',json.dumps(result),flush=True);return
 if json.loads((RUN/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke required')
 if args.stage=='coarse':
  gate=json.loads((pilot/'complete.json').read_text())
  if not gate['pass']:
   atomic_json(out/'blocked.json',dict(status='blocked_by_coarse_pilot',gate=gate['gate']));print('MIXED_COARSE_BLOCKED',seed,flush=True);return
 if stage=='fine':
  previous=RUN/'coarse'/f'seed{seed}'
  if (previous/'blocked.json').exists():
   atomic_json(out/'blocked.json',dict(status='blocked_by_coarse_pilot'));print('MIXED_FINE_BLOCKED',seed,flush=True);return
  complete=json.loads((previous/'complete.json').read_text())
  if complete['status']!='complete':raise RuntimeError('Full coarse stage required')
  frozen_coarse_sha=sha(previous/'final.pt')
  saved=torch.load(previous/'final.pt',map_location='cpu',weights_only=False);generator.load_state_dict(saved['model'])
 else:frozen_coarse_sha=None
 if (out/'complete.json').exists():raise RuntimeError('Stage already complete')
 configure_stage(generator,stage);optimizer=torch.optim.AdamW(generator.coarse_flow.parameters() if stage=='coarse' else generator.fine.parameters(),lr=p[stage]['lr'],weight_decay=1e-4)
 first=0;updates=0;trace=[];started=time.perf_counter()
 if (out/'last.pt').exists():
  saved=restore_checkpoint(out/'last.pt',generator,optimizer,rng,device)
  if saved['protocol_sha256']!=protocol_sha or saved['label_manifest_sha256']!=manifest_sha or saved['stage']!=stage or saved.get('frozen_coarse_sha256')!=frozen_coarse_sha:raise RuntimeError('Mixed resume provenance mismatch')
  first=saved['epoch'];updates=saved['updates'];trace=json.loads((out/'trace.json').read_text())
 if args.stage=='pilot' and not (pilot/'initial_validation.json').exists():
  a=coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps']);b=coarse_evaluation(generator,decoder,items,p['train_probe_ids'],seed,device,p['coarse']['sampling_steps'],controls=False)
  atomic_json(pilot/'initial_validation.json',a);atomic_json(pilot/'initial_train.json',b)
 end=p['coarse']['pilot_epochs'] if args.stage=='pilot' else p[stage]['epochs']
 for epoch in range(first,end):
  total_epochs=p[stage]['epochs'];lr=p[stage]['min_lr']+.5*(p[stage]['lr']-p[stage]['min_lr'])*(1+math.cos(math.pi*epoch/max(1,total_epochs-1)))
  for group in optimizer.param_groups:group['lr']=lr
  records=[];epoch_started=time.perf_counter()
  for i,index in enumerate(rng.permutation(len(train))):
   record=update(generator,decoder,train[index],optimizer,p,stage,device,rng,epoch);records.append(record);updates+=1
   if i%100==0:atomic_json(out/'progress.json',dict(stage=stage,epoch=epoch+1,epochs=end,step_in_epoch=i+1,updates=updates,loss=record['loss'],seconds=time.perf_counter()-started))
  record=dict(stage=stage,epoch=epoch+1,updates=updates,lr=lr,seconds=time.perf_counter()-epoch_started,loss=float(np.mean([r['loss'] for r in records])),terms={k:float(np.mean([r['terms'][k] for r in records])) for k in records[0]['terms']},sampled_coarse_updates=sum(r['sampled_coarse'] for r in records));trace.append(record);atomic_json(out/'trace.json',trace)
  save_checkpoint(out/'last.pt',generator,optimizer,rng,model_config=generator.config,stage=stage,epoch=epoch+1,updates=updates,seed=seed,protocol_sha256=protocol_sha,label_manifest_sha256=manifest_sha,frozen_coarse_sha256=frozen_coarse_sha)
  print('MIXED_GENERATOR_EPOCH',seed,json.dumps(record),flush=True)
  if (epoch+1)%p[stage]['evaluate_every']==0 and epoch+1!=end:
   if stage=='coarse':atomic_json(out/f'epoch{epoch+1}_evaluation.json',coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps']))
   else:
    atomic_json(out/f'epoch{epoch+1}_ce.json',fine_ce(generator,items,seed,device))
    generation_evaluation(generator,decoder,items,p,seed,device,out/f'epoch{epoch+1}_samples',2)
 if args.stage=='pilot':
  val=coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps']);training=coarse_evaluation(generator,decoder,items,p['train_probe_ids'],seed,device,p['coarse']['sampling_steps'],controls=False)
  initial_v=json.loads((pilot/'initial_validation.json').read_text());initial_t=json.loads((pilot/'initial_train.json').read_text())
  gate=dict(train_center_rmse_improves10pct=training['metrics']['center_rmse']<=.9*initial_t['metrics']['center_rmse'],validation_center_rmse_improves5pct=val['metrics']['center_rmse']<=.95*initial_v['metrics']['center_rmse'],validation_within5pct_mean_baseline=val['metrics']['center_rmse']<=1.05*val['metrics']['mean_baseline_center_rmse'],all_finite=all(np.isfinite(x) for x in val['metrics'].values()))
  atomic_json(pilot/'final_validation.json',val);atomic_json(pilot/'final_train.json',training);shutil.copy2(out/'last.pt',pilot/'final.pt')
  result=dict(status='complete',pass_gate=all(gate.values()),gate=gate,validation=val['metrics'],training=training['metrics'],scope='Engineering learnability screen,not proof of sequence conditioning,diversity or independent generalization.')
  result['pass']=result.pop('pass_gate');atomic_json(pilot/'complete.json',result);print('MIXED_COARSE_PILOT_COMPLETE',seed,json.dumps(result),flush=True);return
 if stage=='coarse':
  result=coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps'],samples=4);atomic_json(out/'final_evaluation.json',result)
 else:
  atomic_json(out/'final_ce.json',fine_ce(generator,items,seed,device));result=generation_evaluation(generator,decoder,items,p,seed,device,out/'final_samples',p['fine']['final_samples'])
 save_checkpoint(out/'final.pt',generator,optimizer,rng,model_config=generator.config,stage=stage,epoch=end,updates=updates,seed=seed,protocol_sha256=protocol_sha,label_manifest_sha256=manifest_sha,frozen_coarse_sha256=frozen_coarse_sha)
 atomic_json(out/'complete.json',dict(status='complete',stage=stage,seed=seed,epochs=end,updates=updates,evaluation=result['metrics'],seconds=time.perf_counter()-started));print('MIXED_GENERATOR_STAGE_COMPLETE',stage,seed,json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
