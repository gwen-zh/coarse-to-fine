"""Explicit follow-up after an unaligned-coordinate pilot gate failed.

The original gate and result remain unchanged. This development experiment
continues the existing epoch5 checkpoint to the originally planned epoch60;
no extra epochs and no best-seed selection. Full deployed samples decide quality.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from mixed_hierarchical_io import data,model,tokenizer,sha,save_checkpoint,restore_checkpoint,coarse_evaluation,fine_ce,generation_evaluation
from train_mixed_hierarchical_generator import configure_stage,update

RUN=Path('artifacts/mixed_hierarchical_followup_v1')


def protocol():
 p=json.loads((RUN/'protocol.json').read_text())
 for source,digest in p['source_sha256'].items():
  if sha(source)!=digest:raise RuntimeError('Frozen follow-up input changed '+source)
 return p


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['coarse','fine'],required=True);parser.add_argument('--index',type=int,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
 p=protocol();seed=p['seeds'][args.index];device=torch.device(args.device);torch.set_num_threads(p['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed)
 items,stats=data(p,seed);train=[x for x in items if x['row']['subset']=='train'];generator=model(p,stats,device);decoder=tokenizer(p,seed,device)
 stage=args.stage;out=RUN/stage/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():raise RuntimeError('Already complete')
 configure_stage(generator,stage);optimizer=torch.optim.AdamW(generator.coarse_flow.parameters() if stage=='coarse' else generator.fine.parameters(),lr=p[stage]['lr'],weight_decay=1e-4)
 labels_root=Path(p['label_run'])/'labels'/f'seed{seed}';labels_sha=sha(labels_root/'manifest.json');ph=sha(RUN/'protocol.json');first=0;updates=0;trace=[];coarse_sha=None
 if stage=='fine':
  previous=RUN/'coarse'/f'seed{seed}'
  if json.loads((previous/'complete.json').read_text())['status']!='complete':raise RuntimeError('Complete coarse training required')
  coarse_sha=sha(previous/'final.pt');saved=torch.load(previous/'final.pt',map_location='cpu',weights_only=False);generator.load_state_dict(saved['model'])
 if (out/'last.pt').exists():
  saved=restore_checkpoint(out/'last.pt',generator,optimizer,rng,device)
  if saved['protocol_sha256']!=ph or saved['label_manifest_sha256']!=labels_sha or saved['frozen_coarse_sha256']!=coarse_sha:raise RuntimeError('Follow-up resume differs')
  first=saved['epoch'];updates=saved['updates'];trace=json.loads((out/'trace.json').read_text())
 elif stage=='coarse':
  initial=p['initial_checkpoints'][str(seed)];saved=restore_checkpoint(initial,generator,optimizer,rng,device)
  if saved['epoch']!=5 or saved['updates']!=4495 or saved['label_manifest_sha256']!=labels_sha:raise RuntimeError('Wrong pilot endpoint')
  first=saved['epoch'];updates=saved['updates'];trace=json.loads(Path(p['initial_traces'][str(seed)]).read_text())
  if len(trace)!=5:raise RuntimeError('Pilot trace must remain fixed at5epochs')
 started=time.perf_counter()
 for epoch in range(first,p[stage]['epochs']):
  lr=p[stage]['min_lr']+.5*(p[stage]['lr']-p[stage]['min_lr'])*(1+math.cos(math.pi*epoch/max(1,p[stage]['epochs']-1)))
  for group in optimizer.param_groups:group['lr']=lr
  records=[];epoch_started=time.perf_counter()
  for i,index in enumerate(rng.permutation(len(train))):
   record=update(generator,decoder,train[index],optimizer,p,stage,device,rng,epoch);records.append(record);updates+=1
   if i%100==0:atomic_json(out/'progress.json',dict(stage=stage,epoch=epoch+1,epochs=p[stage]['epochs'],updates=updates,step_in_epoch=i+1,loss=record['loss'],seconds=time.perf_counter()-started))
  record=dict(stage=stage,epoch=epoch+1,updates=updates,lr=lr,seconds=time.perf_counter()-epoch_started,loss=float(np.mean([r['loss'] for r in records])),terms={k:float(np.mean([r['terms'][k] for r in records])) for k in records[0]['terms']},sampled_coarse_updates=sum(r['sampled_coarse'] for r in records));trace.append(record);atomic_json(out/'trace.json',trace)
  save_checkpoint(out/'last.pt',generator,optimizer,rng,model_config=generator.config,stage=stage,epoch=epoch+1,updates=updates,seed=seed,protocol_sha256=ph,label_manifest_sha256=labels_sha,frozen_coarse_sha256=coarse_sha)
  print('MIXED_FOLLOWUP_EPOCH',seed,json.dumps(record),flush=True)
  if (epoch+1)%p[stage]['evaluate_every']==0 and epoch+1!=p[stage]['epochs']:
   if stage=='coarse':atomic_json(out/f'epoch{epoch+1}_evaluation.json',coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps']))
   else:
    atomic_json(out/f'epoch{epoch+1}_ce.json',fine_ce(generator,items,seed,device));generation_evaluation(generator,decoder,items,p,seed,device,out/f'epoch{epoch+1}_samples',2)
 if stage=='coarse':result=coarse_evaluation(generator,decoder,items,p['generation_probe_ids'],seed,device,p['coarse']['sampling_steps'],samples=4);atomic_json(out/'final_evaluation.json',result)
 else:
  atomic_json(out/'final_ce.json',fine_ce(generator,items,seed,device));result=generation_evaluation(generator,decoder,items,p,seed,device,out/'final_samples',p['fine']['final_samples'])
 save_checkpoint(out/'final.pt',generator,optimizer,rng,model_config=generator.config,stage=stage,epoch=p[stage]['epochs'],updates=updates,seed=seed,protocol_sha256=ph,label_manifest_sha256=labels_sha,frozen_coarse_sha256=coarse_sha)
 atomic_json(out/'complete.json',dict(status='complete',stage=stage,seed=seed,epochs=p[stage]['epochs'],updates=updates,evaluation=result['metrics'],seconds=time.perf_counter()-started,original_pilot_gate_pass=False,followup_is_not_original_gate_promotion=True));print('MIXED_FOLLOWUP_COMPLETE',stage,seed,json.dumps(result['metrics']),flush=True)


if __name__=='__main__':main()
