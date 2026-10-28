"""Start only after a complete, healthy frozen tokenizer; sequence-only sampling."""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import RUN,protocol,load_items,save_checkpoint,restore_checkpoint,sha
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_token_generator import HierarchicalTokenGenerator
from compare_two_level_ae_author import metrics
from structural_metrics import ensemble_summary


def load_tokenizer(path,device):
 saved=torch.load(path,map_location='cpu',weights_only=False);model=HierarchicalRigidVQ(**saved['model_config']).to(device);model.load_state_dict(saved['model']);model.eval().requires_grad_(False);return model


@torch.no_grad()
def evaluate_ce(model,items,labels,seed,device):
 model.eval();before=torch.get_rng_state();cuda_before=torch.cuda.get_rng_state_all() if device.type=='cuda' else None;rows=[]
 for item in items:
  if item['row']['subset']!='validation':continue
  name=item['row']['id'];c,f=labels[name];h=item['features'].to(device);result={}
  for stage in ['coarse','fine']:
   torch.manual_seed(seed+int(hashlib.sha256((name+stage).encode()).hexdigest()[:7],16))
   loss,record=model.denoise_loss(h,c.to(device),f.to(device),stage,rate=.5);result[stage]=float(loss)
  rows.append(dict(target=name,**result))
 torch.set_rng_state(before)
 if cuda_before is not None:torch.cuda.set_rng_state_all(cuda_before)
 return dict(status='complete',targets=rows,coarse_masked_ce=float(np.mean([r['coarse'] for r in rows])),fine_teacher_coarse_masked_ce=float(np.mean([r['fine'] for r in rows])),scope='Fixed50percent masked conditional likelihood; fine receives native coarse. This is not deployed sequence-only quality.')


@torch.no_grad()
def sample_evaluation(model,tokenizer,items,p,device,directory,seed,samples):
 model.eval();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);rows=[]
 for item in items:
  name=item['row']['id']
  if name not in p['generation_probe_ids']:continue
  configurations=[];values=[]
  for sample in range(samples):
   sample_seed=int(hashlib.sha256(f'hierarchical-pipeline-v1:{seed}:{name}:{sample}'.encode()).hexdigest()[:8],16)
   c,f=model.sample(item['features'].to(device),sample_seed,p['generator']['sampling_steps'],p['generator']['temperature'])
   bb=tokenizer.decode_tokens(c,f)['backbone'].cpu().numpy()
   # Reference is read only after sampling and decoding have completed.
   quality=metrics(bb,item['bb'].numpy());configurations.append(dict(sample=sample,seed=sample_seed,metrics=quality));values.append(bb[:,1])
   np.savez_compressed(directory/f'{name}_sample{sample}.npz',backbone=bb,coarse=c.cpu().numpy(),fine=f.cpu().numpy())
  spread=ensemble_summary(values)['mean_pairwise_ca_rmsd'];rows.append(dict(target=name,samples=configurations,spread=spread))
 means={key:float(np.mean([np.mean([s['metrics'][key] for s in r['samples']]) for r in rows])) for key in rows[0]['samples'][0]['metrics']}
 result=dict(status='complete',proteins=len(rows),samples_per_protein=samples,metrics=means,mean_spread=float(np.mean([r['spread'] for r in rows])),targets=rows,scope='Unfiltered sequence-only draws, no teacher coarse/fine or reference-based acceptance. Single static reference per protein; not thermodynamic ensemble validation or author-matched superiority.')
 atomic_json(directory/'evaluation.json',result);return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 p=protocol();config=p['generator'];seed=p['seeds'][args.index];device=torch.device(args.device);torch.set_num_threads(config['threads']);torch.manual_seed(seed);rng=np.random.default_rng(seed)
 out=RUN/('smoke_generator' if args.smoke else 'generator')/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
 if args.smoke:
  tokenizer_path=RUN/'smoke_tokenizer'/f'seed{seed}'/'checkpoint.pt'
 else:
  completed=RUN/'tokenizer'/f'seed{seed}'/'complete.json'
  if not completed.exists():raise RuntimeError('Tokenizer is incomplete')
  ready=json.loads(completed.read_text())
  if not ready['ready_for_generator']:
   atomic_json(out/'blocked.json',dict(status='blocked_by_tokenizer_quality',gate=ready['gate'],tokenizer=str(completed),next='Inspect tokenizer diagnostics; no automatic generator trained on failed representation.'))
   print('HIERARCHICAL_GENERATOR_BLOCKED',seed,json.dumps(ready['gate']),flush=True);return
  tokenizer_path=completed.parent/'final.pt'
  if sha(tokenizer_path)!=ready['checkpoint_sha256']:raise RuntimeError('Tokenizer changed after export')
 if (out/'complete.json').exists():raise RuntimeError('Already complete')
 tokenizer=load_tokenizer(tokenizer_path,device);items=load_items(p,features=True);train=[x for x in items if x['row']['subset']=='train'];labels={}
 if args.smoke:
  items=items[:32];train=[x for x in items if x['row']['subset']=='train'][:8]
  for item in train:
   with torch.no_grad():c,f=tokenizer.tokens(item['bb'].to(device))
   labels[item['row']['id']]=(c.cpu(),f.cpu())
 else:
  manifest=json.loads((tokenizer_path.parent/'token_manifest.json').read_text());index={r['id']:r for r in manifest['targets']}
  if len(index)!=len(items) or len(index)!=len(manifest['targets']):raise RuntimeError('Incomplete tokenizer labels')
  for item in items:
   name=item['row']['id'];path=tokenizer_path.parent/'tokens'/(name+'.npz')
   if sha(path)!=index[name]['sha256']:raise RuntimeError('Token labels changed')
   with np.load(path) as z:c=torch.from_numpy(z['coarse'].copy()).long();f=torch.from_numpy(z['fine'].copy()).long()
   if c.shape!=((item['row']['length']+3)//4,) or f.shape!=(item['row']['length'],):raise RuntimeError('Code length mismatch')
   labels[name]=(c,f)
 model=HierarchicalTokenGenerator(**p['generator_model']).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=1e-4);started=time.perf_counter()
 if args.smoke:
  losses=[]
  for i in range(12):
   item=train[i%len(train)];c,f=labels[item['row']['id']];model.train();optimizer.zero_grad(set_to_none=True)
   loss,_=model.denoise_loss(item['features'].to(device),c.to(device),f.to(device),'coarse' if i%2==0 else 'fine')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Nonfinite generator smoke')
   optimizer.step();losses.append(float(loss))
  model.eval();item=train[0];h=item['features'].to(device)
  with torch.no_grad():
   c,f=model.sample(h,seed,steps=3);bb=tokenizer.decode_tokens(c,f)['backbone'];quality=metrics(bb.cpu().numpy(),item['bb'].numpy())
  path=out/'checkpoint.pt';save_checkpoint(path,model,optimizer,rng,generator_config=model.config,seed=seed,protocol_sha256=sha(RUN/'protocol.json'))
  restored=HierarchicalTokenGenerator(**model.config).to(device);restored.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);restored.eval()
  c2,f2=restored.sample(h,seed,steps=3)
  if not torch.equal(c,c2) or not torch.equal(f,f2):raise RuntimeError('Generator checkpoint sampling changed')
  result=dict(status='passed',parameters=sum(v.numel() for v in model.parameters()),steps=12,seconds=time.perf_counter()-started,losses_finite=bool(np.isfinite(losses).all()),checkpoint_sampling_exact=True,generated_backbone_finite=bool(torch.isfinite(bb).all()),note='Technical smoke only; random-generation quality is not an efficacy result.')
  atomic_json(RUN/'generator_smoke.json',result);print('HIERARCHICAL_GENERATOR_SMOKE',json.dumps(result),flush=True);return
 if json.loads((RUN/'generator_smoke.json').read_text())['status']!='passed':raise RuntimeError('Generator smoke prerequisite')
 first=0;updates=0;trace=[]
 if (out/'last.pt').exists():
  saved=restore_checkpoint(out/'last.pt',model,optimizer,rng,device)
  if saved['protocol_sha256']!=sha(RUN/'protocol.json') or saved['tokenizer_sha256']!=sha(tokenizer_path):raise RuntimeError('Resume provenance mismatch')
  first=saved['epoch'];updates=saved['updates'];trace=json.loads((out/'trace.json').read_text())
 for epoch in range(first,config['epochs']):
  lr=config['min_lr']+.5*(config['lr']-config['min_lr'])*(1+math.cos(math.pi*epoch/max(1,config['epochs']-1)))
  for group in optimizer.param_groups:group['lr']=lr
  statistics={'coarse':[],'fine':[]};sampled_count=0;epoch_start=time.perf_counter()
  for i,index in enumerate(rng.permutation(len(train))):
   item=train[index];h=item['features'].to(device);c,f=labels[item['row']['id']];c,f=c.to(device),f.to(device);stage='coarse' if updates%2==0 else 'fine';condition=None
   if stage=='fine' and epoch>=config['generated_coarse_after_epoch'] and rng.random()<config['generated_coarse_probability']:
    model.eval();condition=model.sample_coarse(h,int(rng.integers(2**31)),steps=config['training_coarse_steps']);sampled_count+=1
   model.train();optimizer.zero_grad(set_to_none=True);loss,_=model.denoise_loss(h,c,f,stage,condition)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite full generator loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise RuntimeError('Nonfinite generator gradient')
   optimizer.step();updates+=1;statistics[stage].append(float(loss))
   if i%100==0:atomic_json(out/'progress.json',dict(epoch=epoch+1,epochs=config['epochs'],updates=updates,step_in_epoch=i+1,stage=stage,loss=float(loss),elapsed_seconds=time.perf_counter()-started))
  record=dict(epoch=epoch+1,updates=updates,lr=lr,seconds=time.perf_counter()-epoch_start,coarse_loss=float(np.mean(statistics['coarse'])),fine_loss=float(np.mean(statistics['fine'])),generated_coarse_updates=sampled_count);trace.append(record);atomic_json(out/'trace.json',trace)
  save_checkpoint(out/'last.pt',model,optimizer,rng,generator_config=model.config,epoch=epoch+1,updates=updates,seed=seed,protocol_sha256=sha(RUN/'protocol.json'),tokenizer_sha256=sha(tokenizer_path))
  print('HIERARCHICAL_GENERATOR_EPOCH',seed,json.dumps(record),flush=True)
  if (epoch+1)%config['evaluate_every']==0 or epoch+1==config['epochs']:
   ce=evaluate_ce(model,items,labels,seed,device);atomic_json(out/f'epoch{epoch+1}_ce.json',ce)
   if epoch+1!=config['epochs']:sample_evaluation(model,tokenizer,items,p,device,out/f'epoch{epoch+1}_samples',seed,2)
 final=sample_evaluation(model,tokenizer,items,p,device,out/'final_samples',seed,config['final_samples'])
 save_checkpoint(out/'final.pt',model,optimizer,rng,generator_config=model.config,epoch=config['epochs'],updates=updates,seed=seed,protocol_sha256=sha(RUN/'protocol.json'),tokenizer_sha256=sha(tokenizer_path))
 result=dict(status='complete',seed=seed,updates=updates,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started,generation=final)
 atomic_json(out/'complete.json',result);print('HIERARCHICAL_GENERATOR_COMPLETE',seed,json.dumps(final['metrics']),flush=True)


if __name__=='__main__':main()
