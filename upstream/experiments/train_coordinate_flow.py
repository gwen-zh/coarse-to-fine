"""Staged new continuous-backbone model: smoke, train-only sanity, generalization."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coordinate_flow import CoordinateFlow,frames_from_backbone,centered,random_prior,training_loss,sample
from structural_metrics import summarize_backbone
from diagnostic_io import atomic_json


def seed_for(name,tag):return int(hashlib.sha256((tag+':'+name).encode()).hexdigest()[:8],16)


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['smoke','overfit','full'],required=True);a=p.parse_args();torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261026);device=torch.device('cuda');protocol=json.loads((a.run/'protocol.json').read_text());out=a.run/a.stage;out.mkdir(exist_ok=True)
 if a.stage=='full':
  gate=json.loads((a.run/'overfit'/'summary.json').read_text())
  if not gate['sanity_gate_passed']:
   atomic_json(out/'summary.json',dict(status='not_run_sanity_gate_failed',reason='New network did not pass predeclared training-only free-generation sanity check; no generalization claim.'));print('COORDINATE_FULL_NOT_RUN_SANITY_GATE',flush=True);return
 for file,digest in protocol['code_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
 for chunk in range(protocol['cache_chunks']):
  marker=a.run/f'sequence_chunk{chunk}.json'
  for attempt in range(31):
   if marker.exists():break
   if attempt==30:raise RuntimeError('Pure-sequence feature cache incomplete')
   time.sleep(2)
  assert json.loads(marker.read_text())['status']=='complete'
 source=Path(protocol['corpus_manifest']);assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['corpus_manifest_sha256'];rows=json.loads(source.read_text())['targets'];train=sorted([r for r in rows if r['subset']=='train'],key=lambda r:r['id']);val=sorted([r for r in rows if r['subset']=='validation'],key=lambda r:r['id']);assert len(train)==899 and len(val)==101
 if a.stage!='full':train=train[:2 if a.stage=='smoke' else protocol['overfit_proteins']];val=[]
 cache={}
 for row in train+val:
  with np.load(Path('artifacts/pdb_training_v2')/row['backbone']) as z:bb=torch.from_numpy(z['backbone'].copy()).float()
  with np.load(a.run/'sequence_features'/(row['id']+'.npz')) as z:h=torch.from_numpy(z['h'].copy()).float()
  assert h.shape==(row['length'],1536) and bb.shape==(row['length'],3,3) and torch.isfinite(h).all() and torch.isfinite(bb).all();r,x=frames_from_backbone(bb);cache[row['id']]=dict(r=r[None],x=centered(x[None]),h=h[None],reference=bb.numpy())
 def item(row):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cache[row['id']].items()}
 model=CoordinateFlow().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr']);parameters=sum(v.numel() for v in model.parameters());history=[];started=time.perf_counter();peak=0
 atomic_json(out/'initial.json',dict(stage=a.stage,train_proteins=len(train),validation_proteins=len(val),parameters=parameters,gpu=torch.cuda.get_device_name(),initialization='random FrameFlow/sequence projection, no ESM decoder weights'))
 @torch.no_grad()
 def validate():
  model.eval();values=[]
  for row in val:
   d=item(row);rng=torch.Generator(device=device).manual_seed(seed_for(row['id'],'coordinate-validation'));r0,x0=random_prior(1,row['length'],device,rng);parts=[]
   for t in [.25,.75]:
    loss,_=training_loss(model,d['r'],d['x'],d['h'],torch.tensor([t],device=device),r0,x0);parts.append(loss.item())
   values.append(float(np.mean(parts)))
  model.train();return float(np.mean(values))
 def snapshot(epoch,score):torch.save(dict(state=model.state_dict(),protocol=protocol,epoch=epoch,validation_objective=score),out/'best.pt')
 best=validate() if val else None;best_epoch=0;snapshot(0,best)
 rng=np.random.default_rng(protocol['seed']);steps=0;epochs=protocol['epochs'] if a.stage=='full' else 1;total_steps=2 if a.stage=='smoke' else protocol['overfit_steps']
 for epoch in range(1,epochs+1):
  order=rng.permutation(len(train)).tolist() if a.stage=='full' else rng.integers(len(train),size=total_steps).tolist();parts=[]
  for index in order:
   row=train[index];d=item(row);optimizer.zero_grad(set_to_none=True);q,_=random_prior(1,1,device);rotation=q@d['r'];translation=torch.einsum('bnij,bnj->bni',q,d['x']);r0,x0=random_prior(1,row['length'],device);t=torch.rand(1,device=device)*.98+.01;loss,terms=training_loss(model,rotation,translation,d['h'],t,r0,x0)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite coordinate-flow loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm) or norm<=0:raise RuntimeError('Invalid coordinate-flow gradient')
   optimizer.step();steps+=1;parts.append(loss.item())
   if steps%100==0:print('COORDINATE_TRAIN_STEP',a.stage,steps,float(np.mean(parts[-100:])),time.perf_counter()-started,flush=True)
  score=validate() if val else None
  if not val or score<best:best=score;best_epoch=epoch;snapshot(epoch,score)
  history.append(dict(epoch=epoch,steps=steps,train_objective=float(np.mean(parts)),validation_objective=score,seconds=time.perf_counter()-started));atomic_json(out/'history.json',history);print('COORDINATE_EPOCH',json.dumps(history[-1]),flush=True)
 if a.stage=='overfit':
  model.eval();records=[];directory=out/'predictions';directory.mkdir(exist_ok=True)
  for row in train:
   d=item(row);bb=sample(model,d['h'],seed_for(row['id'],'coordinate-overfit'),protocol['ode_steps'])[0].cpu().numpy();np.savez_compressed(directory/(row['id']+'.npz'),backbone=bb);metrics=summarize_backbone(bb,[d['reference'][:,1]]);records.append(dict(id=row['id'],length=row['length'],metrics=metrics))
  all_finite=all(r['metrics'].get('closest_available_reference_ca_rmsd') is not None and r['metrics'].get('backbone_cn_finite') for r in records);rmsd=float(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in records])) if all_finite else None;cn=float(np.mean([r['metrics']['backbone_cn_outliers']/(r['length']-1) for r in records])) if all_finite else None;passed=bool(all_finite and rmsd<=4 and cn<=.2)
  atomic_json(out/'summary.json',dict(status='complete',sanity_gate_passed=passed,mean_training_ca_rmsd=rmsd,mean_training_cn_rate=cn,records=records,steps=steps,parameters=parameters,seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Training-only eight-protein overfit sanity. No validation/generalization/efficacy claim. Full899-protein training allowed only if fixed gate passes.'));print('COORDINATE_SANITY_GATE',passed,rmsd,cn,flush=True)
 else:atomic_json(out/'summary.json',dict(status='smoke_passed' if a.stage=='smoke' else 'complete',steps=steps,selected_epoch=best_epoch,validation_objective=best,parameters=parameters,seconds=time.perf_counter()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
