"""One bounded same-objective continuation of the same eight TRAINING proteins."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coordinate_flow import CoordinateFlow,frames_from_backbone,centered,random_prior,training_loss,sample
from structural_metrics import summarize_backbone
from diagnostic_io import atomic_json


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--extra-steps',type=int,default=10000);a=p.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(20261027);np.random.seed(20261027);device=torch.device('cuda');protocol=json.loads((a.run/'protocol.json').read_text());source=a.run/'overfit'/'best.pt';out=a.run/'overfit_continuation_v1';out.mkdir(exist_ok=True)
 for f,h in protocol['code_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 records=json.loads((a.run/'overfit'/'summary.json').read_text())['records'];rows={r['id']:r for r in json.loads(Path(protocol['corpus_manifest']).read_text())['targets']};cache={}
 for record in records:
  name=record['id'];row=rows[name];assert row['subset']=='train'
  with np.load(Path('artifacts/pdb_training_v2')/row['backbone']) as z:bb=z['backbone'].copy()
  with np.load(a.run/'sequence_features'/(name+'.npz')) as z:h=torch.from_numpy(z['h'].copy()).to(device)[None]
  r,x=frames_from_backbone(torch.from_numpy(bb).to(device)[None]);cache[name]=dict(r=r,x=centered(x),h=h,reference=bb)
 model=CoordinateFlow().to(device);checkpoint=torch.load(source,map_location=device,weights_only=False);model.load_state_dict(checkpoint['state']);del checkpoint;optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr']);rng=np.random.default_rng(20261027);names=list(cache);started=time.perf_counter();history=[];parts=[];output_protocol=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),extra_steps=a.extra_steps,lr=protocol['lr'],objective='Unchanged coordinate_flow.training_loss, architecture,8proteins,64-step sampler and original gate. No geometry-weight or sampler tuning.',optimizer='Fresh AdamW: old checkpoint contains model weights only, not exact optimizer-state resume.',seed=20261027,scope='Training-only optimization-budget diagnostic; no validation/test access or full-training launch. Fixed10000 extra steps; intermediate rollouts recorded, final fixed checkpoint determines same sanity gate. Both torch and NumPy seeded.')
 atomic_json(out/'protocol.json',output_protocol)
 @torch.no_grad()
 def evaluate(step):
  model.eval();results=[];directory=out/('step'+str(step));directory.mkdir(exist_ok=True)
  for name,d in cache.items():
   seed=int(hashlib.sha256(('coordinate-overfit:'+name).encode()).hexdigest()[:8],16);bb=sample(model,d['h'],seed,64)[0].cpu().numpy();np.savez_compressed(directory/(name+'.npz'),backbone=bb);metrics=summarize_backbone(bb,[d['reference'][:,1]]);results.append(dict(id=name,length=len(bb),metrics=metrics))
  finite=all(r['metrics'].get('closest_available_reference_ca_rmsd') is not None and r['metrics'].get('backbone_cn_finite') for r in results);rmsd=float(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in results])) if finite else None;cn=float(np.mean([r['metrics']['backbone_cn_outliers']/(r['length']-1) for r in results])) if finite else None
  result=dict(step=step,total_steps=2000+step,mean_training_ca_rmsd=rmsd,mean_training_cn_rate=cn,sanity_gate_passed=bool(finite and rmsd<=4 and cn<=.2),records=results,seconds=time.perf_counter()-started);atomic_json(directory/'summary.json',result);model.train();return result
 initial=evaluate(0);history.append(initial);atomic_json(out/'history.json',history)
 for step in range(1,a.extra_steps+1):
  d=cache[names[int(rng.integers(len(names)))]];optimizer.zero_grad(set_to_none=True);q,_=random_prior(1,1,device);r=q@d['r'];x=torch.einsum('bnij,bnj->bni',q,d['x']);r0,x0=random_prior(1,x.shape[1],device);t=torch.rand(1,device=device)*.98+.01;loss,_=training_loss(model,r,x,d['h'],t,r0,x0)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite same-objective continuation loss')
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm) and norm>0;optimizer.step();parts.append(loss.item())
  if step%200==0:print('COORDINATE_CONTINUATION_STEP',step,float(np.mean(parts[-200:])),time.perf_counter()-started,flush=True)
  if step%2000==0 or step==a.extra_steps:
   result=evaluate(step);result['recent_train_objective']=float(np.mean(parts[-200:]));history.append(result);atomic_json(out/'history.json',history);torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,protocol=output_protocol),out/'latest.pt');print('COORDINATE_CONTINUATION_ROLLOUT',json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
 final=history[-1];atomic_json(out/'summary.json',dict(status='complete',**final,initial_rollout=initial,scope=output_protocol['scope']));print('COORDINATE_CONTINUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
