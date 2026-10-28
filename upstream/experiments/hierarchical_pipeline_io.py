"""Shared audited I/O/evaluation for full two-level tokenizer -> generator training."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from compare_two_level_ae_author import metrics
from hierarchical_latent_contract import pool_ca
from hierarchical_rigid_vq import BB_LOCAL

RUN=Path('artifacts/hierarchical_rigid_pipeline_v1')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()


def protocol():
 p=json.loads((RUN/'protocol.json').read_text())
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
 return p


def load_items(p,features=False):
 items=[]
 for row in p['targets']:
  with np.load(row['reference']) as z:bb=torch.from_numpy(z['backbone'].copy()).float()
  if bb.shape!=(row['length'],3,3) or not torch.isfinite(bb).all():raise RuntimeError('Bad reference '+row['id'])
  item=dict(row=row,bb=bb)
  if features:
   with np.load(row['features']) as z:h=torch.from_numpy(z['h'].copy()).float()
   if h.shape!=(row['length'],1536) or not torch.isfinite(h).all():raise RuntimeError('Bad sequence feature '+row['id'])
   item['features']=h
  items.append(item)
 return items


def save_checkpoint(path,model,optimizer,rng,**metadata):
 path=Path(path);temporary=path.with_suffix('.tmp.pt')
 torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 numpy_rng=rng.bit_generator.state,**metadata),temporary)
 temporary.replace(path)


def restore_checkpoint(path,model,optimizer,rng,device):
 state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
 for value in optimizer.state.values():
  for k,v in value.items():
   if torch.is_tensor(v):value[k]=v.to(device)
 rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
 if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
 return state


def usage(counts):
 counts=np.asarray(counts,float);probs=counts[counts>0]/counts.sum()
 return dict(used=int((counts>0).sum()),vocabulary=len(counts),perplexity=float(np.exp(-(probs*np.log(probs)).sum())))


@torch.no_grad()
def reconstruction_evaluation(model,items,p,device,quantized,directory):
 model.eval();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);rows=[]
 selected=[x for x in items if x['row']['subset']=='validation' or x['row']['id'] in p['train_probe_ids']]
 counts={s:[np.zeros(model.coarse_book.size,dtype=np.int64),np.zeros(model.fine_book.size,dtype=np.int64)] for s in ['train','validation']}
 maximum_drift=0.;rigid_error=0.
 for item in selected:
  row=item['row'];x=item['bb'].to(device);output=model(x,quantized=quantized);bb=output['backbone'];n=len(x)
  drift=float((pool_ca(bb)-output['coarse']).abs().max());maximum_drift=max(maximum_drift,drift)
  local=bb.new_tensor(BB_LOCAL)
  for a,b in [(0,1),(1,2),(0,2)]:rigid_error=max(rigid_error,float(((bb[:,a]-bb[:,b]).norm(dim=-1)-(local[a]-local[b]).norm()).abs().max()))
  if quantized:
   c,f=output['ids'];counts[row['subset']][0]+=np.bincount(c.cpu().numpy(),minlength=model.coarse_book.size);counts[row['subset']][1]+=np.bincount(f.cpu().numpy(),minlength=model.fine_book.size)
   decoded=model.decode_tokens(c,f)['backbone'];torch.testing.assert_close(bb,decoded,atol=2e-5,rtol=2e-5)
   cvec=model.coarse_book.codes[c];fvec=model.fine_book.codes[f]
  else:cvec=output['z_c'];fvec=output['z_f']
  zero_f=model.decode(cvec,torch.zeros_like(fvec),n);zero_c=model.decode(torch.zeros_like(cvec),fvec,n)
  target_centers=pool_ca(output['target']);target_residual=output['target']-target_centers.repeat_interleave(4,0)[:n,None,:]
  data=dict(target=row['id'],subset=row['subset'],length=n,metrics=metrics(bb.cpu().numpy(),item['bb'].numpy()),
            coarse_rmse=float((output['coarse']-target_centers).square().mean().sqrt()),zero_coarse_rmse=float((zero_c['coarse']-target_centers).square().mean().sqrt()),
            fine_mse=float((output['fine_residual']-target_residual).square().mean()),zero_fine_mse=float((zero_f['fine_residual']-target_residual).square().mean()))
  if quantized:data['nominal_bits_per_residue']=(8*len(c)+10*n)/n
  rows.append(data);np.savez_compressed(directory/(row['id']+'.npz'),backbone=bb.cpu().numpy())
 groups={}
 for subset in ['train','validation']:
  selected_rows=[r for r in rows if r['subset']==subset]
  groups[subset]=dict(proteins=len(selected_rows),metrics={key:float(np.mean([r['metrics'][key] for r in selected_rows])) for key in selected_rows[0]['metrics']},
                      **{k:float(np.mean([r[k] for r in selected_rows])) for k in ['coarse_rmse','zero_coarse_rmse','fine_mse','zero_fine_mse']})
  if quantized:groups[subset]['codebooks']=[dict(usage(v),counts=v.tolist()) for v in counts[subset]]
 result=dict(status='complete',quantized=quantized,targets=rows,groups=groups,max_centroid_error=maximum_drift,max_rigid_error=rigid_error)
 atomic_json(directory/'evaluation.json',result);return result
