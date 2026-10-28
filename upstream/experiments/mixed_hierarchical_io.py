"""Immutable data, checkpoints and deployed evaluation for the mixed generator."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha,load_items,save_checkpoint,restore_checkpoint
from mixed_hierarchical_generator import HybridTokenizer,MixedHierarchicalGenerator
from compare_two_level_ae_author import metrics
from structural_metrics import aligned_rmsd,ensemble_summary

RUN=Path('artifacts/mixed_hierarchical_generator_v1')


def protocol():
 p=json.loads((RUN/'protocol.json').read_text())
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise RuntimeError('Frozen input changed '+path)
 return p


def tokenizer(p,seed,device='cpu'):
 paths=p['tokenizers'][str(seed)]
 return HybridTokenizer.from_checkpoints(paths['warmup'],paths['final'],device)


def data(p,seed):
 root=RUN/'labels'/f'seed{seed}';manifest=json.loads((root/'manifest.json').read_text());stats=json.loads((root/'stats.json').read_text());index={r['id']:r for r in manifest['targets']}
 if len(index)!=len(p['targets']) or len(index)!=len(manifest['targets']):raise RuntimeError('Incomplete label export')
 if sha(root/'stats.json')!=manifest['stats_sha256']:raise RuntimeError('Train normalization changed')
 items=load_items(p,features=True)
 for item in items:
  name=item['row']['id'];path=root/(name+'.npz')
  if sha(path)!=index[name]['sha256']:raise RuntimeError('Label changed '+name)
  with np.load(path) as z:
   item['coarse']=torch.from_numpy(z['coarse'].copy()).float();item['fine']=torch.from_numpy(z['fine'].copy()).long();item['centers']=torch.from_numpy(z['centers'].copy()).float()
  n=item['row']['length']
  if item['coarse'].shape!=((n+3)//4,16) or item['fine'].shape!=(n,) or item['centers'].shape!=((n+3)//4,3):raise RuntimeError('Label length mismatch '+name)
 return items,stats


def model(p,stats,device):
 return MixedHierarchicalGenerator(stats['mean'],stats['std'],**p['model']).to(device)


def sample_seed(seed,name,index):return int(hashlib.sha256(f'mixed-hierarchy-v1:{seed}:{name}:{index}'.encode()).hexdigest()[:8],16)


def resampled_other_features(source,length):
 indices=torch.linspace(0,len(source)-1,length).round().long()
 return source[indices]


@torch.no_grad()
def coarse_evaluation(generator,decoder,items,ids,seed,device,steps,samples=2,controls=True):
 generator.eval();selected=[x for x in items if x['row']['id'] in ids];rows=[]
 if {x['row']['id'] for x in selected}!=set(ids):raise RuntimeError('Missing coarse evaluation proteins')
 for i,item in enumerate(selected):
  name=item['row']['id'];n=item['row']['length'];h=item['features'].to(device);target=item['centers'].numpy();draws=[];positions=[]
  wrong=resampled_other_features(selected[(i+1)%len(selected)]['features'],n).to(device)
  for sample in range(samples):
   key=sample_seed(seed,name,sample);c=generator.sample_coarse(h,key,steps);centers=decoder.coarse_centers(c,n).cpu().numpy();positions.append(centers)
   record=dict(sample=sample,seed=key,center_rmse=float(np.sqrt(np.mean(np.sum((centers-target)**2,-1)))),aligned_center_rmsd=aligned_rmsd(centers,target),latent_mse=float((generator.normalize(c)-generator.normalize(item['coarse'].to(device))).square().mean()))
   if controls:
    wrong_c=generator.sample_coarse(wrong,key,steps);wrong_x=decoder.coarse_centers(wrong_c,n).cpu().numpy();record['shuffled_condition_center_rmse']=float(np.sqrt(np.mean(np.sum((wrong_x-target)**2,-1))))
   draws.append(record)
  mean_latent=generator.mean.expand(len(item['coarse']),-1);mean_centers=decoder.coarse_centers(mean_latent,n).cpu().numpy()
  rows.append(dict(target=name,samples=draws,mean_baseline_center_rmse=float(np.sqrt(np.mean(np.sum((mean_centers-target)**2,-1)))),spread=ensemble_summary(positions)['mean_pairwise_ca_rmsd']))
 keys=['center_rmse','aligned_center_rmsd','latent_mse']+(['shuffled_condition_center_rmse'] if controls else [])
 aggregate={key:float(np.mean([np.mean([s[key] for s in r['samples']]) for r in rows])) for key in keys}
 aggregate['mean_baseline_center_rmse']=float(np.mean([r['mean_baseline_center_rmse'] for r in rows]));aggregate['spread']=float(np.mean([r['spread'] for r in rows]))
 if not all(np.isfinite(v) for v in aggregate.values()):raise RuntimeError('Nonfinite coarse samples')
 return dict(status='complete',metrics=aggregate,targets=rows,scope='Sequence-only coarse samples; training-structure latent/centers used only as evaluation reference. Wrong-sequence features resampled to same length are a descriptive conditioning control.')


@torch.no_grad()
def fine_ce(generator,items,seed,device):
 generator.eval();before=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None;rows=[]
 for item in items:
  if item['row']['subset']!='validation':continue
  name=item['row']['id'];torch.manual_seed(sample_seed(seed,name,900));loss=generator.fine.denoise_loss(item['features'].to(device),generator.normalize(item['coarse'].to(device)),item['fine'].to(device),rate=.5)
  rows.append(dict(target=name,ce=float(loss)))
 torch.set_rng_state(before)
 if cuda is not None:torch.cuda.set_rng_state_all(cuda)
 return dict(status='complete',teacher_coarse_masked_ce=float(np.mean([r['ce'] for r in rows])),targets=rows,scope='Fine likelihood with native continuous coarse and50percent target masking,not deployed sample quality.')


@torch.no_grad()
def generation_evaluation(generator,decoder,items,p,seed,device,directory,samples):
 generator.eval();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);rows=[]
 for item in items:
  name=item['row']['id']
  if name not in p['generation_probe_ids']:continue
  records=[];coords=[]
  for sample in range(samples):
   key=sample_seed(seed,name,sample);c,f=generator.sample(item['features'].to(device),key,p['coarse']['sampling_steps'],p['fine']['sampling_steps'],p['fine']['temperature']);bb=decoder.decode(c,f).cpu().numpy()
   # Neither reference coordinates nor native labels enter sampling or decoding.
   record=dict(sample=sample,seed=key,metrics=metrics(bb,item['bb'].numpy()));records.append(record);coords.append(bb[:,1])
   np.savez_compressed(directory/f'{name}_sample{sample}.npz',backbone=bb,coarse=c.cpu().numpy(),fine=f.cpu().numpy())
  rows.append(dict(target=name,samples=records,spread=ensemble_summary(coords)['mean_pairwise_ca_rmsd']))
 if len(rows)!=len(p['generation_probe_ids']):raise RuntimeError('Incomplete generation evaluation')
 aggregate={key:float(np.mean([np.mean([r['metrics'][key] for r in row['samples']]) for row in rows])) for key in rows[0]['samples'][0]['metrics']}
 result=dict(status='complete',proteins=len(rows),samples_per_protein=samples,metrics=aggregate,spread=float(np.mean([r['spread'] for r in rows])),targets=rows,scope='Unfiltered sequence-only full samples;different-rate mixed representation;not thermodynamic ensemble or author-matched confirmation.')
 atomic_json(directory/'evaluation.json',result);return result
