"""Training-only diagnosis of a failed eight-protein sanity; no new fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from coordinate_flow import CoordinateFlow,frames_from_backbone,backbone_from_frames,centered,random_prior,interpolate,sample,training_loss
from structural_metrics import summarize_backbone
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();assert torch.cuda.is_available();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda');protocol=json.loads((a.run/'protocol.json').read_text());model=CoordinateFlow().to(device).eval();checkpoint_path=a.run/'overfit'/'best.pt';checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False);model.load_state_dict(checkpoint['state']);del checkpoint
 names=[r['id'] for r in json.loads((a.run/'overfit'/'summary.json').read_text())['records']];rows={r['id']:r for r in json.loads(Path(protocol['corpus_manifest']).read_text())['targets']};records=[]
 for name in names:
  row=rows[name];assert row['subset']=='train'
  with np.load(Path('artifacts/pdb_training_v2')/row['backbone']) as z:reference=z['backbone'].copy()
  with np.load(a.run/'sequence_features'/(name+'.npz')) as z:h=torch.from_numpy(z['h'].copy()).to(device)[None]
  rotation,translation=frames_from_backbone(torch.from_numpy(reference).to(device)[None]);translation=centered(translation);rng=torch.Generator(device=device).manual_seed(773);r0,x0=random_prior(1,row['length'],device,rng);denoising=[]
  permutation=torch.randperm(row['length'],generator=torch.Generator().manual_seed(773)).to(device)
  for value in [0.,.1,.3,.5,.75,.9,.99]:
   t=torch.tensor([value],device=device);rt,xt=interpolate(rotation,translation,r0,x0,t);unmodified=backbone_from_frames(rt,xt)[0].cpu().numpy();arms={}
   for arm,features in [('normal',h),('permuted_positions',h[:,permutation]),('zero_features',torch.zeros_like(h))]:
    pr,px=model(rt,xt,t,features);bb=backbone_from_frames(pr,px)[0].cpu().numpy();loss,terms=training_loss(model,rotation,translation,features,t,r0,x0);arms[arm]=dict(metrics=summarize_backbone(bb,[reference[:,1]]),objective=float(loss),terms={k:float(v) for k,v in terms.items()})
   denoising.append(dict(t=value,corrupted_input_metrics=summarize_backbone(unmodified,[reference[:,1]]),arms=arms))
  seed=int(hashlib.sha256(('coordinate-overfit:'+name).encode()).hexdigest()[:8],16);rollouts={}
  for steps in [32,64,128]:
   bb=sample(model,h,seed,steps)[0].cpu().numpy();rollouts[str(steps)]=summarize_backbone(bb,[reference[:,1]])
  with np.load(a.run/'overfit'/'predictions'/(name+'.npz')) as z:old=z['backbone'].copy()
  replay=sample(model,h,seed,64)[0].cpu().numpy();replay_error=float(np.max(np.abs(replay-old)));records.append(dict(id=name,length=row['length'],denoising=denoising,rollouts=rollouts,replay_max_absolute_coordinate_difference=replay_error));print('COORDINATE_DIAGNOSED',name,flush=True)
 aggregate={}
 for steps in ['32','64','128']:aggregate[steps]=dict(reference_ca_rmsd=float(np.mean([r['rollouts'][steps]['closest_available_reference_ca_rmsd'] for r in records])),cn_rate=float(np.mean([r['rollouts'][steps]['backbone_cn_outliers']/(r['length']-1) for r in records])))
 by_t=[]
 for i,value in enumerate([0.,.1,.3,.5,.75,.9,.99]):
  arms={}
  for arm in ['normal','permuted_positions','zero_features']:arms[arm]=dict(reference_ca_rmsd=float(np.mean([r['denoising'][i]['arms'][arm]['metrics']['closest_available_reference_ca_rmsd'] for r in records])),cn_rate=float(np.mean([r['denoising'][i]['arms'][arm]['metrics']['backbone_cn_outliers']/(r['length']-1) for r in records])),translation_loss=float(np.mean([r['denoising'][i]['arms'][arm]['terms']['translation'] for r in records])),rotation_loss=float(np.mean([r['denoising'][i]['arms'][arm]['terms']['rotation'] for r in records])))
  by_t.append(dict(t=value,arms=arms))
 result=dict(status='complete',checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),scope='Same eight TRAINING proteins; fixed-time oracle-corruption diagnostic and same-prior ODE step sensitivity. References explicitly used to construct diagnostic corruptions, never free-rollout inputs. No fitting or validation/test access. Shuffled/zero features are perturbation probes, not trained baselines.',rollout_summary=aggregate,denoising_summary=by_t,records=records,initialization_caveat='Current v1 trainer seeds torch but official SciPy initializer also uses global NumPy RNG, which was not seeded. Saved checkpoint is fixed; do not claim exact initialization replay or multi-seed robustness. Future fits must seed both.')
 atomic_json(a.run/'sanity_diagnosis.json',result);print(json.dumps(dict(rollouts=aggregate,denoising=by_t),indent=2))


if __name__=='__main__':main()
