"""Sequence-only free generation and post-hoc reference evaluation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from coordinate_flow import CoordinateFlow,sample
from structural_metrics import summarize_backbone,aligned_rmsd
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--chunk',type=int,required=True);a=p.parse_args();torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;protocol=json.loads((a.run/'protocol.json').read_text());summary_path=a.run/'full'/'summary.json'
 for attempt in range(31):
  if summary_path.exists():break
  if attempt==30:raise RuntimeError('Full-training completion marker unavailable')
  time.sleep(2)
 status=json.loads(summary_path.read_text())['status']
 if status!='complete':atomic_json(a.run/f'evaluation_chunk{a.chunk}.json',dict(status='not_run_sanity_gate_failed'));return
 for file,digest in protocol['code_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
 device=torch.device('cuda');model=CoordinateFlow().to(device).eval();checkpoint=torch.load(a.run/'full'/'best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['state'],strict=True);del checkpoint
 names=protocol['geometry_targets'][a.chunk::protocol['evaluation_chunks']];rows={r['id']:r for r in json.loads(Path(protocol['corpus_manifest']).read_text())['targets']};directory=a.run/'evaluation';directory.mkdir(exist_ok=True)
 for name in names:
  row=rows[name];assert row['subset']=='validation'
  with np.load(a.run/'sequence_features'/(name+'.npz')) as z:h=torch.from_numpy(z['h'].copy()).to(device)[None]
  structures=[];times=[]
  for index in range(8):
   seed=int(hashlib.sha256(f'coordinate-evaluation:{name}:{index}'.encode()).hexdigest()[:8],16);torch.cuda.synchronize();started=time.perf_counter();bb=sample(model,h,seed,protocol['ode_steps'])[0].cpu().numpy();torch.cuda.synchronize();times.append(time.perf_counter()-started);structures.append(bb)
  # Real reference first read after all eight free generation draws are finalized.
  with np.load(Path('artifacts/pdb_training_v2')/row['backbone']) as z:reference=z['backbone'].copy()
  baseline_path=Path(protocol['baseline'])/'decoded'/(name+'.npz');records=[];baseline=[]
  with np.load(baseline_path) as z:
   for index in range(8):baseline.append(z[f'native_fine_{index}'].copy())
  for index,(bb,base) in enumerate(zip(structures,baseline)):
   values={}
   for key,value in [('new',bb),('original',base)]:
    try:quality=measurements(value,reference);error=None
    except ValueError as e:quality=None;error=str(e)
    values[key]=dict(metrics=summarize_backbone(value,[reference[:,1]]),quality=quality,quality_error=error)
   records.append(dict(index=index,**values,sampling_seconds=times[index]))
  def spread(items):
   pairs=[aligned_rmsd(x[:,1],y[:,1]) for i,x in enumerate(items) for y in items[i+1:]]
   return float(np.mean(pairs)) if all(x is not None for x in pairs) else None
  np.savez_compressed(directory/(name+'.npz'),backbone=np.stack(structures));atomic_json(directory/(name+'.json'),dict(status='complete',target=name,length=row['length'],samples=records,spread=dict(new=spread(structures),original=spread(baseline))));print('COORDINATE_EVALUATED',name,flush=True)
 atomic_json(a.run/f'evaluation_chunk{a.chunk}.json',dict(status='complete',targets=names))


if __name__=='__main__':main()
