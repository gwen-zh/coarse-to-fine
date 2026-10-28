import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from report_bounded_fine_search import interval


def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'protocol.json').read_text());training=json.loads((a.run/'full'/'summary.json').read_text());overfit=json.loads((a.run/'overfit'/'summary.json').read_text())
 if training['status']!='complete':
  result=dict(status='stopped_sanity_gate',scope=protocol['scope'],overfit=overfit,training=training,decision='Do not interpret a failed overfit probe as general impossibility. Full training/evaluation not run; first inspect optimization and representation.');atomic_json(a.run/'comparison.json',result);(a.run/'results.rst').write_text('Continuous coordinate-flow sanity gate did not pass.\n\n'+json.dumps(overfit,indent=2)+'\n');print(json.dumps(result,indent=2));return
 rows=[json.loads((a.run/'evaluation'/(name+'.json')).read_text()) for name in protocol['geometry_targets']];assert len(rows)==24;per={};errors=0
 for row in rows:
  record={}
  for key in ['original','new']:
   group=[s[key] for s in row['samples']];assert len(group)==8;rmsds=[s['metrics'].get('closest_available_reference_ca_rmsd') for s in group];all_finite=all(x is not None for x in rmsds);errors+=sum(not s['metrics'].get('backbone_cn_finite',False) for s in group)
   metrics=dict(reference_ca_rmsd=float(np.mean(rmsds)) if all_finite else None,cn_rate=float(np.mean([s['metrics']['backbone_cn_outliers']/(row['length']-1) for s in group])) if all_finite else None,ca_clash_density=float(np.mean([s['metrics']['upstream_ca_clashes']/row['length'] for s in group])) if all_finite else None,spread=row['spread'][key],quality_errors=sum(s['quality_error'] is not None for s in group))
   for source,label in [('reference_ca_c_n_angle_mae_degrees','ca_c_n_angle_error'),('reference_c_n_ca_angle_mae_degrees','c_n_ca_angle_error')]:metrics[label]=float(np.mean([s['quality'][source] for s in group])) if all(s['quality'] is not None for s in group) else None
   record[key]=metrics
  per[row['target']]=record
 keys=list(next(iter(per.values()))['new']);summary={method:{k:float(np.mean([r[method][k] for r in per.values()])) if all(r[method][k] is not None for r in per.values()) else None for k in keys} for method in ['original','new']};comparisons={}
 for k in keys:
  values=[r['new'][k]-r['original'][k] if r['new'][k] is not None and r['original'][k] is not None else None for r in per.values()];comparisons[k]=interval(values) if all(v is not None for v in values) else dict(mean=None,undefined_proteins=sum(v is None for v in values))
 result=dict(status='complete',scope=protocol['scope'],training=training,overfit=overfit,methods=summary,new_minus_original= comparisons,per_protein=per,nonfinite_sample_count=errors,mean_new_sampling_seconds=float(np.mean([s['sampling_seconds'] for r in rows for s in r['samples']])),limitations='One seed,899static training structures,20epochs,24 reused validation proteins x8 samples, not independent test. First reference read after generation. Original baseline has same proteins/sample count from prior frozen native draws, not same latent-noise space or matched compute.64small-network calls and frozen sequence-feature extraction cost not directly comparable to7large-model calls. No ESM codebook or structure decoder in new path. Analytic ideal per-residue geometry does not ensure inter-residue peptide geometry; spread is not coverage; no thermodynamic claim.');atomic_json(a.run/'comparison.json',result)
 lines=['Continuous coordinate-flow development probe','===========================================','',result['limitations'],'']
 for method,v in summary.items():lines.append(method+': '+json.dumps(v))
 lines+=['','New minus original protein-paired exploratory intervals:',json.dumps(comparisons,indent=2)];(a.run/'results.rst').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(methods=summary,new_minus_original=comparisons),indent=2))


if __name__=='__main__':main()
