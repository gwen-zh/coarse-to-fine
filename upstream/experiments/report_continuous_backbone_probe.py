"""Apply the predeclared feasibility gate; never relabel it as model superiority."""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    run=Path('artifacts/development/continuous_backbone_probe_v1');protocol=json.loads((run/'protocol.json').read_text());seeds=[]
    for seed in protocol['seeds']:
        data=json.loads((run/f'seed{seed}/decoded.json').read_text())
        if data['status']!='complete' or len(data['records'])!=16:raise RuntimeError('Incomplete paired evaluation')
        summary={}
        for subset in ['train','validation']:
            rows=[r for r in data['records'] if r['subset']==subset];n={x['id']:x['length'] for x in protocol['targets']}
            subset_result=dict(native_rmsd=float(np.mean([r['native']['closest_available_reference_ca_rmsd'] for r in rows])),oracle_rmsd=float(np.mean([r['oracle']['closest_available_reference_ca_rmsd'] for r in rows])))
            for step in [0,protocol['steps']]:
                subset_result[str(step)]=dict(rmsd=float(np.mean([np.mean([s['metrics']['closest_available_reference_ca_rmsd'] for s in r['steps'][str(step)]['samples']]) for r in rows])),cn_rate=float(np.mean([np.mean([s['metrics']['backbone_cn_outliers']/(n[r['target']]-1) for s in r['steps'][str(step)]['samples']]) for r in rows])),conditional_spread=float(np.mean([r['steps'][str(step)]['conditional_spread'] for r in rows])),latent_mse=float(np.mean([np.mean([s['latent_mse'] for s in r['steps'][str(step)]['samples']]) for r in rows])))
            summary[subset]=subset_result
        final=str(protocol['steps']);train=summary['train'];val=summary['validation']
        criteria=dict(training_rmsd_reduction_at_least_10pct=train[final]['rmsd']<=.9*train['0']['rmsd'],training_cn_below_25pct=train[final]['cn_rate']<.25,validation_rmsd_within_twice_native=val[final]['rmsd']<=2*val['native_rmsd'])
        seeds.append(dict(seed=seed,subsets=summary,criteria=criteria,feasibility_pass=all(criteria.values())))
    result=dict(status='complete',seeds=seeds,both_seeds_feasibility_pass=all(s['feasibility_pass'] for s in seeds),scope=protocol['scope'],decision='Only feasibility: expansion needs a separately frozen next-stage protocol; failure permits one diagnosis, no automatic larger run.')
    atomic_json(run/'comparison.json',result)
    lines=['连续主干小规模可学性检查','='*42,'',protocol['scope'],'']
    for s in seeds:
        lines.append(f"seed {s['seed']}: 可学性门槛通过={s['feasibility_pass']}")
        for subset,r in s['subsets'].items():lines.append(f"  {subset}: RMSD step0 {r['0']['rmsd']:.3f} → step200 {r['200']['rmsd']:.3f} Å；原路径 {r['native_rmsd']:.3f} Å；CN {100*r['200']['cn_rate']:.2f}%")
    lines.extend(['','这只是8训练/8开发蛋白的固定预算检查，不是独立确认、系综改进或优于原模型的声明。'])
    (run/'results_zh.rst').write_text('\n'.join(lines)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
