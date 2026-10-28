"""Apply frozen feasibility gates to both two-level AE seeds, no selection."""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    run=Path('artifacts/development/two_level_autoencoder_v1');p=json.loads((run/'protocol.json').read_text());results=[];expected={x['id'] for x in p['targets']}
    for seed in p['seeds']:
        directory=run/f'seed{seed}';data=json.loads((directory/f"step{p['steps']}/evaluation.json").read_text());initial=json.loads((directory/'step0/evaluation.json').read_text())
        if data['status']!='complete' or {x['target'] for x in data['targets']}!=expected or len(data['targets'])!=len(expected):raise RuntimeError('Incomplete paired evaluation')
        groups={}
        for subset in ['train','validation']:
            rows=[r for r in data['targets'] if r['subset']==subset];first=[r for r in initial['targets'] if r['subset']==subset]
            metrics={name:{key:float(np.mean([r['metrics'][name][key] for r in rows])) for key in ['rmsd','cn_rate','clashes_per_residue']} for name in rows[0]['metrics']}
            groups[subset]=dict(proteins=len(rows),initial_full_rmsd=float(np.mean([r['metrics']['full']['rmsd'] for r in first])),methods=metrics,**{k:float(np.mean([r[k] for r in rows])) for k in ['coarse_rmse','zero_coarse_rmse','fine_mse','zero_fine_mse']},max_centroid_drift=max(r['centroid_invariance_max_error'] for r in rows))
        t=groups['train'];v=groups['validation']
        criteria=dict(train_rmsd_below1A=t['methods']['full']['rmsd']<=1.,train_cn_below5pct=t['methods']['full']['cn_rate']<=.05,validation_rmsd_below2A=v['methods']['full']['rmsd']<=2.,validation_cn_below10pct=v['methods']['full']['cn_rate']<=.10,coarse_used=all(x['zero_coarse_rmse']>=1.05*x['coarse_rmse'] for x in [t,v]),fine_used=all(x['zero_fine_mse']>=1.05*x['fine_mse'] for x in [t,v]),fine_cannot_move_centers=max(t['max_centroid_drift'],v['max_centroid_drift'])<=2e-5)
        results.append(dict(seed=seed,subsets=groups,criteria=criteria,feasibility_pass=all(criteria.values())))
    result=dict(status='complete',seeds=results,both_seeds_feasibility_pass=all(x['feasibility_pass'] for x in results),scope=p['scope'],decision='No automatic larger AE or generator training. Passing supports capacity-matched controls and broader-data study; failure triggers diagnosis rather than budget extension. Does not establish superiority over original fine reconstruction.')
    atomic_json(run/'comparison.json',result)
    lines=['可训练两层autoencoder首轮结果','='*44,'',p['scope'],'']
    for result_seed in results:
        lines.append(f"seed{result_seed['seed']}：可学性门槛通过={result_seed['feasibility_pass']}")
        for subset,x in result_seed['subsets'].items():
            full=x['methods']['full'];zero=x['methods']['zero_fine'];lines.append(f"  {subset}: CA RMSD {full['rmsd']:.4f} Å，CN异常{100*full['cn_rate']:.2f}%；关闭fine后RMSD {zero['rmsd']:.4f} Å；coarse中心RMSE {x['coarse_rmse']:.4f} Å。")
    lines+=['','coarse中心不随fine改变是架构保证，不是经验收益。关闭latent是分支使用诊断，不是物理合法的新构象。Oraclecoarse/fine含真实结构信息，仅用于误差定位。尚无同容量单层对照、独立测试或序列生成结果。']
    (run/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('TWO_LEVEL_AE_REPORT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
