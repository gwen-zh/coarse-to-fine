"""Report every matched arm/seed; distinguish representation signal from deployment."""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    run=Path('artifacts/development/multiscale_discrete_v1');p=json.loads((run/'protocol.json').read_text());groups={};checks=[]
    expected={r['target'] for r in p['targets'] if r['subset']=='validation'}
    for seed in p['seeds']:
        groups[str(seed)]={}
        for arm in p['arms']:
            data=json.loads((run/f'{arm}_{seed}/evaluation.json').read_text())
            if data['status']!='complete' or {r['target'] for r in data['targets']}!=expected or len(data['targets'])!=len(expected):raise RuntimeError('Incomplete or duplicate evaluation')
            methods={}
            for method in data['targets'][0]['methods']:
                methods[method]={k:float(np.mean([x['methods'][method][k] for x in data['targets']])) for k in ['ca_rmsd','cn_rate','ca_clashes_per_residue','long_range_distance_mae','hidden_accuracy']}
            baseline={k:float(np.mean([x['baseline'][k] for x in data['targets']])) for k in ['ca_rmsd','cn_rate','ca_clashes_per_residue']}
            groups[str(seed)][arm]=dict(methods=methods,baseline=baseline,parameters=data['parameters'],code_counts=data['code_counts'])
        arms=groups[str(seed)];mixed=arms['mixed'];m=mixed['methods']['intact'];controls=[arms[a]['methods']['intact'] for a in ['local','global']]
        signal=dict(rmsd_5pct_better_than_both=all(m['ca_rmsd']<=.95*c['ca_rmsd'] for c in controls),cn_matched_and_below10pct=m['cn_rate']<=.10 and all(m['cn_rate']<=c['cn_rate']+.005 for c in controls),clashes_matched=all(m['ca_clashes_per_residue']<=c['ca_clashes_per_residue']+.005 for c in controls),all_branches_used=all(np.count_nonzero(c)>=2 for c in mixed['code_counts']),shuffling_degrades=mixed['methods']['shuffled_all']['ca_rmsd']>m['ca_rmsd'])
        native=dict(rmsd_within_125pct=m['ca_rmsd']<=1.25*mixed['baseline']['ca_rmsd'],cn_within_half_pp=m['cn_rate']<=mixed['baseline']['cn_rate']+.005)
        checks.append(dict(seed=seed,representation_signal=signal,representation_signal_pass=all(signal.values()),native_compatibility=native,native_compatibility_pass=all(native.values())))
    result=dict(status='complete',groups=groups,gates=checks,both_seeds_representation_signal=all(x['representation_signal_pass'] for x in checks),both_seeds_native_compatibility=all(x['native_compatibility_pass'] for x in checks),scope=p['scope'],note='Geometry-only encoder and original-discrete completion; not directly comparable to old continuous-embedding half-fine runs. No automatic generator training or extra epochs.')
    atomic_json(run/'comparison.json',result)
    lines=['多尺度离散表示对照','='*36,'',p['scope'],'']
    for seed,arms in groups.items():
        for arm,x in arms.items():
            m=x['methods']['intact'];lines.append(f"seed{seed} {arm}: RMSD {m['ca_rmsd']:.4f} Å，CN {100*m['cn_rate']:.2f}%，隐藏fine准确率{100*m['hidden_accuracy']:.2f}%；完整fine基线{ x['baseline']['ca_rmsd']:.4f} Å。")
    lines+=['',f"两个种子均有表征信号：{result['both_seeds_representation_signal']}；均过原fine接口质量门槛：{result['both_seeds_native_compatibility']}。",'',result['note']]
    (run/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('MULTISCALE_DISCRETE_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
