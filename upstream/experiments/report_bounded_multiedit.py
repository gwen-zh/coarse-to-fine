"""Complete-cohort exploratory report; bootstrap proteins, never individual pairs."""
import argparse,csv,hashlib,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from structural_metrics import aligned_rmsd
from diagnostic_io import atomic_json


def interval(values,draws=20000,seed=20261026):
    x=np.asarray(values,float)
    if len(x)==0 or not np.isfinite(x).all():raise ValueError('Invalid paired values')
    rng=np.random.default_rng(seed);means=x[rng.integers(len(x),size=(draws,len(x)))].mean(1)
    return dict(mean=float(x.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist(),proteins=len(x))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());ph=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest()
    groups=defaultdict(list);rows=[];raw={};quality_keys={'ca_c_n_angle':'reference_ca_c_n_angle_mae_degrees','c_n_ca_angle':'reference_c_n_ca_angle_mae_degrees'}
    for item in protocol['targets']:
        d=json.loads((a.run/item['target']/'summary.json').read_text())
        assert d['status']=='complete' and d['protocol_sha256']==ph and set(d['arms'])==set(protocol['arms'])
        assert d['target']==item['target'];raw[item['target']]=d;groups[item['protein']].append(item)
        for arm,v in d['arms'].items():
            assert v['decoder_calls']==item['decoder_call_budget']==len(v['trace'])
            row=dict(protein=item['protein'],target=item['target'],arm=arm,calls=v['decoder_calls'],search_seconds=v['search_seconds'],returned_original=v['returned_original'])
            for label,key in quality_keys.items():
                row[label+'_before']=d['baseline_quality'][key];row[label+'_after']=v['posthoc_quality'][key]
            for label,key in [('reference_rmsd','closest_available_reference_ca_rmsd'),('cn_count','backbone_cn_outliers'),('ca_clashes','upstream_ca_clashes')]:
                row[label+'_before']=d['baseline'][key];row[label+'_after']=v['posthoc_metrics'][key]
            row['cn_rate_before']=100*row['cn_count_before']/(d['length']-1);row['cn_rate_after']=100*row['cn_count_after']/(d['length']-1)
            row['ca_density_before']=row['ca_clashes_before']/d['length'];row['ca_density_after']=row['ca_clashes_after']/d['length']
            row['edited_fraction']=v['gate']['changed_fraction'];row['ca_displacement']=v['gate']['aligned_ca_displacement']
            rows.append(row)
    assert len(groups)==24 and all(len(v)==8 for v in groups.values()) and len(raw)==192
    summary={};per={};metric_keys=['ca_c_n_angle','c_n_ca_angle','reference_rmsd','cn_rate','ca_density','spread']
    def spread(arrays):return float(np.mean([aligned_rmsd(x,y) for i,x in enumerate(arrays) for y in arrays[i+1:]]))
    for arm in protocol['arms']:
        per[arm]={}
        for protein,items in sorted(groups.items()):
            r=[x for x in rows if x['protein']==protein and x['arm']==arm];record=dict(protein=protein,samples=len(r))
            for key in metric_keys[:-1]:
                before=float(np.mean([x[key+'_before'] for x in r]));after=float(np.mean([x[key+'_after'] for x in r]))
                record[key]=dict(before=before,after=after,delta=after-before)
            base=[];selected=[]
            for item in items:
                with np.load(a.run/item['target']/'baseline.npz') as z:base.append(z['backbone'][:,1].copy())
                with np.load(a.run/item['target']/(arm+'.npz')) as z:selected.append(z['backbone'][:,1].copy())
            b,s=spread(base),spread(selected);record['spread']=dict(before=b,after=s,delta=s-b)
            per[arm][protein]=record
        aggregates={key:dict(before=float(np.mean([r[key]['before'] for r in per[arm].values()])),after=float(np.mean([r[key]['after'] for r in per[arm].values()])),paired_delta=interval([r[key]['delta'] for r in per[arm].values()],protocol['bootstrap_draws'],protocol['bootstrap_seed'])) for key in metric_keys}
        ar=[d['arms'][arm] for d in raw.values()];hist=Counter()
        for v in ar:hist.update(v['proposal_edit_histogram'])
        calls=sum(v['decoder_calls'] for v in ar);assert calls==8625
        aggregates.update(calls=calls,padding_calls=sum(v['padding_calls'] for v in ar),samples=len(ar),returned_original=sum(v['returned_original'] for v in ar),mean_search_seconds=float(np.mean([v['search_seconds'] for v in ar])),rejections={k:sum(v['rejection_diagnostics_overlapping'][k] for v in ar) for k in ar[0]['rejection_diagnostics_overlapping']},proposal_edit_histogram=dict(hist),mean_edited_fraction=float(np.mean([v['gate']['changed_fraction'] for v in ar])),mean_ca_displacement=float(np.mean([v['gate']['aligned_ca_displacement'] for v in ar])),per_protein=list(per[arm].values()))
        if arm.startswith('independent_multi_'):assert aggregates['rejections']['edit_cap_exceeded']==0
        summary[arm]=aggregates
    contrasts={}
    for first,second in protocol['contrasts']:
        contrasts[first+'_minus_'+second]={key:interval([per[first][p][key]['delta']-per[second][p][key]['delta'] for p in sorted(groups)],protocol['bootstrap_draws'],protocol['bootstrap_seed']) for key in metric_keys}
    limitations='Reused development only; descriptive unadjusted95% protein bootstrap. Report both predeclared multi-edit distributions, no winner selection or independent confirmation. Equal decoder calls not end-to-end compute. Proposal cap alone does not guarantee coordinate feasibility. No ranking by references.'
    result=dict(status='complete',protocol_sha256=ph,scope=protocol['scope'],proteins=24,samples=192,arms=summary,contrasts=contrasts,records=rows,limitations=limitations)
    atomic_json(a.run/'comparison.json',result)
    with (a.run/'output_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['受限多点候选对照：完整开发集结果','='*60,'','24 蛋白×8 输出；每组 8,625 次 decoder 调用。未使用 64/58 确认集。','', '五组均为同一样本上同 CPU 主机、FP32 解码；参考仅在全部候选选择结束后读取。','', '各组相对原输出的变化','-'*40,'']
    for arm,v in summary.items():
        lines.append(arm)
        for key in metric_keys:
            x=v[key];s=x['paired_delta'];lines.append(f"  {key}: {x['before']:.6f} → {x['after']:.6f}; Δ {s['mean']:.6f}, 描述性95%区间 {s['bootstrap95']}")
        lines.append(f"  回退 {v['returned_original']}/192；均值搜索时间 {v['mean_search_seconds']:.3f}s；编辑上限拒绝 {v['rejections']['edit_cap_exceeded']}；位移拒绝 {v['rejections']['displacement_exceeded']}")
        lines.append('')
    lines+=['按蛋白配对对照（负值表示前者该指标更低）','-'*60,'']
    for name,values in contrasts.items():
        lines.append(name)
        for key in ['ca_c_n_angle','c_n_ca_angle','reference_rmsd','spread']:
            s=values[key];lines.append(f"  {key}: {s['mean']:.6f}, 95%区间 {s['bootstrap95']}")
        lines.append('')
    lines+=['解释边界','-'*20,'',limitations,'','多点提案是两种固定分布，不代表穷尽最优 best-of-N。角度改善是未门控的观测；CN/碰撞改善受接受规则约束。任何新方案的独立确认仍需新的、未用于选点的数据。']
    (a.run/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    print('BOUNDED_MULTIEDIT_REPORT_COMPLETE',len(raw),len(summary),'calls_per_arm',8625,flush=True)
    print(json.dumps({k:{m:v[m]['paired_delta'] for m in ['ca_c_n_angle','c_n_ca_angle','reference_rmsd','cn_rate','spread']} for k,v in summary.items()},indent=2))

if __name__=='__main__':main()
