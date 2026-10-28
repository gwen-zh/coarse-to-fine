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
        assert aggregates['rejections']['edit_cap_exceeded']==0
        summary[arm]=aggregates
    contrasts={}
    for first,second in protocol['contrasts']:
        contrasts[first+'_minus_'+second]={key:interval([per[first][p][key]['delta']-per[second][p][key]['delta'] for p in sorted(groups)],protocol['bootstrap_draws'],protocol['bootstrap_seed']) for key in metric_keys}
    limitations='Reused development only; descriptive unadjusted95% protein bootstrap. Report all four predeclared edit counts, no winner selection or independent confirmation. Nested proposal coupling, one stream per output; not a causal universal dose law. Equal decoder calls not end-to-end compute. Proposal cap alone does not guarantee coordinate feasibility. No ranking by references.'
    result=dict(status='complete',protocol_sha256=ph,scope=protocol['scope'],proteins=24,samples=192,arms=summary,contrasts=contrasts,records=rows,limitations=limitations)
    for arm,v in summary.items():
        dose=int(arm.split('_')[-1])
        assert set(v['proposal_edit_histogram'])=={str(dose)}
        v['candidate_edits']=dose
        v['mean_final_edit_count']=dose*(1-v['returned_original']/v['samples'])
    result['dose_note']='Candidate dose is exact. Final selected output edits are either0or dose; all outputs includingrollbacks/zero budgets retained. No reference-based selection.'
    atomic_json(a.run/'comparison.json',result)
    with (a.run/'output_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['固定候选编辑数量：完整开发集结果','='*60,'','24 蛋白×8 输出；每组 8,625 次 decoder 调用。未使用 64/58 确认集。','', '四组均为同一样本上同 CPU 主机、FP32 解码；参考仅在全部候选选择结束后读取。','', '各组相对原输出的变化','-'*40,'']
    for arm,v in summary.items():
        lines.append(arm)
        for key in metric_keys:
            x=v[key];s=x['paired_delta'];lines.append(f"  {key}: {x['before']:.6f} → {x['after']:.6f}; Δ {s['mean']:.6f}, 描述性95%区间 {s['bootstrap95']}")
        lines.append(f"  候选编辑数 {v['candidate_edits']}；最终平均编辑数 {v['mean_final_edit_count']:.3f}；回退 {v['returned_original']}/192；均值搜索时间 {v['mean_search_seconds']:.3f}s；编辑上限拒绝 {v['rejections']['edit_cap_exceeded']}；位移拒绝 {v['rejections']['displacement_exceeded']}")
        lines.append('')
    lines+=['按蛋白配对对照（负值表示前者该指标更低）','-'*60,'']
    for name,values in contrasts.items():
        lines.append(name)
        for key in ['ca_c_n_angle','c_n_ca_angle','reference_rmsd','spread']:
            s=values[key];lines.append(f"  {key}: {s['mean']:.6f}, 95%区间 {s['bootstrap95']}")
        lines.append('')
    lines+=['解释边界','-'*20,'',limitations,'','四档为候选的固定修改数；最终输出可能回退到零改动。不能把候选数当作成功修改数，也不能由开发曲线倒写原10%上限的选择过程。角度改善是未门控的观测；CN/碰撞改善受接受规则约束。任何新方案的独立确认仍需新的、未用于选点的数据。']
    (a.run/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key,label in zip(axes.flat,['ca_c_n_angle','c_n_ca_angle','reference_rmsd','spread'],['CA-C-N reference angle error (degrees)','C-N-CA reference angle error (degrees)','Reference CA RMSD (angstrom)','Mean pairwise CA RMSD (angstrom)']):
        x=protocol['doses'];stats=[summary['edits_'+str(k)][key]['paired_delta'] for k in x]
        means=np.array([v['mean'] for v in stats]);lo=np.array([v['bootstrap95'][0] for v in stats]);hi=np.array([v['bootstrap95'][1] for v in stats])
        ax.errorbar(x,means,yerr=[means-lo,hi-means],fmt='o-',capsize=4);ax.axhline(0,color='.6',linestyle='--');ax.set_xticks(x);ax.set_xlabel('Edits per proposed candidate');ax.set_ylabel('Change from initial output');ax.set_title(label)
    fig.suptitle('Reused development: fixed decoder calls, 10% edit cap, 1 angstrom drift cap\nDescriptive 95% protein-bootstrap intervals; all rollbacks retained')
    for ext in ['png','pdf','svg']:fig.savefig(a.run/('dose_curve.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('EDIT_DOSE_REPORT_COMPLETE',len(raw),len(summary),'calls_per_arm',8625,flush=True)
    print(json.dumps({k:{m:v[m]['paired_delta'] for m in ['ca_c_n_angle','c_n_ca_angle','reference_rmsd','cn_rate','spread']} for k,v in summary.items()},indent=2))

if __name__=='__main__':main()
