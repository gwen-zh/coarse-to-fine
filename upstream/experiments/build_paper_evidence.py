"""Build paper tables/figures from frozen outputs; never selects models or reruns inference."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper'

def read(path):return json.loads((ROOT/path).read_text())
def save_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def csv_file(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired(values,n=20000,seed=20261026):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    means=x[rng.integers(len(x),size=(n,len(x)))].mean(1)
    return dict(mean=float(x.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist(),proteins=len(x))
def export(fig,name):
    for ext in ('pdf','svg','png'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    for p in ('figures','tables','protocols'):(OUT/p).mkdir(parents=True,exist_ok=True)
    sources=['artifacts/fine_search_confirmation_v1/primary_confirmation.json','artifacts/fine_search_confirmation_v1/independent_backbone_quality.json','artifacts/fine_search_confirmation_v1/comparison.json','artifacts/fine_search_confirmation_v1/analysis_summary.json','artifacts/fine_search_confirmation_v1/evaluation_plan.json','artifacts/fine_search_confirmation_v1/protocol.json','artifacts/development/matched_bestofn_v1/comparison.json','artifacts/development/native_search_latency_v1/comparison.json','artifacts/development/medium_candidate_probe_v1/gpu_replay.json','artifacts/development/latent_flow_probe_20epoch_v1/comparison.json']
    primary,quality,comparison,analysis,plan,protocol,matched,latency,gpu,latent=map(read,sources)
    groups=defaultdict(list)
    for r in quality['records']:groups[r['protein']].append(r)
    proteins=sorted(groups);assert len(proteins)==64 and len(quality['records'])==512
    assert len({r['target'] for r in quality['records']})==512 and all(len(v)==8 for v in groups.values())
    endpoints=['reference_ca_c_n_angle_mae_degrees','reference_c_n_ca_angle_mae_degrees']
    short=['CA-C-N angle MAE','C-N-CA angle MAE'];per=[];table=[];delta={}
    for metric,label in zip(endpoints,short):
        for arm in ('discrete_topk','discrete_random_sites'):
            b=np.array([np.mean([r['metrics']['baseline'][metric] for r in groups[p]]) for p in proteins])
            s=np.array([np.mean([r['metrics'][arm][metric] for r in groups[p]]) for p in proteins])
            delta[metric,arm]=s-b
            for i,p in enumerate(proteins):per.append(dict(protein=p,arm=arm,metric=metric,baseline=float(b[i]),selected=float(s[i]),delta=float(s[i]-b[i]),samples=8))
            if arm=='discrete_topk':
                stats=primary['primary'][metric]
                rng=np.random.default_rng(2026);draws=(s-b)[rng.integers(64,size=(20000,64))].mean(1)
                np.testing.assert_allclose(np.quantile(draws,[.0125,.9875]),stats['bootstrap97_5'],atol=1e-12,rtol=0)
                np.testing.assert_allclose((s-b).mean(),stats['mean_delta'],atol=1e-12,rtol=0)
                ci=stats['bootstrap97_5'];level=97.5;role='prespecified primary'
            else:
                stats=quality['arms'][arm][metric]['paired_delta'];ci=stats['bootstrap95'];level=95;role='secondary random-site control'
            table.append(dict(cohort='confirmation64x8',arm=arm,metric=label,unit='degrees',role=role,baseline=float(b.mean()),selected=float(s.mean()),delta=float((s-b).mean()),ci_level=level,ci_low=ci[0],ci_high=ci[1]))
    for arm in ('discrete_topk','discrete_random_sites'):
        a=analysis['arms'][arm];c=comparison['arms'][arm]
        for label,b,s,stat in [('Reference CA RMSD',a['ref_before'],a['ref_after'],c['paired_reference_rmsd_delta']),('Mean pairwise CA RMSD',a['baseline_mean_spread'],a['selected_mean_spread'],c['paired_spread_delta'])]:
            table.append(dict(cohort='confirmation64x8',arm=arm,metric=label,unit='angstrom',role='secondary;not equivalence test',baseline=b,selected=s,delta=stat['mean'],ci_level=95,ci_low=stat['bootstrap95'][0],ci_high=stat['bootstrap95'][1]))
        table.append(dict(cohort='confirmation64x8',arm=arm,metric='CN outlier rate',unit='percent / delta percentage-points',role='gated proxy;descriptive',baseline=a['cn_before'],selected=a['cn_after'],delta=a['cn_after']-a['cn_before'],ci_level='',ci_low='',ci_high=''))
    csv_file(OUT/'tables/confirmation_endpoints.csv',table);csv_file(OUT/'tables/confirmation_protein_angles.csv',per)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),gridspec_kw={'width_ratios':[1.65,1,1]},layout='constrained')
    rng=np.random.default_rng(19)
    ax=axes[0]
    for y,(metric,label) in enumerate(zip(endpoints,short)):
        x=delta[metric,'discrete_topk'];stats=primary['primary'][metric];lo,hi=stats['bootstrap97_5'];m=stats['mean_delta']
        ax.scatter(x,y+rng.uniform(-.13,.13,len(x)),s=12,c='#97b8c7',alpha=.65,zorder=1)
        ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#004c6d',capsize=4,zorder=3)
    ax.axvline(0,color='.5',ls='--',lw=1);ax.set_yticks([0,1],short);ax.set_xlabel('Search minus original (degrees)');ax.set_title('A  Primary endpoints\n97.5% nominal intervals')
    for ax,key,title,label in [(axes[1],'reference_rmsd_delta','B  Reference error','CA RMSD change (Å)'),(axes[2],'spread_delta','C  Sample spread','Pairwise CA RMSD change (Å)')]:
        rows=comparison['arms']['discrete_topk']['per_protein'];x=np.array([r[key] for r in rows]);stat=comparison['arms']['discrete_topk']['paired_reference_rmsd_delta' if key.startswith('reference') else 'paired_spread_delta'];m=stat['mean'];lo,hi=stat['bootstrap95']
        ax.scatter(x,rng.uniform(-.1,.1,len(x)),s=12,c='#97b8c7',alpha=.65);ax.errorbar(m,0,xerr=[[m-lo],[hi-m]],fmt='o',color='#004c6d',capsize=4);ax.axvline(0,color='.5',ls='--');ax.set_yticks([]);ax.set_xlabel(label);ax.set_title(title+'\n95% descriptive interval')
    fig.suptitle('64 confirmation proteins × 8 outputs; points are protein means',fontsize=12)
    export(fig,'figure1_confirmation')
    order=['global_temperature_0.5','global_temperature_1.0','local_independent','local_greedy','existing_targeted'];labels=['Whole-sequence T=0.5','Whole-sequence T=1.0','Independent single edit','Accumulating random edits','Violation-site search']
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');dev=[]
    for ax,metric,title,unit in zip(axes,['cn_delta_pp','ca_c_n_angle_delta','c_n_ca_angle_delta'],['CN outliers (gated)','CA-C-N error (ungated)','C-N-CA error (ungated)'],['percentage points','degrees','degrees']):
        for y,arm in enumerate(order):
            s=matched['arms'][arm][metric];m=s['mean'];lo,hi=s['bootstrap95'];assert matched['arms'][arm]['calls']==8625
            ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',color='#00796b',capsize=3)
            dev.append(dict(cohort='reused_development24x8',arm=arm,metric=metric,delta=m,ci_low=lo,ci_high=hi,calls=8625,returned_original=matched['arms'][arm]['returned_original']))
        ax.axvline(0,color='.6',ls='--');ax.set_title(title);ax.set_xlabel('Change from original ('+unit+')');ax.set_yticks(range(5),labels if ax is axes[0] else ['']*5);ax.invert_yaxis()
    fig.suptitle('Development controls: 8,625 decoder calls per method; 95% descriptive intervals',fontsize=12)
    export(fig,'figure2_development_controls');csv_file(OUT/'tables/development_controls.csv',dev)
    # New posthoc descriptive analysis, explicitly separate from the frozen primaries.
    spread={};spreadrows=[]
    fig,ax=plt.subplots(figsize=(6.8,3.8),layout='constrained')
    for ai,arm in enumerate(('discrete_topk','discrete_random_sites')):
        rows=comparison['arms'][arm]['per_protein'];assert len(rows)==64 and all(r['samples']==8 for r in rows)
        spread[arm]={}
        for j,q in enumerate((.1,.5,.9)):
            x=np.array([r['selected_pairwise_ca_rmsd_quantiles'][j]-r['baseline_pairwise_ca_rmsd_quantiles'][j] for r in rows]);s=paired(x);spread[arm][str(q)]=s
            y=j+(ai-.5)*.15;m=s['mean'];lo,hi=s['bootstrap95'];ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='o',capsize=3,color=['#004c6d','#c47135'][ai],label=['Violation-site search','Random-site search'][ai] if j==0 else None)
            for r,v in zip(rows,x):spreadrows.append(dict(protein=r['protein'],arm=arm,quantile=q,delta=float(v),samples=8,pairs=28))
    ax.axvline(0,color='.6',ls='--');ax.set_yticks(range(3),['Within-protein 10th percentile','Within-protein median','Within-protein 90th percentile']);ax.set_xlabel('Pairwise CA-distance quantile change (Å)');ax.legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,-.22),ncol=2);ax.set_title('Posthoc spread sensitivity; protein bootstrap\nDescriptive 95% intervals, not ensemble-equivalence tests')
    export(fig,'figure3_spread_sensitivity');csv_file(OUT/'tables/posthoc_spread_quantiles.csv',spreadrows)
    save_json(OUT/'tables/posthoc_spread_summary.json',dict(scope='Posthoc on consumed confirmation cohort. No tuning or new confirmation;28pair distances/protein are dependent;resample64proteins,20000draws,seed20261026.',arms=spread))
    inventory=[]
    for name,m in latent['methods'].items():
        t=m['training'];inventory.append(dict(method=name,parameters=t['parameters'],selected_epoch=t['selected_epoch'],training_proteins=t['train_proteins'],validation_proteins=t['validation_proteins'],geometry_proteins=24,reference_rmsd=m['mean_reference_ca_rmsd'],native_baseline_rmsd=m['mean_original_native_ca_rmsd'],cn_rate=m['mean_cn_rate'],sample_count_per_protein=2 if name=='fm' else 1,head_forwards_per_sample=16 if name=='fm' else 1,scope='reused development; frozen sampled-coarse features;not compute matched'))
    csv_file(OUT/'tables/continuous_baseline_inventory.csv',inventory)
    stats_audit={'primary_recomputed_matches_saved':True,'proteins':64,'samples':512,'samples_per_protein':8,'primary_bootstrap':{'draws':20000,'seed':2026,'interval':97.5},'secondary_original_report':{'draws':10000,'seed':2026,'interval':95},'discrepancy':'Evaluation plan broadly states20000bootstrap; implemented secondary RMSD/spread report uses10000. Preserve results, disclose discrepancy. Primary implementation matches plan. New quantile analysis is posthoc,20000draws,seed20261026.','no_model_or_threshold_selection':True}
    save_json(OUT/'tables/evidence_audit.json',stats_audit)
    sources += ['artifacts/fine_search_confirmation_v1/manifest.json','artifacts/development/medium_candidate_probe_v1/comparison.json','upstream/experiments/report_bounded_search_confirmation.py','upstream/experiments/fine_search_constraints.py','upstream/experiments/bounded_fine_search_ensemble.py']
    manifest=dict(scope='Paper build from frozen results; no inference/training/model selection.',sources=[dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()) for p in sources],builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(OUT/'evidence_manifest.json',manifest)
    print('PAPER_EVIDENCE_BUILT',len(table),'endpoint rows;3 figures PDF/SVG/PNG;primary bootstrap reproduced')
    print('POSTHOC_SPREAD',json.dumps(spread))

if __name__=='__main__':main()
