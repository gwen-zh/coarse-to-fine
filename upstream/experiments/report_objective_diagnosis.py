"""Exploratory A/C summaries and exportable figures, including contrary outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from likelihood_geometry import protein_interval
from diagnostic_io import atomic_json


def save_figure(figure,path):
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'),dpi=180)
    figure.savefig(path.with_suffix('.svg'))
    plt.close(figure)


def report_a(run,protocol,protocol_hash):
    records=[json.loads((run/'A'/(name+'.json')).read_text()) for name in protocol['geometry_targets']]
    if any(r['protocol_sha256']!=protocol_hash for r in records):raise ValueError('A protocol mismatch')
    correlations={}
    for temperature in protocol['A']['temperatures']:
        correlations[str(temperature)]={metric:protein_interval([r['correlations'][str(temperature)][metric] for r in records])
            for metric in ['cn_rate','ca_clashes_per_residue']}
    controls={}
    for source in ['argmax','encoded_reference']:
        rows=[next(v for v in r['records'] if v['source']==source) for r in records]
        controls[source]=dict(cn_rate=protein_interval([r['cn_rate'] for r in rows]),
            ca_clashes_per_residue=protein_interval([r['ca_clashes_per_residue'] for r in rows]),
            ca_validity=float(np.mean([r['metrics']['upstream_ca_clash_valid'] for r in rows])),
            mean_reference_rmsd=float(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in rows])))
    controls['experimental_reference']=dict(cn_rate=protein_interval([r['experimental_reference_metrics']['backbone_cn_outliers']/max(1,r['length']-1) for r in records]),
        ca_validity=float(np.mean([r['experimental_reference_metrics']['upstream_ca_clash_valid'] for r in records])))
    per_temperature={}
    for temperature in protocol['A']['temperatures']:
        rows=[[v for v in r['records'] if v['source']=='sampled' and v['temperature']==temperature] for r in records]
        per_temperature[str(temperature)]={metric:protein_interval([float(np.mean([v[metric] for v in group])) for group in rows])
            for metric in ['log_score','cn_rate','ca_clashes_per_residue']}
    result=dict(protocol_sha256=protocol_hash,proteins=len(records),samples_per_protein=48,
        primary_T1_score_cn_spearman=correlations['1.0']['cn_rate'],within_temperature_correlations=correlations,
        controls=controls,per_temperature_means=per_temperature,
        note='Conditional fine-only4096 head scores, not full diffusion likelihood. Temperature sampling is an independent-per-position fine-token diagnostic replacing the published fine argmax policy; original argmax is a separate control. Reused development targets; exploratory protein bootstrap conditional on sampled contexts. Undefined correlations excluded with counts. No pooled-temperature primary test or universal causal claim.')
    atomic_json(run/'A_summary.json',result)
    figure,axes=plt.subplots(1,2,figsize=(11,4.4))
    for row in records:
        values=[r for r in row['records'] if r['source']=='sampled' and r['temperature']==1.]
        x=np.asarray([r['log_score'] for r in values]);y=np.asarray([r['cn_rate'] for r in values])
        axes[0].scatter(x-x.mean(),100*(y-y.mean()),s=12,alpha=.35,color='#2463a5')
    axes[0].set(xlabel='Fixed T=1 mean log score, centered per protein',ylabel='C-N outlier rate deviation (percentage points)',title='T=1 samples; display only, no pooled correlation')
    rho=[r['correlations']['1.0']['cn_rate'] for r in records]
    for i,value in enumerate(rho):
        if value is not None:axes[1].scatter(value,i,color='#2463a5',s=22)
        else:axes[1].text(0,i,'undefined',fontsize=7,va='center')
    axes[1].axvline(0,color='gray',linewidth=1)
    axes[1].set_yticks(range(len(records)),[r['target'] for r in records],fontsize=6)
    axes[1].set(xlabel='Within-protein Spearman (score vs C-N errors)',title='Positive: higher score, more errors',xlim=(-1.1,1.1))
    save_figure(figure,run/'A_score_geometry')
    print('A_SCORE_GEOMETRY_REPORT',json.dumps(result),flush=True)
    return result


def report_c(run,protocol,protocol_hash):
    config=protocol['C'];records={};progress={}
    for arm in config['arms']:
        for seed in config['seeds']:
            directory=run/'C'/f'{arm}_seed{seed}'
            row=json.loads((directory/'geometry_summary.json').read_text())
            if row['protocol_sha256']!=protocol_hash or [r['step'] for r in row['snapshots']]!=config['snapshot_steps']:
                raise ValueError('C protocol or snapshots differ')
            records[(arm,seed)]=row
            progress[(arm,seed)]=json.loads((directory/'progress.json').read_text())
            for name in protocol['geometry_targets']:
                with np.load(run/'A'/(name+'.npz')) as a:original=a['fine'][0].copy();coarse=a['coarse'].copy()
                with np.load(directory/'predictions'/f'step0_{name}.npz') as a:
                    if not np.array_equal(a['fine'],original) or not np.array_equal(a['coarse'],coarse):
                        raise ValueError('A/C original no-update controls differ')
    contrasts={};delta_by_arm={};curves={}
    for arm in config['arms']:
        deltas=[];ce_deltas=[];curves[arm]=[]
        for seed in config['seeds']:
            snapshots=records[(arm,seed)]['snapshots']
            arrays=[{r['target']:r for r in point['targets']} for point in snapshots]
            deltas.append([arrays[-1][name]['cn_rate']-arrays[0][name]['cn_rate'] for name in protocol['geometry_targets']])
            ce_rows=[{r['target']:r for r in point['validation']} for point in progress[(arm,seed)]['history']]
            ce_deltas.append([ce_rows[-1][name]['ce']-ce_rows[0][name]['ce'] for name in protocol['geometry_targets']])
            monotonic=sum(all(arrays[i+1][name]['cn_rate']>=arrays[i][name]['cn_rate']-1e-12 for i in range(len(arrays)-1))
                for name in protocol['geometry_targets'])
            curves[arm].append(dict(seed=seed,cn_rate=[r['cn_rate'] for r in snapshots],
                validation_ce=[r['validation_ce'] for r in snapshots],train_ce_fixed16=[r['train_ce_fixed16'] for r in snapshots],
                ca_validity=[r['ca_validity'] for r in snapshots],nondecreasing_cn_targets=monotonic))
        delta_by_arm[arm]=np.asarray(deltas)
        ce_mean=np.asarray(ce_deltas).mean(0);cn_mean=delta_by_arm[arm].mean(0)
        contrasts[arm+'_final_minus_original']=dict(cn_rate=protein_interval(cn_mean),
            per_seed_mean_cn_change=delta_by_arm[arm].mean(1).tolist(),
            seed_averaged_ce_decreases_cn_increases_targets=int(((ce_mean<0)&(cn_mean>0)).sum()),
            seed_averaged_ce_decreases_cn_decreases_targets=int(((ce_mean<0)&(cn_mean<0)).sum()))
    contrasts['pure_ce_minus_ce_kl_cn_change']=protein_interval((delta_by_arm['ce']-delta_by_arm['ce_kl']).mean(0))
    result=dict(protocol_sha256=protocol_hash,proteins=24,seeds=config['seeds'],arms=config['arms'],
        snapshot_steps=config['snapshot_steps'],actual_epochs=[step/899 for step in config['snapshot_steps']],
        contrasts=contrasts,curves=curves,
        note='All predeclared checkpoints; no selection. Bootstrap unit is protein after averaging3 seeds, conditional on those seeds and cached contexts. Seed curves and monotonicity descriptive. Reused development validation, not independent confirmation or proof of universal causation.')
    atomic_json(run/'C_summary.json',result)
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    colors=dict(ce='#b44a36',ce_kl='#2463a5');epochs=result['actual_epochs']
    for arm in config['arms']:
        for row in curves[arm]:
            axes[0].plot(epochs,row['validation_ce'],color=colors[arm],alpha=.3,linewidth=1)
            axes[1].plot(epochs,100*np.asarray(row['cn_rate']),color=colors[arm],alpha=.3,linewidth=1)
        axes[0].plot(epochs,np.mean([r['validation_ce'] for r in curves[arm]],axis=0),label=arm,color=colors[arm])
        axes[1].plot(epochs,100*np.mean([r['cn_rate'] for r in curves[arm]],axis=0),label=arm,color=colors[arm])
    baseline_ce=curves['ce'][0]['validation_ce'][0];baseline_cn=curves['ce'][0]['cn_rate'][0]
    axes[0].axhline(baseline_ce,color='black',linestyle='--',label='No update')
    axes[1].axhline(100*baseline_cn,color='black',linestyle='--',label='No update')
    axes[0].set(xlabel='Training epochs',ylabel='Reference-token validation CE (101 proteins)')
    axes[1].set(xlabel='Training epochs',ylabel='Mean C-N outlier rate %, 24 proteins')
    for ax in axes:ax.legend(fontsize=8)
    save_figure(figure,run/'C_training_geometry')
    print('C_OBJECTIVE_TRAJECTORY_REPORT',json.dumps(contrasts),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--part',choices=['A','C','all'],required=True)
    args=p.parse_args();path=args.run/'protocol.json';protocol=json.loads(path.read_text())
    protocol_hash=hashlib.sha256(path.read_bytes()).hexdigest();results={}
    if args.part in ['A','all']:results['A']=report_a(args.run,protocol,protocol_hash)
    if args.part in ['C','all']:results['C']=report_c(args.run,protocol,protocol_hash)
    if args.part=='all':atomic_json(args.run/'summary.json',dict(status='complete',protocol_sha256=protocol_hash,results=results))


if __name__=='__main__':main()
