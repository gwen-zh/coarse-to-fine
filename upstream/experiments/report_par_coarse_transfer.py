"""Report the full fixed pretrained-coarse transfer pilot, including regressions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_coarse_transfer import ARMS,identity
from run_author_codec_interface import METRIC_KEYS,GEOMETRY_KEYS,load
from run_par_coarse_transfer import ROOT,verify


def summary(rows):
    good=all(r['metrics']['metrics_available'] for r in rows)
    out=dict(metrics_available=good,failed_samples=sum(r['metrics']['failure_count'] for r in rows))
    if good:
        out.update({key:float(np.mean([r['metrics'][key] for r in rows])) for key in METRIC_KEYS})
        out['geometry']={key:float(np.mean([r['metrics']['geometry'][key] for r in rows])) for key in GEOMETRY_KEYS}
    if all(r['coarse_metrics'] is not None for r in rows):
        out['coarse']={key:float(np.mean([r['coarse_metrics'][key] for r in rows])) for key in [
            'mean_sample_nearest_reference_center_rmsd_A','mean_reference_nearest_sample_center_rmsd_A',
            'generated_conditional_span_flag_rate','generated_spread_center_rmsd_A']}
    return out


def main():
    p=load(ROOT/'protocol.json');verify(p);all_rows=[];paired=[];runs=[]
    for seed in p['seeds']:
        traces=[]
        for arm in ARMS:
            folder=ROOT/arm/f'seed{seed}';done=load(folder/'complete.json');evaluated=load(folder/'evaluation.json')
            if done['updates']!=512 or not done['official_weights_unchanged'] or len(evaluated['rows'])!=8:
                raise RuntimeError('Incomplete fixed pretrained transfer run')
            trace=load(folder/'trace.json')
            if len(trace)!=512:raise RuntimeError('Incomplete native transfer trace')
            traces.append([identity(r) for r in trace]);runs.append(done);all_rows.extend(evaluated['rows'])
        if traces[0]!=traces[1]:raise RuntimeError('Actual native cases/noise/time/rotation/SC/LR differ between controls')
        for target in p['targets']:
            for steps in p['sampling']:
                paths=[ROOT/arm/f"seed{seed}/step0/{target['protein']}/coarse_nfe{steps}.npz" for arm in ARMS]
                with np.load(paths[0]) as z:a=z['coarse'].copy()
                with np.load(paths[1]) as z:b=z['coarse'].copy()
                np.testing.assert_array_equal(a,b)
        paired.append(dict(seed=seed,all512_training_events_paired=True,initial_coarse_samples_bitwise_equal=True))
    result={}
    for steps in [int(k) for k in p['sampling']]:
        result[str(steps)]={}
        for arm in ARMS:
            result[str(steps)][arm]={str(endpoint):summary([r for r in all_rows if r['steps']==steps and r['arm']==arm and r['endpoint']==endpoint]) for endpoint in [0,512]}
    report=dict(status='complete_awaiting_review',summary=result,rows=all_rows,runs=runs,paired_audit=paired,
        scope=p['scope'],primary_steps=16,diagnostic_steps=400,model_default_replaced=False,
        test_coordinates_opened=False,R3_opened=False,independent_test_claim=False,
        protocol_sha256=sha(ROOT/'protocol.json'))
    panels=[('mean_sample_nearest_reference_ca_rmsd_A','Reference CA RMSD (Å)',False,1),
        ('mean_reference_nearest_sample_ca_rmsd_A','Reverse coverage error (Å)',False,1),
        ('cn_rate','C–N outliers (%)',True,100)]
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row_index,steps in enumerate(['16','400']):
        selected=[result[steps]['sequence']['0'],result[steps]['position_only']['512'],result[steps]['sequence']['512']]
        for ax,(key,label,geometry,scale) in zip(axes[row_index],panels):
            values=[(r['geometry'] if geometry else r)[key]*scale if r['metrics_available'] else np.nan for r in selected]
            ax.bar(range(3),values,color=['#777777','#699c74','#3577a7'])
            for i,v in enumerate(values):
                if np.isfinite(v):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
            ax.set_xticks(range(3),['Zero init','Position only','AA condition'],fontsize=9);ax.set_ylabel(label)
            ax.set_title(f'{steps} coarse FM steps');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen pretrained PAR coarse backbone + small sequence adapter\n'
        '512 matched native updates · 2 seeds · 2 reused development proteins · own fine and decoder fixed',fontsize=12)
    folder=Path('paper/figures/par_coarse_transfer_v1');folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(folder/('PAR_Coarse_Transfer.'+ext),dpi=170)
    plt.close(fig);report['figure']=str(folder/'PAR_Coarse_Transfer.pdf')
    atomic_json(ROOT/'report.json',report);print('PAR_COARSE_TRANSFER_REPORT',json.dumps(result),flush=True)


if __name__=='__main__':main()
