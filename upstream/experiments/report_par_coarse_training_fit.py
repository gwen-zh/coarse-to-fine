"""Report training fit separately from reused development, without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_par_coarse_training_fit import ROOT,PARENT,read
from run_par_coarse_transfer import verify


DOMAINS=['PDB_train','MD_train','development']
KEYS=['mean_sample_nearest_reference_center_rmsd_A','mean_reference_nearest_sample_center_rmsd_A',
    'generated_spread_center_rmsd_A','reference_spread_center_rmsd_A','js_pairwise_center_distance_nats',
    'mean_generated_center_rg_A','mean_reference_center_rg_A','generated_conditional_span_flag_rate']


def summarize(rows,probes,p):
    out={}
    for domain in DOMAINS:
        out[domain]={}
        for arm in p['arms']:
            out[domain][arm]={}
            for ep in p['endpoints']:
                rr=[r for r in rows if r['domain']==domain and r['arm']==arm and r['endpoint']==ep]
                value=dict(proteins_per_seed=len(rr)//len(p['seeds']),failed_samples=sum(r['failure_count'] for r in rr),
                    metrics_available=all(r['metrics'] is not None for r in rr))
                if value['metrics_available']:
                    value.update({key:float(np.mean([r['metrics'][key] for r in rr])) for key in KEYS})
                value['native_denoising_probe']={}
                for t in p['probe_times']:
                    selected=[r for r in probes if r['domain']==domain and r['arm']==arm and r['endpoint']==ep and r['time']==t]
                    groups={(r['seed'],r['protein']) for r in selected}
                    means=[]
                    for seed,protein in sorted(groups):
                        parts=[r for r in selected if r['seed']==seed and r['protein']==protein]
                        means.append({key:float(np.mean([r[key] for r in parts])) for key in ['velocity_MSE_nm2','relative_MSE','zero_velocity_MSE_nm2']})
                    value['native_denoising_probe'][str(t)]={key:float(np.mean([r[key] for r in means])) for key in means[0]}
                out[domain][arm][str(ep)]=value
    return out


def main():
    p=read(ROOT/'protocol.json');verify(p);rows=[];probes=[];audits=[]
    expected_probes=len(p['probe_case_order'])*len(p['probe_times'])*len(p['probe_replicates'])*2
    for seed in p['seeds']:
        draw_keys=[]
        for arm in p['arms']:
            folder=ROOT/arm/f'seed{seed}';done=read(folder/'complete.json');result=read(folder/'evaluation.json')
            if done['optimizer_updates']!=0 or not done['weights_unchanged'] or len(result['rows'])!=18 or len(result['probes'])!=expected_probes:
                raise RuntimeError('Incomplete fixed training-fit diagnosis')
            rows.extend(result['rows']);probes.extend(result['probes'])
            per_ep=[]
            for ep in p['endpoints']:
                per_ep.append([(r['case_id'],r['time'],r['replicate'],r['draw_sha256'],r['self_condition'],r['zero_velocity_MSE_nm2'])
                    for r in result['probes'] if r['endpoint']==ep])
            if per_ep[0]!=per_ep[1]: raise RuntimeError('Native probe draws differ by endpoint')
            draw_keys.append(per_ep[0])
        if draw_keys[0]!=draw_keys[1]: raise RuntimeError('Native probe draws differ by arm')
        for target in p['targets']:
            initial=[]
            for arm in p['arms']:
                with np.load(ROOT/arm/f"seed{seed}/step0/{target['protein']}.npz") as z:initial.append(z['coarse'].copy())
                if target['domain']=='development':
                    for ep in p['endpoints']:
                        with np.load(ROOT/arm/f"seed{seed}/step{ep}/{target['protein']}.npz") as a, np.load(PARENT/arm/f"seed{seed}/step{ep}/{target['protein']}/coarse_nfe16.npz") as b:
                            np.testing.assert_array_equal(a['coarse'],b['coarse'][:16])
            np.testing.assert_array_equal(*initial)
        audits.append(dict(seed=seed,all_native_probe_draws_paired=True,zero_initial_training_coarse_exact=True,
            development_prefixes_exact=True))
    summary=summarize(rows,probes,p)
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    colors=['#777777','#699c74','#3577a7'];labels=['Zero init','Position only','AA condition']
    choices=[('sequence','0'),('position_only','512'),('sequence','512')]
    for ax,key,label in zip(axes[0],KEYS[:2]+[KEYS[4]],['Coarse reference RMSD (Å)','Reverse coverage error (Å)','Pair-distance JS divergence (nats)']):
        for j,(arm,ep) in enumerate(choices):
            yy=[summary[d][arm][ep][key] if summary[d][arm][ep]['metrics_available'] else np.nan for d in DOMAINS]
            ax.bar(np.arange(3)+(j-1)*.24,yy,width=.24,color=colors[j],label=labels[j])
        ax.set_xticks(range(3),['PDB train\n1 reference','MD train\n128 references','Reused dev\n128 references'])
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8)
    for ax,domain in zip(axes[1],DOMAINS):
        for j,(arm,ep) in enumerate(choices):
            ax.plot(p['probe_times'],[summary[domain][arm][ep]['native_denoising_probe'][str(t)]['relative_MSE'] for t in p['probe_times']],
                marker='o',color=colors[j],label=labels[j])
        ax.set_title(domain.replace('_',' '));ax.set_xlabel('Native corruption time t');ax.set_ylabel('Velocity MSE / zero-field MSE')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Read-only PAR coarse fit diagnosis · 16 samples / 16 FM steps · 2 seeds\n'
        '5 presented MD train proteins + 2 presented PDB proteins + 2 reused dev proteins; bottom: native-conditioned probes, not generation',fontsize=11)
    folder=Path('paper/figures/par_coarse_training_fit_v1');folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(folder/('PAR_Coarse_Training_Fit.'+ext),dpi=170)
    plt.close(fig)
    report=dict(status='complete_awaiting_review',summary=summary,rows=rows,probes=probes,paired_audit=audits,
        official_interface=read(ROOT/'official_interface_audit.json'),requested_coarse_predictions=1152,
        new_training_coarse_predictions=896,new_development_predictions=0,optimizer_updates=0,
        all_requested_failures=sum(r['failure_count'] for r in rows),protocol_sha256=sha(ROOT/'protocol.json'),
        model_default_replaced=False,test_coordinates_opened=False,R3_opened=False,independent_test_claim=False,
        scope=p['scope'],figure=str(folder/'PAR_Coarse_Training_Fit.pdf'))
    atomic_json(ROOT/'report.json',report)
    jobs=ROOT/'jobs.json'
    if jobs.is_file():
        value=read(jobs);value.update(status=report['status'],all_models_running=False,report_status='complete',figure=report['figure'])
        atomic_json(jobs,value)
        run_state=Path('artifacts/run_state.json');overall=read(run_state);overall['par_coarse_training_fit_v1']=value;atomic_json(run_state,overall)
    print('TRAINING_FIT_REPORT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
