"""Report every declared two-protein overfit endpoint and matched solver probe."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coarse_generation_diagnostic import weighted_rmsd_matrix
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_two_protein_fit import ARMS, identity, sample_keys
from run_par_two_protein_fit import ROOT,read,cases_and_targets
from run_par_coarse_transfer import verify


def measure(values,refs,length):
    if not np.isfinite(values).all():
        return dict(metrics_available=False,failure_count=int((~np.isfinite(values).all(axis=(1,2))).sum()))
    distances=[weighted_rmsd_matrix(values,ref[None],length)[:,0] for ref in refs]
    spread=weighted_rmsd_matrix(values,values,length)
    indices=np.triu_indices(len(values),1)
    return dict(metrics_available=True,failure_count=0,samples=len(values),
        mean_distance_to_targets_A=[float(v.mean()) for v in distances],
        nearest_sample_to_targets_A=[float(v.min()) for v in distances],
        generated_spread_A=float(spread[indices].mean()))


def main():
    p=read(ROOT/'protocol.json');verify(p)
    cases,targets=cases_and_targets();refs=[c['coarse'].numpy() for c in cases]
    rows=[];traces={};native_probes={};arrays={};completed=[]
    for arm in ARMS:
        folder=ROOT/arm;done=read(folder/'complete.json')
        trace=read(folder/'trace.json')
        if done['updates']!=1024 or not done['official_parameters_unchanged'] or len(trace)!=1024:
            raise RuntimeError('An arm did not finish the fixed budget')
        completed.append(done);traces[arm]=trace;native_probes[arm]=read(folder/'native_probes.json')
        for endpoint in p['endpoints']:
            budgets=[p['primary']]+([p['diagnostic']] if endpoint in p['diagnostic']['endpoints'] else [])
            for budget in budgets:
                for i,target in enumerate(targets):
                    file=folder/f"step{endpoint}/{target['protein']}/coarse_nfe{budget['steps']}.npz"
                    with np.load(file) as z:
                        values=z['coarse'].copy()
                        np.testing.assert_array_equal(z['seeds'],np.asarray(sample_keys(p['seed'],budget['samples']),dtype=np.uint32))
                    if values.shape!=(budget['samples'],23,3):raise RuntimeError('Requested output missing')
                    arrays[arm,endpoint,budget['steps'],i]=values
                    rows.append(dict(arm=arm,endpoint=endpoint,steps=budget['steps'],protein=target['protein'],
                        condition_index=i,metrics=measure(values,refs,90)))
    paired=[identity(r) for r in next(iter(traces.values()))]
    if any([identity(r) for r in trace]!=paired for trace in traces.values()):
        raise RuntimeError('Base data/noise/rotation/time-uniform draws/SC/LR differ between arms')
    for rule in ['beta','uniform']:
        a,b=[traces[arm] for arm,(mode,t) in ARMS.items() if t==rule]
        if [[e['time'] for e in r['batch']] for r in a]!=[[e['time'] for e in r['batch']] for r in b]:
            raise RuntimeError('Time differs within a sequence-control pair')
    for arm,trace in traces.items():
        for row in trace:
            for e in row['batch']:
                expected=e['uniform_time_draw']
                if ARMS[arm][1]=='beta' and e['mixture_coin']>=.02:expected=expected**(1/1.9)
                if abs(expected-e['time'])>1e-7:raise RuntimeError('Wrong realized time distribution')
    for steps in [16,64]:
        for i in range(2):
            initial=[arrays[arm,0,steps,i] for arm in ARMS]
            for a in initial[1:]:np.testing.assert_array_equal(initial[0],a)
    for arm,(mode,rule) in ARMS.items():
        for endpoint in p['endpoints']:
            for steps in ([16,64] if endpoint in [0,1024] else [16]):
                a,b=[arrays[arm,endpoint,steps,i] for i in range(2)]
                if mode=='position_only':np.testing.assert_array_equal(a,b)
    curve={arm:{} for arm in ARMS};solver={};bound=p['length_only_macro_RMSD_lower_bound_A']
    for arm in ARMS:
        for endpoint in p['endpoints']:
            rr=[r for r in rows if r['arm']==arm and r['endpoint']==endpoint and r['steps']==16]
            valid=all(r['metrics']['metrics_available'] for r in rr)
            value=dict(metrics_available=valid)
            if valid:
                error=[r['metrics']['mean_distance_to_targets_A'][r['condition_index']] for r in rr]
                wrong=[r['metrics']['mean_distance_to_targets_A'][1-r['condition_index']] for r in rr]
                value.update(per_protein_coarse_RMSD_A=error,macro_coarse_RMSD_A=float(np.mean(error)),
                    cross_minus_correct_margin_A=float(np.mean(wrong)-np.mean(error)),
                    both_training_targets_within_2A=bool(max(error)<=2.),
                    below_length_only_lower_bound=bool(np.mean(error)<bound-1e-5))
                if ARMS[arm][0]=='position_only' and value['macro_coarse_RMSD_A']<bound-1e-5:
                    raise RuntimeError('Length-only prediction violated its geometric lower bound')
            curve[arm][str(endpoint)]=value
        solver[arm]={}
        for steps in [16,64]:
            values=[measure(arrays[arm,1024,steps,i][:8],refs,90) for i in range(2)]
            available=all(v['metrics_available'] for v in values)
            solver[arm][str(steps)]=dict(metrics_available=available,
                macro_coarse_RMSD_A=float(np.mean([v['mean_distance_to_targets_A'][i] for i,v in enumerate(values)])) if available else None,
                samples_per_protein=8)
    probes={}
    first_identity=None
    for arm,record in native_probes.items():
        probes[arm]={}
        for ep in ['0','256','1024']:
            same=[{k:v for k,v in r.items() if k!='velocity_MSE_nm2'} for r in record[ep]]
            if first_identity is None:first_identity=same
            if first_identity!=same:raise RuntimeError('Native denoising controls used different corruptions')
            probes[arm][ep]={str(t):float(np.mean([r['velocity_MSE_nm2'] for r in record[ep] if r['time']==t])) for t in [.05,.5,.9]}
    colors=['#777777','#3577a7','#b18754','#4a9a72'];styles=['--','-','--','-']
    labels=['No AA · beta','AA · beta','No AA · uniform','AA · uniform']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for j,arm in enumerate(ARMS):
        yy=[curve[arm][str(ep)].get('macro_coarse_RMSD_A',np.nan) for ep in p['endpoints']]
        axes[0,0].plot(p['endpoints'],yy,marker='o',ls=styles[j],color=colors[j],label=labels[j])
        axes[0,1].plot(p['endpoints'],[curve[arm][str(ep)].get('cross_minus_correct_margin_A',np.nan) for ep in p['endpoints']],marker='o',ls=styles[j],color=colors[j])
        yy=[solver[arm][str(steps)]['macro_coarse_RMSD_A'] for steps in [16,64]]
        axes[1,0].bar(np.arange(2)+(j-1.5)*.18,yy,width=.18,color=colors[j])
        axes[1,1].plot([.05,.5,.9],[probes[arm]['1024'][str(t)] for t in [.05,.5,.9]],marker='o',ls=styles[j],color=colors[j])
    axes[0,0].axhline(bound,color='#999999',lw=1,ls=':',label=f'No-AA lower bound ({bound:.2f} Å)')
    axes[0,0].set(xlabel='Optimizer updates',ylabel='Mean coarse reference RMSD (Å)');axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(0,color='#aaaaaa',lw=1);axes[0,1].set(xlabel='Optimizer updates',ylabel='Wrong minus correct target error (Å)')
    axes[1,0].set_xticks([0,1],['16 FM steps','64 FM steps']);axes[1,0].set_ylabel('Final coarse RMSD · same 8 noises (Å)')
    axes[1,1].set(xlabel='Native corruption time t',ylabel='Final native velocity MSE (nm²)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Two same-length training proteins · one fixed frame each · one seed\n1024 updates · frozen PAR backbone · AA availability × time distribution · memorization control, not ensemble evidence',fontsize=11)
    folder=Path('paper/figures/par_two_protein_fit_v1');folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(folder/('PAR_Two_Protein_Fit.'+ext),dpi=170)
    plt.close(fig)
    report=dict(status='complete_awaiting_review',primary_curve=curve,matched_8_sample_solver_diagnostic=solver,
        native_probe_MSE=probes,rows=rows,runs=completed,native_coarse_separation_A=p['native_coarse_separation_A'],
        length_only_macro_RMSD_lower_bound_A=bound,failed_samples=sum(r['metrics']['failure_count'] for r in rows),
        requested_predictions_retained=sum(len(v) for v in arrays.values()),
        audits=dict(all1024_training_events_paired=True,initial_predictions_exact=True,
            position_only_same_noise_predictions_identical=True,native_probe_corruptions_identical=True,
            time_distribution_formula_passed=True),
        figure=str(folder/'PAR_Two_Protein_Fit.pdf'),protocol_sha256=sha(ROOT/'protocol.json'),
        scope=p['scope'],model_default_replaced=False,independent_test_claim=False,
        new_development_predictions=0,test_coordinates_opened=False,R3_opened=False)
    atomic_json(ROOT/'report.json',report)
    jobs=ROOT/'jobs.json'
    if jobs.is_file():
        state=read(jobs);state.update(status=report['status'],all_models_running=False,report_status='complete',figure=report['figure'])
        atomic_json(jobs,state)
        f=Path('artifacts/run_state.json');overall=read(f);overall['par_two_protein_fit_v1']=state;atomic_json(f,overall)
    print('TWO_PROTEIN_FIT_REPORT',json.dumps(curve),flush=True)


if __name__=='__main__':main()
