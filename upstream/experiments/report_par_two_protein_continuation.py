"""Report the fixed continuation without selecting an endpoint or promoting it."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_two_protein_fit import sample_keys
from par_two_protein_continuation import ARMS,ADDITIONAL_UPDATES,learning_rate,common_identity
from report_par_two_protein_fit import measure
from run_par_two_protein_continuation import ROOT,PARENT,read
from run_par_two_protein_fit import cases_and_targets
from run_par_coarse_transfer import verify


def main():
    p=read(ROOT/'protocol.json');verify(p);cases,targets=cases_and_targets()
    refs=[c['coarse'].numpy() for c in cases];rows=[];arrays={};traces={};runs=[];baselines={};native={}
    for arm,(parent,policy) in ARMS.items():
        folder=ROOT/arm;done=read(folder/'complete.json');trace=read(folder/'trace.json')
        if done['global_updates']!=4096 or done['additional_updates']!=3072 or not done['official_parameters_unchanged'] or not done['parent_optimizer_preserved']:
            raise RuntimeError('Incomplete fixed continuation')
        if len(trace)!=ADDITIONAL_UPDATES or [r['global_update'] for r in trace]!=list(range(1025,4097)):
            raise RuntimeError('Continuation trace did not advance the parent steps')
        for i,row in enumerate(trace):
            if row['lr']!=learning_rate(i,policy):raise RuntimeError('Realized LR differs from its declared policy')
        traces[arm]=trace;runs.append(done);baselines[arm]=read(folder/'baseline.json');native[arm]=read(folder/'native_probes.json')
        if baselines[arm]['parent_checkpoint_sha256']!=sha(PARENT/parent/'last.pt'):raise RuntimeError('Wrong parent checkpoint')
        for endpoint in p['endpoints']:
            budgets=[p['primary']]+([p['diagnostic']] if endpoint in p['diagnostic']['endpoints'] else [])
            for budget in budgets:
                for i,target in enumerate(targets):
                    file=folder/f"step{endpoint}/{target['protein']}/coarse_nfe{budget['steps']}.npz"
                    if endpoint==1024:
                        source=PARENT/parent/f"step1024/{target['protein']}/coarse_nfe{budget['steps']}.npz"
                        if sha(file)!=sha(source):raise RuntimeError('Baseline output was changed')
                    with np.load(file) as z:
                        values=z['coarse'].copy()
                        np.testing.assert_array_equal(z['seeds'],np.asarray(sample_keys(p['seed'],budget['samples']),dtype=np.uint32))
                    if values.shape!=(budget['samples'],23,3):raise RuntimeError('A requested output is missing')
                    arrays[arm,endpoint,budget['steps'],i]=values
                    rows.append(dict(arm=arm,global_updates=endpoint,steps=budget['steps'],condition_index=i,
                        protein=target['protein'],metrics=measure(values,refs,90)))
    common=[common_identity(r) for r in next(iter(traces.values()))]
    if any([common_identity(r) for r in trace]!=common for trace in traces.values()):
        raise RuntimeError('Underlying data/noise/rotation/time draws differ across continuation arms')
    for prefix in ['beta','uniform']:
        a,b=prefix+'_floor',prefix+'_restart'
        if baselines[a]['optimizer_state_sha256']!=baselines[b]['optimizer_state_sha256']:
            raise RuntimeError('LR variants did not start with the same Adam moments')
        if [[e['time'] for e in r['batch']] for r in traces[a]]!=[[e['time'] for e in r['batch']] for r in traces[b]]:
            raise RuntimeError('LR variants received different actual times')
    curve={arm:{} for arm in ARMS};solver={};probe_means={};reference_probe_identity=None
    for arm in ARMS:
        for endpoint in p['endpoints']:
            rr=[r for r in rows if r['arm']==arm and r['global_updates']==endpoint and r['steps']==16]
            available=all(r['metrics']['metrics_available'] for r in rr);value=dict(metrics_available=available)
            if available:
                errors=[r['metrics']['mean_distance_to_targets_A'][r['condition_index']] for r in rr]
                wrong=[r['metrics']['mean_distance_to_targets_A'][1-r['condition_index']] for r in rr]
                value.update(per_protein_coarse_RMSD_A=errors,macro_coarse_RMSD_A=float(np.mean(errors)),
                    cross_minus_correct_margin_A=float(np.mean(wrong)-np.mean(errors)),both_training_targets_within_2A=bool(max(errors)<=2.))
            curve[arm][str(endpoint)]=value
        solver[arm]={}
        for steps in [16,64]:
            metrics=[measure(arrays[arm,4096,steps,i][:8],refs,90) for i in range(2)]
            ok=all(m['metrics_available'] for m in metrics)
            solver[arm][str(steps)]=dict(samples_per_protein=8,metrics_available=ok,
                macro_coarse_RMSD_A=float(np.mean([m['mean_distance_to_targets_A'][i] for i,m in enumerate(metrics)])) if ok else None)
        probe_means[arm]={}
        for endpoint in p['endpoints']:
            probes=native[arm][str(endpoint)]
            ident=[{k:v for k,v in r.items() if k!='velocity_MSE_nm2'} for r in probes]
            if reference_probe_identity is None:reference_probe_identity=ident
            if ident!=reference_probe_identity:raise RuntimeError('Native corruption probes differ')
            probe_means[arm][str(endpoint)]={str(t):float(np.mean([r['velocity_MSE_nm2'] for r in probes if r['time']==t])) for t in [.05,.5,.9]}
    colors=['#3577a7','#3577a7','#4a9a72','#4a9a72'];styles=['--','-','--','-']
    labels=['Beta · LR floor','Beta · LR restart','Uniform · LR floor','Uniform · LR restart']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for j,arm in enumerate(ARMS):
        for index,ax in enumerate(axes[0]):
            yy=[curve[arm][str(ep)].get('per_protein_coarse_RMSD_A',[np.nan,np.nan])[index] for ep in p['endpoints']]
            ax.plot(p['endpoints'],yy,marker='o',ls=styles[j],color=colors[j],label=labels[j])
        yy=[solver[arm][str(steps)]['macro_coarse_RMSD_A'] for steps in [16,64]]
        axes[1,0].bar(np.arange(2)+(j-1.5)*.18,[v if v is not None else np.nan for v in yy],width=.18,color=colors[j],alpha=.65 if styles[j]=='--' else 1.)
        axes[1,1].plot([.05,.5,.9],[probe_means[arm]['4096'][str(t)] for t in [.05,.5,.9]],marker='o',ls=styles[j],color=colors[j])
    for ax,target in zip(axes[0],targets):
        ax.set(title=target['protein'],xlabel='Total optimizer updates',ylabel='Training coarse reference RMSD (Å)')
        ax.axhline(2.,color='#999999',ls=':',label='Predeclared 2 Å fit target')
    axes[0,0].legend(fontsize=8)
    axes[1,0].set_xticks([0,1],['16 FM steps','64 FM steps']);axes[1,0].set_ylabel('Final macro RMSD · same 8 noises (Å)')
    axes[1,1].set(xlabel='Native corruption time t',ylabel='Final native velocity MSE (nm²)')
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Separate fixed continuation · 1024 → 4096 updates · Adam moments preserved\nTwo training structures · one seed · beta/uniform time × LR floor/restart · no ensemble-quality claim',fontsize=11)
    folder=Path('paper/figures/par_two_protein_continuation_v1');folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(folder/('PAR_Two_Protein_Continuation.'+ext),dpi=170)
    plt.close(fig)
    report=dict(status='complete_awaiting_review',primary_curve=curve,matched_8_sample_solver_diagnostic=solver,
        native_probe_MSE=probe_means,rows=rows,runs=runs,failed_samples=sum(r['metrics']['failure_count'] for r in rows),
        requested_predictions_retained=sum(len(v) for v in arrays.values()),protocol_sha256=sha(ROOT/'protocol.json'),
        audits=dict(all3072_additional_events_paired=True,global_noise_steps_advanced=True,
            LR_policy_formulas_passed=True,baseline_files_byte_exact=True,parent_Adam_moments_preserved=True,
            native_probe_corruptions_identical=True,official_weights_unchanged=True),
        scope=p['scope'],figure=str(folder/'PAR_Two_Protein_Continuation.pdf'),
        model_default_replaced=False,new_development_predictions=0,test_coordinates_opened=False,R3_opened=False)
    atomic_json(ROOT/'report.json',report)
    file=ROOT/'jobs.json'
    if file.is_file():
        state=read(file);state.update(status=report['status'],all_models_running=False,report_status='complete',figure=report['figure'])
        atomic_json(file,state)
        f=Path('artifacts/run_state.json');overall=read(f);overall['par_two_protein_continuation_v1']=state;atomic_json(f,overall)
    print('CONTINUATION_REPORT',json.dumps(curve),flush=True)


if __name__=='__main__':main()
