"""Independent coordinate/trajectory audit after all fixed SC workers finish."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from review_par_chiral_velocity import explicit_alignment
from run_par_two_protein_fit import cases_and_targets


ROOT=Path('artifacts/par_self_condition_probe_v1')
PARENT=Path('artifacts/par_chiral_velocity_v1')
ARMS=['native','polar','axial'];SEEDS=[2029,2030];POLICIES=['always','never','delayed_half']


def read(path):return json.loads(Path(path).read_text())


def main():
    torch.set_num_threads(2)
    if (ROOT/'review_manifest.json').exists():
        manifest=read(ROOT/'review_manifest.json')
        for name,digest in manifest['audit_source_sha256'].items():
            if sha(name)!=digest:raise RuntimeError('Audit source changed: '+name)
    recovery=None
    if (ROOT/'recovery_manifest.json').exists():
        recovery=read(ROOT/'recovery_manifest.json')
        for name,digest in {**recovery['recovery_source_sha256'],**recovery['original_saved_trajectory_files_sha256']}.items():
            if sha(name)!=digest:raise RuntimeError('Recovery or preserved original input changed: '+name)
        if sha(ROOT/'protocol.json')!=recovery['original_protocol_sha256']:
            raise RuntimeError('Recovery protocol is not byte-identical')
        if sha(ROOT/'original_failed_job_log.txt')!=recovery['original_failed_log_sha256']:
            raise RuntimeError('Original failure log changed')
        precheck=read(ROOT/'replay_precheck.json')
        if (precheck['status']!='passed' or not precheck['traced_vs_native_exact'] or
            precheck['original_atol_A']!=1e-3 or precheck['original_rtol']!=1e-5):
            raise RuntimeError('Parent-hardware replay precheck failed')
    p=read(ROOT/'protocol.json');report=read(ROOT/'report.json')
    if report['protocol_sha256']!=sha(ROOT/'protocol.json'):raise RuntimeError('Protocol mismatch')
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    cases,targets=cases_and_targets()
    if targets!=p['targets']:raise RuntimeError('References changed')
    weights=np.asarray([4]*22+[2],dtype=np.float64)/90
    total=0;replayed=0;max_mean_error=0.;max_replay_error=0.;max_euler_error=0.;max_unit_error=0.
    final={};timings=[];output_audit=[]
    for seed in SEEDS:
        final[str(seed)]={}
        for arm in ARMS:
            out=ROOT/arm/f'seed{seed}'
            complete=read(out/'complete.json')
            if complete['optimizer_updates']!=0 or not complete['all_weights_and_buffers_unchanged']or complete['samples']!=384:
                raise RuntimeError('Missing fixed-weight worker')
            by_policy={name:[]for name in POLICIES}
            for case,target in zip(cases,targets):
                caches={}
                for policy in POLICIES:
                    directory=out/policy/target['protein']
                    with np.load(directory/'trajectories.npz')as z:
                        x=z['coarse'].copy();states=z['states_nm'].copy();clean=z['clean_predictions_nm'].copy()
                        flags=z['self_condition_used'];keys=z['seeds']
                        if int(z['flow_calls'])!=32 or int(z['AR_calls'])!=2:raise RuntimeError('Different call budget')
                    if x.shape!=(64,23,3)or states.shape!=(17,64,23,3)or clean.shape!=(16,64,23,3)or keys.tolist()!=p['evaluation_noise_keys']:
                        raise RuntimeError('Missing/changed output or noise keys')
                    if not all(np.isfinite(v).all()for v in [x,states,clean]):raise RuntimeError('Nonfinite retained outputs')
                    expected_flags=([False]+[True]*15 if policy=='always'else [False]*16 if policy=='never'else [False]*8+[True]*8)
                    np.testing.assert_array_equal(flags,expected_flags)
                    for step in range(16):
                        expected=states[step]+(clean[step]-states[step])/(16-step)
                        expected=expected-expected.mean(1,keepdims=True)
                        error=float(np.abs(expected-states[step+1]).max())
                        max_euler_error=max(max_euler_error,error)
                        if error>1e-4:raise RuntimeError('Stored states do not follow declared Euler updates')
                    expected=states[-1].astype(np.float64)*10
                    expected=expected-(expected*weights[None,:,None]).sum(1,keepdims=True)
                    error=float(np.abs(expected-x).max());max_unit_error=max(max_unit_error,error)
                    if error>1e-3:raise RuntimeError('Wrong unit/centering conversion')
                    caches[policy]=states
                    fit=explicit_alignment(x,case['coarse'].numpy())
                    old=next(v for v in report['scores'][str(seed)][arm][policy]['per_protein']if v['protein']==target['protein'])
                    proper=float(fit['proper'].mean());improper=float(fit['reflection_permitted'].mean())
                    for value,key in [(proper,'mean_proper_RMSD_A'),(improper,'mean_reflection_permitted_RMSD_A')]:
                        error=abs(value-old[key]);max_mean_error=max(max_mean_error,error)
                        if error>1e-6:raise RuntimeError('Independent metric differs')
                    negative=int(fit['negative_fit'].sum())
                    if negative!=old['negative_orthogonal_fit_count']:raise RuntimeError('Wrong parity count')
                    row=dict(protein=target['protein'],proper_RMSD_A=proper,reflection_permitted_RMSD_A=improper,
                             negative_fit_count=negative,samples=64)
                    by_policy[policy].append(row);total+=64
                    output_audit.append(dict(seed=seed,arm=arm,policy=policy,protein=target['protein'],
                            proper_RMSD_A=fit['proper'].tolist(),reflection_permitted_RMSD_A=fit['reflection_permitted'].tolist()))
                    timing=read(directory/'sampling.json');timings.append(dict(seed=seed,arm=arm,**timing))
                    if policy=='always':
                        with np.load(PARENT/arm/f'seed{seed}'/'step1024'/target['protein']/'coarse_nfe16.npz')as z:
                            previous=z['coarse']
                        np.testing.assert_allclose(x,previous,atol=1e-3,rtol=1e-5)
                        max_replay_error=max(max_replay_error,float(np.abs(x-previous).max()));replayed+=64
                np.testing.assert_allclose(caches['always'][:2],caches['never'][:2],atol=1e-5,rtol=1e-5)
                np.testing.assert_allclose(caches['never'][:9],caches['delayed_half'][:9],atol=1e-5,rtol=1e-5)
            final[str(seed)][arm]={name:dict(per_protein=rows,macro_proper_RMSD_A=float(np.mean([v['proper_RMSD_A']for v in rows])),
                negative_fit_count=sum(v['negative_fit_count']for v in rows),both_targets_mean_within_2A=all(v['proper_RMSD_A']<=2 for v in rows))for name,rows in by_policy.items()}
    if total!=2304 or replayed!=768:raise RuntimeError('Missing outputs/replay controls')
    effects={}
    for seed in SEEDS:
        values=final[str(seed)]['axial']
        effects[str(seed)]={name+'_minus_always_A':values[name]['macro_proper_RMSD_A']-values['always']['macro_proper_RMSD_A']for name in ['never','delayed_half']}
    passed=all(v['delayed_half_minus_always_A']<0 for v in effects.values())
    if passed!=report['directional_consistency_passed']:raise RuntimeError('Prespecified primary criterion differs')
    status=('complete_reviewed_delayed_SC_directional_gain_no_promotion'if passed else
            'complete_reviewed_delayed_SC_not_consistent_no_promotion')
    audit=dict(frozen_input_hashes_verified=len(p['source_sha256']),independently_aligned_outputs=total,
               normal_policy_parent_outputs_replayed=replayed,max_parent_replay_abs_A=max_replay_error,
               max_report_mean_difference_A=max_mean_error,max_Euler_reconstruction_error_nm=max_euler_error,
               max_final_unit_centering_difference_A=max_unit_error,SC_flags_and_equal_NFE_verified=True,
               shared_initial_noise_and_early_prefix_verified=True,all_weights_unchanged=True,optimizer_updates=0,
               all_requested_outputs_finite_and_retained=True,figure_visually_inspected=False)
    result=dict(status=status,reviewed_at_UTC=datetime.now(timezone.utc).isoformat(),scores=final,
                primary_axial_effects=effects,directional_consistency_passed=passed,audits=audit,
                never_SC_is_secondary_not_registered_policy_choice=True,
                protocol_sha256=sha(ROOT/'protocol.json'),report_sha256=sha(ROOT/'report.json'),
                review_code_sha256=sha(Path(__file__)),alignment_helper_sha256=sha(Path('upstream/experiments/review_par_chiral_velocity.py')),
                model_default_replaced=False,independent_test_claim=False,next_experiment_submitted=False,
                interpretation=['Two reused TRAIN structures,one fixed frame each,existing trained fields and parent sampling noises. Not new-sequence/ensemble/thermal superiority.',
                                'The only prespecified primary policy is delayed_half versus always in axial fields for BOTH checkpoints. Never-SC is secondary even if descriptively favorable.',
                                'All fixed fields,policies and outputs retained. Changing SC can change shape and preferred reflection fit;this is not an isolated causal proof of mirror locking.',
                                'Close the fixed probe. No default promotion,extra delay boundary or outcome-triggered extension. Any next study requires a separate specification.'])
    if recovery is not None:
        result['technical_recovery']=dict(original_job=recovery['original_job'],original_node=recovery['original_node'],
            original_saved_outputs_preserved=recovery['original_saved_outputs'],
            recovery_node=recovery['recovery_node'],protocol_byte_identical=True,tolerance_unchanged=True,
            first_attempt_not_pooled_as_extra_samples=True)
    atomic_json(ROOT/'independent_alignment_audit.json',dict(audits=audit,records=output_audit,sampling_times=timings))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9.6,7))
    ymax=max(final[str(s)]['axial'][policy]['macro_proper_RMSD_A']for s in SEEDS for policy in POLICIES)*1.28
    for col,seed in enumerate(SEEDS):
        values=final[str(seed)]['axial']
        for row,key in [(0,'macro_proper_RMSD_A'),(1,'negative_fit_count')]:
            ax=axes[row,col];ys=[values[name][key]for name in POLICIES]
            ax.bar(range(3),ys,color=['#777777','#4c78a8','#e1812c'])
            ax.set_xticks(range(3),['Always','Never','Delayed half'])
            ax.set_ylim(0,ymax if row==0 else 140)
            for i,v in enumerate(ys):ax.text(i,v+(ymax*.02 if row==0 else 2),f'{v:.3f}'if row==0 else str(v),ha='center',fontsize=10)
            ax.spines[['top','right']].set_visible(False)
            ax.set_ylabel('Proper coarse RMSD (Å)'if row==0 else 'Negative orthogonal fits / 128')
        axes[0,col].set_title(f'Axial field: training-draw seed {seed}')
    fig.suptitle('Frozen self-conditioning probe: '+('primary criterion passed'if passed else 'primary criterion not met'),fontsize=13)
    fig.text(.5,.015,'Two reused TRAIN structures · fixed weights · 16 flow calls · all 2304 outputs retained',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    directory=Path('paper/figures/par_self_condition_probe_v1');directory.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(directory/f'PAR_Self_Condition_Probe.{ext}',dpi=180)
    plt.close(fig)
    result['figure']=str(directory/'PAR_Self_Condition_Probe.pdf');atomic_json(ROOT/'review.json',result)
    path=ROOT/'jobs.json';state=read(path);state.update(status=status,review=str(ROOT/'review.json'),completed_runs=6,
        all_models_running=False,optimizer_updates=0,directional_consistency_passed=passed,next_experiment_submitted=False,
        last_verified_UTC=result['reviewed_at_UTC'],figure=result['figure'])
    atomic_json(path,state)
    path=Path('artifacts/run_state.json');overall=read(path);overall['par_self_condition_probe_v1']=state;atomic_json(path,overall)
    print(json.dumps(dict(status=status,primary_axial_effects=effects,audits=audit),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default=str(ROOT))
    args=parser.parse_args();ROOT=Path(args.root)
    main()
