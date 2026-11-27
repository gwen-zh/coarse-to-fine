"""Frozen-model SC schedule control on the completed chiral velocity models."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_coarse_capacity import geometry_diagnostic, summary
from par_coarse_transfer import assert_frozen, frozen_versions
from par_self_condition_probe import POLICIES, trace_coarse, concatenate_batches
from run_par_chiral_velocity import build
from run_par_coarse_transfer import FeatureBank, verify
from run_par_two_protein_fit import cases_and_targets


ROOT=Path('artifacts/par_self_condition_probe_v1')
PARENT=Path('artifacts/par_chiral_velocity_v1')
ARMS=['native','polar','axial']
SEEDS=[2029,2030]


def read(path):return json.loads(Path(path).read_text())


def state_update(**fields):
    path=ROOT/'jobs.json';state=read(path)if path.exists()else{}
    state.update(fields);atomic_json(path,state)
    path=Path('artifacts/run_state.json');overall=read(path)
    overall['par_self_condition_probe_v1']=state;atomic_json(path,overall)


def load_model(arm,seed):
    model,view,optimizer,rng,_,_=build(arm)
    saved=torch.load(PARENT/arm/f'seed{seed}'/'last.pt',map_location='cpu',weights_only=False)
    if saved['arm']!=arm or saved['seed']!=seed or saved['updates']!=1024 or saved['protocol_sha256']!=sha(PARENT/'protocol.json'):
        raise RuntimeError('Wrong reviewed checkpoint')
    view.load_state_dict(saved['model'],strict=True)
    del saved,optimizer,rng,view
    model.eval().requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):raise RuntimeError('All weights must be frozen')
    return model


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise RuntimeError('Preflight precedes protocol freeze')
    units=read(ROOT/'unit_tests.json')
    if units['status']!='passed' or units['tests']!=4:raise RuntimeError('Targeted tests required')
    verify(units)
    if read(PARENT/'review.json')['status']!='complete_reviewed_training_pair_directional_gain_2A_target_not_met_no_promotion':
        raise RuntimeError('Reviewed parent required')
    old=read(PARENT/'protocol.json');target=old['targets'][0]
    bank=FeatureBank(old['training_feature_index'])
    print('SC_PREFLIGHT_LOADING_NATIVE_2029',flush=True)
    model=load_model('native',2029);guard=frozen_versions(model);records=[]
    expected=np.load(PARENT/'native/seed2029/step1024'/target['protein']/'coarse_nfe16.npz')['coarse'][:32]
    maximum=None
    for policy in POLICIES:
        clock=time.perf_counter()
        value=trace_coarse(model,bank(target['sequence']),90,old['evaluation_noise_keys'][:32],policy,16)
        if not all(np.isfinite(value[key]).all()for key in ['coarse','states_nm','clean_predictions_nm']):
            raise RuntimeError('Nonfinite actual-model preflight')
        if policy=='always':
            np.testing.assert_allclose(value['coarse'],expected,atol=1e-3,rtol=1e-5)
            maximum=float(np.abs(value['coarse']-expected).max())
        records.append(dict(policy=policy,seconds=time.perf_counter()-clock,
                            flow_calls=value['flow_calls'],self_condition_steps=value['self_condition_used'].tolist()))
    assert_frozen(model,guard)
    atomic_json(ROOT/'preflight.json',dict(status='passed',native_replay_max_abs_A=maximum,
                native_replay_atol_A=1e-3,native_replay_rtol=1e-5,runs=records,
                all_parameters_and_buffers_unchanged=True,optimizer_updates=0,
                references_used_for_sampling=False,host=platform.node(),torch_version=str(torch.__version__)))
    state_update(status='real_weight_preflight_passed',real_weight_preflight_passed=True,optimizer_updates=0)
    print('SC_PREFLIGHT_PASSED',maximum,flush=True)


def prepare():
    if (ROOT/'protocol.json').exists():raise RuntimeError('Fresh namespace required')
    pre=read(ROOT/'preflight.json')
    if pre['status']!='passed':raise RuntimeError('Real-model preflight required')
    verify(read(ROOT/'unit_tests.json'))
    old=read(PARENT/'protocol.json')
    p=dict(seeds=SEEDS,arms=ARMS,policies=POLICIES,steps=16,samples=64,batch_size=32,
           targets=old['targets'],training_feature_index=old['training_feature_index'],
           evaluation_noise_keys=old['evaluation_noise_keys'],optimizer_updates=0,
           generation='6fixed reviewed final fields x2TRAINproteins x3SCpolicies x64paired noises =2304newly traced outputs. SAME64Gaussian keys as the parent for exact paired policy comparisons,not independent sampling confirmation.16Eulersteps and2batches32 per protein/policy. Every policy uses the same AA features,random initial state,centering,native nm and final weighted-centered Angstrom interface. No reference input,selection,reflection correction,masking,rejection or resampling.',
           policies_definition='always:previous raw clean prediction at every step after the first,matching the released pilot sampler;never:omit x_sc at every step;delayed_half:omit x_sc for steps0..7,then feed the previous clean prediction at steps8..15. Every policy COMPUTES the clean prediction every step and makes EXACTLY16flow calls per sample and one AR call per batch. The delay boundary.5 is fixed before scoring,no time scan.',
           hypothesis='A previous clean prediction may reinforce an early erroneous orientation branch. Fixed SC interventions test sensitivity to that recurrence without changing weights,capacity,loss or NFE. Existing old-parent trajectories motivate the test but do not prove a self-conditioning mechanism. Policies can change shape quality as well as parity;proper RMSD remains primary.',
           primary='Prespecified delayed_half versus always in AXIAL fields:proper weighted block-center macro RMSD across the2TRAINtargets,ALL64outputs. Directional consistency requires delayed_half-minus-always<0 in BOTH fixed training-draw checkpoints2029/2030. This is not significance or generalization. Also report whether BOTH per-protein means<=2A;no checkpoint promotion. No-SC and native/polar fields are secondary controls,not interchangeable primary policy choices.',
           replay='All768always-policy outputs must reproduce the parent saved outputs within1e-3A/1e-5 with same batches32. Abort on replay mismatch. Models/checkpoint views are restored strictly and EVERYmodel parameter is frozen;parameter/buffer version guards apply to all weights,including AA/LoRA/residual.',
           trajectory='Save all17noisy states and16clean predictions in native nm,plus every SC-use flag and final weighted-centered A coordinates,for every output. After ALL6model workers finish,score all fixed times using proper and reflection-permitted alignment to the same training reference. Time-resolved parity counts and paired branch-switch decompositions are descriptive,not a causal proof or inference oracle.',
           references='Known train coordinates enter only the POST-generation report. The sampler takes AA features,noise keys,checkpoint and fixed SC schedule;no native target or reference-quality gate.',
           compute='All3policies have identical AR/flow call counts for each field/protein/noise. Record elapsed time with the same trace capture and CPU threads. No pretraining/capacity match or speedup claim against Hierarchical ConfGen.',
           scope='Fixed-model posthoc mechanism probe on2repeatedly used90-residue TRAINstructures,one fixedMDframeeach. Common parameter warm start;two existing training-draw checkpoints. No new-protein/dev/test/R3,ensemble-coverage,thermodynamic or full-model-superiority claim.',
           stop_rule='One fixed factorial probe,2304outputs,zero optimizer updates. Report every field/seed/policy/protein and all failures. Do not add delay boundaries,noise seeds,model choices or training based on outcomes. No automatic choice of the best policy,default promotion or MD expansion.',
           failure_policy='Retain all attempted samples and trajectories. Any nonfinite sample invalidates its group;do not filter or replace it. Replay,shape,hash or frozen-weight failures stop the worker. Explicit identical-protocol recovery only.')
    sources=dict(old['source_sha256'])
    for name,digest in sources.items():
        if sha(name)!=digest:raise RuntimeError('Inherited source changed: '+name)
    paths=[Path(__file__),Path('upstream/experiments/par_self_condition_probe.py'),
           Path('upstream/tests/test_par_self_condition_probe.py'),Path('upstream/scripts/par_self_condition_probe.sbatch'),
           ROOT/'preflight.json',ROOT/'unit_tests.json',PARENT/'protocol.json',PARENT/'report.json',PARENT/'review.json']
    for seed in SEEDS:
        for arm in ARMS:
            paths.extend([PARENT/arm/f'seed{seed}'/'last.pt',PARENT/arm/f'seed{seed}'/'complete.json'])
            paths.extend(PARENT/arm/f'seed{seed}'/'step1024'/target['protein']/'coarse_nfe16.npz'for target in p['targets'])
    sources.update({str(q):sha(q)for q in paths});p['source_sha256']=sources
    atomic_json(ROOT/'protocol.json',p)
    state_update(status='protocol_frozen',frozen_inputs=len(sources),optimizer_updates=0,
                 seeds=SEEDS,arms=ARMS,policies=POLICIES,expected_outputs=2304,
                 model_default_replaced=False,test_coordinates_opened=False,R3_opened=False)
    print('SC_PROTOCOL_FROZEN',len(sources),flush=True)


def run(p,seed,arm):
    out=ROOT/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Frozen model already sampled')
    if list(out.glob('*/*/trajectories.npz')):raise RuntimeError('Inspect incomplete output before explicit recovery')
    model=load_model(arm,seed);guard=frozen_versions(model);bank=FeatureBank(p['training_feature_index'])
    records=[];done=0
    for target in p['targets']:
        expected=np.load(PARENT/arm/f'seed{seed}'/'step1024'/target['protein']/'coarse_nfe16.npz')['coarse']
        for policy in POLICIES:
            atomic_json(out/'progress.json',dict(phase='sampling',policy=policy,protein=target['protein'],completed_groups=done,total_groups=6))
            clock=time.perf_counter()
            value=concatenate_batches([trace_coarse(model,bank(target['sequence']),90,
                    p['evaluation_noise_keys'][i:i+32],policy,16)for i in [0,32]])
            directory=out/policy/target['protein'];directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'trajectories.npz',**value)
            finite=all(np.isfinite(value[key]).all()for key in ['coarse','states_nm','clean_predictions_nm'])
            if not finite:raise RuntimeError('Nonfinite full trajectory group;retained and stopped')
            maximum=None
            if policy=='always':
                np.testing.assert_allclose(value['coarse'],expected,atol=1e-3,rtol=1e-5)
                maximum=float(np.abs(value['coarse']-expected).max())
            row=dict(policy=policy,protein=target['protein'],samples=64,
                     flow_calls_per_sample=16,batched_flow_calls=value['flow_calls'],AR_calls=value['AR_calls'],
                     elapsed_seconds=time.perf_counter()-clock,parent_replay_max_abs_A=maximum,
                     references_used_for_sampling=False,all_samples_retained=True)
            records.append(row);atomic_json(directory/'sampling.json',row)
            assert_frozen(model,guard);done+=1
            print('SC_SAMPLED',seed,arm,policy,target['protein'],flush=True)
    model.nn.init_repr_factory.features=None;assert_frozen(model,guard)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arm=arm,groups=6,samples=384,
                optimizer_updates=0,all_weights_and_buffers_unchanged=True,records=records,host=platform.node()))
    atomic_json(out/'progress.json',dict(phase='complete',completed_groups=6,total_groups=6))
    print('SC_MODEL_COMPLETE',seed,arm,flush=True)


def report(p):
    for seed in SEEDS:
        for arm in ARMS:
            complete=read(ROOT/arm/f'seed{seed}'/'complete.json')
            if not complete['all_weights_and_buffers_unchanged']or complete['optimizer_updates']!=0 or complete['samples']!=384:
                raise RuntimeError('Wait for all frozen model workers')
    cases,targets=cases_and_targets()
    if targets!=p['targets']:raise RuntimeError('Reference identity changed')
    scores={};trajectories=[];effects={};total=0;fit_cache={}
    for seed in SEEDS:
        scores[str(seed)]={}
        for arm in ARMS:
            scores[str(seed)][arm]={}
            for policy in POLICIES:
                rows=[]
                for case,target in zip(cases,targets):
                    directory=ROOT/arm/f'seed{seed}'/policy/target['protein']
                    with np.load(directory/'trajectories.npz')as z:
                        final=z['coarse'];states=z['states_nm'];clean=z['clean_predictions_nm'];keys=z['seeds']
                        flags=z['self_condition_used']
                    if final.shape!=(64,23,3)or states.shape!=(17,64,23,3)or clean.shape!=(16,64,23,3)or keys.tolist()!=p['evaluation_noise_keys']:
                        raise RuntimeError('Missing fixed trajectory outputs')
                    reference=case['coarse'].numpy();metric=summary(final,reference,90)
                    fit_cache[(seed,arm,policy,target['protein'])]=geometry_diagnostic(final,reference,90)
                    rows.append(dict(protein=target['protein'],**metric));total+=64
                    trajectories.append(dict(seed=seed,arm=arm,policy=policy,protein=target['protein'],
                        times=np.linspace(0,1,17).tolist(),clean_times=(np.arange(16)/16).tolist(),
                        self_condition_used=flags.tolist(),
                        states=[summary(v*10,reference,90)for v in states],
                        clean_predictions=[summary(v*10,reference,90)for v in clean]))
                scores[str(seed)][arm][policy]=dict(per_protein=rows,
                    macro_proper_RMSD_A=float(np.mean([v['mean_proper_RMSD_A']for v in rows])),
                    negative_fit_count=sum(v['negative_orthogonal_fit_count']for v in rows),
                    both_targets_mean_within_2A=all(v['mean_proper_RMSD_A']<=2 for v in rows))
        q=scores[str(seed)]['axial']
        effects[str(seed)]=dict(delayed_half_minus_always_A=q['delayed_half']['macro_proper_RMSD_A']-q['always']['macro_proper_RMSD_A'],
                               never_minus_always_A=q['never']['macro_proper_RMSD_A']-q['always']['macro_proper_RMSD_A'])
    if total!=2304:raise RuntimeError('Fixed requested outputs missing')
    decomposition=[]
    for seed in SEEDS:
        for arm in ARMS:
            for target in targets:
                name=target['protein'];old=fit_cache[(seed,arm,'always',name)]
                for policy in ['never','delayed_half']:
                    new=fit_cache[(seed,arm,policy,name)]
                    flips=(old['orthogonal_fit_determinant']<0)!=(new['orthogonal_fit_determinant']<0)
                    delta=new['proper_RMSD_A']-old['proper_RMSD_A']
                    decomposition.append(dict(seed=seed,arm=arm,protein=name,policy=policy,
                        proper_mean_change_A=float(delta.mean()),
                        negative_to_positive=int(((old['orthogonal_fit_determinant']<0)&(new['orthogonal_fit_determinant']>=0)).sum()),
                        positive_to_negative=int(((old['orthogonal_fit_determinant']>=0)&(new['orthogonal_fit_determinant']<0)).sum()),
                        change_from_parity_switches_A=float(delta[flips].sum()/64),change_from_others_A=float(delta[~flips].sum()/64)))
    passed=all(v['delayed_half_minus_always_A']<0 for v in effects.values())
    atomic_json(ROOT/'trajectory_diagnostic.json',dict(status='complete',records=trajectories,
                paired_final_effect_decomposition=decomposition,descriptive_not_causal=True))
    result=dict(status='complete_awaiting_independent_review',scores=scores,primary_axial_effects=effects,
                directional_consistency_passed=passed,all_requested_outputs=2304,all_outputs_retained_and_finite=True,
                all_weights_unchanged=True,optimizer_updates=0,protocol_sha256=sha(ROOT/'protocol.json'),
                model_default_replaced=False,independent_test_claim=False,next_experiment_submitted=False,scope=p['scope'])
    atomic_json(ROOT/'report.json',result)
    state_update(status=result['status'],all_models_running=False,completed_runs=6,optimizer_updates=0,
                 report_status='complete',directional_consistency_passed=passed,next_experiment_submitted=False)
    print('SC_REPORTED',json.dumps(effects),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['preflight','prepare','run','report'])
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='preflight':preflight()
    elif args.stage=='prepare':prepare()
    else:
        p=read(ROOT/'protocol.json');verify(p)
        if args.stage=='report':report(p)
        else:
            if args.seed is None or args.arm is None:parser.error('--seed and --arm required')
            run(p,args.seed,args.arm)


if __name__=='__main__':main()
