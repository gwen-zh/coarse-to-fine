"""A fixed two-seed, three-arm residual-architecture learnability study."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint, sha
from par_chiral_velocity import (ARMS, SEEDS, UPDATES, DRAW_OFFSET, WIDTH, LENGTH_FLOOR_NM,
                                 RESIDUAL_INIT_SEED, install_residual, evaluation_keys,
                                 paired_identity, update)
from par_coarse_backbone import sample_coarse_batch
from par_coarse_capacity import summary
from par_coarse_transfer import assert_frozen, frozen_versions
from run_par_coarse_capacity import warm_model
from run_par_coarse_transfer import FeatureBank, verify
from run_par_two_protein_fit import cases_and_targets


ROOT = Path('artifacts/par_chiral_velocity_v1')
PARENT = Path('artifacts/par_two_protein_continuation_v1')
CAPACITY = Path('artifacts/par_coarse_capacity_v1')
CLOSED = Path('artifacts/par_signed_volume_replication_v1')


def read(path):
    return json.loads(Path(path).read_text())


def state_update(**fields):
    path = ROOT / 'jobs.json'
    state = read(path) if path.exists() else {}
    state.update(fields)
    atomic_json(path, state)
    path = Path('artifacts/run_state.json')
    overall = read(path) if path.exists() else {}
    overall['par_chiral_velocity_v1'] = state
    atomic_json(path, overall)


def build(arm):
    if arm not in ARMS:
        raise ValueError('Declared architecture required')
    model, view, optimizer, rng, moments, names = warm_model('adapter_flow_lora')
    if arm != 'native':
        residual = install_residual(model, view, arm)
        optimizer.add_param_group(dict(params=list(residual.parameters())))
    if {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in view.parameters()}:
        raise RuntimeError('Checkpoint view and live trainable parameter sets differ')
    return model, view, optimizer, rng, moments, names


def parameter_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'protocol.json').exists():
        raise RuntimeError('Preflight must precede the frozen experiment')
    unit = read(ROOT/'unit_tests.json')
    if unit['status'] != 'passed' or unit['tests'] != 5:
        raise RuntimeError('Pass the five targeted module tests first')
    verify(unit)
    if read(CLOSED/'review.json')['status'] != 'complete_reviewed_directional_seed_replication_failed_no_promotion':
        raise RuntimeError('Reviewed, closed signed-volume replication required')
    cases, targets = cases_and_targets()
    bank = FeatureBank(read(PARENT/'protocol.json')['training_feature_index'])
    reference, rows = None, []
    for arm in ARMS:
        model, view, optimizer, rng, moments, names = build(arm)
        guard = frozen_versions(model)
        initial = [sample_coarse_batch(model, bank(t['sequence']), 90, evaluation_keys()[:4], 16).numpy()
                   for t in targets]
        if reference is None:
            reference = initial
        else:
            for a, b in zip(reference, initial):np.testing.assert_array_equal(a, b)
        residual_hash = parameter_digest(view.velocity_residual) if arm != 'native' else None
        clock = time.perf_counter()
        first = update(model, optimizer, cases, bank, SEEDS[0], 0)
        head_gradient = (float(view.velocity_residual.output.weight.grad.norm()) if arm != 'native' else None)
        second = update(model, optimizer, cases, bank, SEEDS[0], 1)
        hidden_gradient = (float(view.velocity_residual.hidden.weight.grad.norm()) if arm != 'native' else None)
        seconds = time.perf_counter()-clock
        assert_frozen(model, guard)
        if arm != 'native' and not (np.isfinite(head_gradient) and head_gradient > 0 and
                                    np.isfinite(hidden_gradient) and hidden_gradient > 0):
            raise RuntimeError('Residual lacks a real two-update gradient path')
        rows.append(dict(arm=arm, trainable_parameters=sum(p.numel() for p in view.parameters()),
                         residual_parameters=sum(p.numel() for p in view.velocity_residual.parameters()) if arm != 'native' else 0,
                         common_parent_Adam_sha256=moments, residual_initial_sha256=residual_hash,
                         first_update_loss=first['loss'], first_event=paired_identity(first),
                         finite_gradient_norm=first['gradient_norm'], head_first_gradient_norm=head_gradient,
                         hidden_second_gradient_norm=hidden_gradient, two_discarded_updates_seconds=seconds,
                         native_LoRA_modules=names, official_base_unchanged=True))
        del model, view, optimizer, rng
        gc.collect()
        atomic_json(ROOT/'preflight_progress.json', dict(completed_arms=len(rows), total_arms=3, arm=arm))
    if len({r['common_parent_Adam_sha256'] for r in rows}) != 1:
        raise RuntimeError('Parent Adam state is not shared')
    if rows[1]['trainable_parameters'] != rows[2]['trainable_parameters'] or rows[1]['residual_initial_sha256'] != rows[2]['residual_initial_sha256']:
        raise RuntimeError('Residual capacity or initial weights differ')
    if any(r['first_update_loss'] != rows[0]['first_update_loss'] or r['first_event'] != rows[0]['first_event'] for r in rows):
        raise RuntimeError('Initial native objective or training draws differ')
    atomic_json(ROOT/'preflight.json', dict(status='passed', runs=rows, zero_initial_samples_exact=True,
                paired_native_MSE_loss_exact=True, discarded_optimizer_updates=6, actual_trial_updates=0,
                both_residual_gradient_paths_nonzero=True, residual_parameter_counts_equal=True,
                residual_initial_parameters_equal=True, host=platform.node(),
                torch_version=str(torch.__version__),python_version=platform.python_version()))
    state_update(status='real_weight_preflight_passed', real_weight_preflight_passed=True,
                 discarded_preflight_updates=6, actual_trial_updates=0)
    print('CHIRAL_VELOCITY_PREFLIGHT_PASSED', json.dumps(rows), flush=True)


def prepare():
    if (ROOT/'protocol.json').exists():raise RuntimeError('Fresh namespace required')
    pre = read(ROOT/'preflight.json')
    if pre['status'] != 'passed':raise RuntimeError('Real-weight preflight required')
    verify(read(ROOT/'unit_tests.json'))
    previous = read(CAPACITY/'protocol.json')
    _, targets = cases_and_targets()
    p = dict(arms=ARMS, seeds=SEEDS, updates=UPDATES, draw_offset=DRAW_OFFSET,
             parent_updates=4096, endpoints=[0,1024], primary_steps=16, primary_samples=64,
             sampling_batch=32, evaluation_noise_keys=evaluation_keys(),
             checkpoint_every=128, progress_every=32, targets=targets,
             training_feature_index=previous['training_feature_index'],
             trainable_parameter_counts={r['arm']:r['trainable_parameters'] for r in pre['runs']},
             residual_parameter_count=pre['runs'][1]['residual_parameters'],
             architecture=dict(condition_dim=512, width=WIDTH, basis_vectors=6,
                               length_floor_nm=LENGTH_FLOOR_NM, initialization_seed=RESIDUAL_INIT_SEED),
             initialization='Every run reloads the SAME reviewed uniform_restart AA adapter/Adam at4096,then the SAME zero-B native last-two-block rank8 LoRA. Residual arms add the SAME initialized scalar MLP with zero output. All pretrained117107160parameters remain frozen. New LoRA/residual optimizer moments start empty. The6preflight updates are discarded.',
             controlled_factor='Native control:AA adapter+flow LoRA only. Polar and axial arms add an exactly equal-parameter scalar MLP using AA-only z,time,SC-presence,and reflection-even length/dot geometry. Polar bases are normalized left edge,right edge,and centered radial vector for current state and SC. Axial bases are their3pairwise cross products,for state and SC. Each output residual is centered. Basis changes are an architectural inductive-bias intervention,not isolated proof of a unique chirality mechanism;basis magnitudes can differ and are logged.',
             symmetry='Residual module only is proper-rotation equivariant and translation invariant for AA-only invariant z. Under reflection,polar residual follows R and axial follows det(R)R. Whole pretrained model is NOT assumed equivariant. Smooth .1nm norm regularization handles collapsed/straight traces without world-axis fallback. No claim of guaranteed handedness or target-distribution preservation.',
             training='Exactly2fixed training-draw seeds2029/2030 x3arms x1024additional updates. ORIGINAL uniform-time velocity MSE,not signed-volume or geometry loss. Same4balanced train slots,Gaussian/proper-rotation OT,time,SC coins and1e-4->1e-5 local cosine LR;gradient clip1,AdamW weight_decay1e-4. Draw steps6144..7167. Native teacher-free self-conditioning is model-derived and may diverge across arms;random events are paired. Label order and parameter warm start are fixed across seeds.',
             data='SAME repeatedly used2TRAIN proteins90residues,one fixed MD frame each,23physical coarse centers. No development/test/R3 coordinates,full MD expansion,or fine/decoder training. This is a bounded learnability control,not independent generalization or ensemble evidence.',
             sampling='One common128-output zero-init baseline plus6final runs x128outputs =896requested outputs.64new fixed noise keys shared across proteins,arms,seeds;16Eulersteps,in2batches32. Baseline is computed once because all initial functions and keys are identical,with real preflight replay for all3arms. No intermediate quality scoring,reference-guided selection,reflection correction,sample filtering,rejection or resampling.',
             primary='Proper weighted-aligned coarse RMSD averaged across all64outputs per protein,then macro across2proteins. Report every arm/seed and per-protein result. Directional consistency requires axial MINUS polar<0 AND axial MINUS native<0 for BOTH prespecified training seeds. This is a directional sanity criterion,not a statistical significance test. Also report whether each protein mean<=2A,without selecting weights on it.',
             secondary='Reflection-permitted RMSD,unconstrained negative-determinant fits,thresholded mirror-like count,within2A count,median,p90,training loss/gradient clip fraction,extra residual basis/output RMS,parameter counts,and measured training/sampling time. Neither improper alignment nor lower supervised loss replaces primary AA-only generation.',
             stop_rule='Close at6runs x1024updates and final scoring. No extra outcome-triggered seed,weight/basis/LR scan,adaptive extension,checkpoint ranking,full MD expansion or default promotion. A positive outcome requires independent review and a separately specified next experiment.',
             failure_policy='Retain all attempted outputs and explicit failures. Nonfinite groups are unavailable,never filtered. Unexpected shapes,gradient failures,hash changes or invalid checkpoint restoration stop the run. Recovery is explicit and resumes the same frozen protocol only.',
             scope='Exploratory training-pair architecture control motivated by the closed failed signed-volume study. Not a new-protein test,thermal population estimate,fullPARreplication or superiority claim.')
    sources = {name:digest for name,digest in previous['source_sha256'].items()
               if name.endswith('.py') or name.startswith('third_party/') or name.endswith('.ckpt')
               or name.endswith('/train_labels.pt')}
    for name, digest in sources.items():
        if sha(name) != digest:raise RuntimeError('Inherited frozen input changed: '+name)
    paths = [Path(__file__), Path('upstream/experiments/par_chiral_velocity.py'),
             Path('upstream/tests/test_par_chiral_velocity.py'), Path('upstream/scripts/par_chiral_velocity.sbatch'),
             Path('upstream/experiments/run_par_coarse_capacity.py'),
             Path('upstream/experiments/par_signed_volume_replication.py'),
             Path('upstream/experiments/par_signed_volume.py'),
             ROOT/'preflight.json', ROOT/'unit_tests.json', Path('.venv-par-preflight/pyvenv.cfg'),
             PARENT/'protocol.json', PARENT/'review.json',
             PARENT/'uniform_restart/last.pt', CAPACITY/'protocol.json', CAPACITY/'review.json',
             CLOSED/'review.json', CLOSED/'failure_analysis.json', Path(p['training_feature_index'])]
    index = read(p['training_feature_index'])
    paths.extend(Path(index[t['sequence']]['path']) for t in targets)
    sources.update({str(q):sha(q) for q in paths})
    p['source_sha256'] = sources
    atomic_json(ROOT/'protocol.json',p)
    state_update(status='protocol_frozen',frozen_inputs=len(sources),seeds=SEEDS,arms=ARMS,
                 additional_updates_per_run=UPDATES,model_default_replaced=False,
                 new_development_predictions=0,test_coordinates_opened=False,R3_opened=False)
    print('CHIRAL_VELOCITY_PROTOCOL_FROZEN',len(sources),flush=True)


@torch.no_grad()
def sample(p, model, bank, folder):
    for target in p['targets']:
        clock=time.perf_counter()
        values=np.concatenate([sample_coarse_batch(model,bank(target['sequence']),90,
                    p['evaluation_noise_keys'][i:i+32],16).numpy() for i in [0,32]])
        out=folder/target['protein'];out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'coarse_nfe16.npz',coarse=values,
                seeds=np.asarray(p['evaluation_noise_keys'],dtype=np.uint32),
                valid=np.isfinite(values).all(axis=(1,2)))
        atomic_json(out/'sampling.json',dict(seconds=time.perf_counter()-clock,samples=64,steps=16,
                    reference_used_for_generation=False))
        print('CHIRAL_VELOCITY_SAMPLED',str(folder),target['protein'],flush=True)
    model.nn.init_repr_factory.features=None


def baseline(p):
    if (ROOT/'baseline/complete.json').exists():raise RuntimeError('Baseline already complete')
    model,view,optimizer,rng,_,_=build('native')
    guard=frozen_versions(model)
    bank=FeatureBank(p['training_feature_index'])
    state_update(status='baseline_sampling',all_models_running=False)
    sample(p,model,bank,ROOT/'baseline')
    assert_frozen(model,guard)
    atomic_json(ROOT/'baseline/complete.json',dict(status='complete',samples=128,updates=0,
                reused_by_identical_initial_functions=True))
    state_update(status='training_arms_starting',baseline_complete=True,all_models_running=True)


def train(p,seed,arm,resume):
    cases,targets=cases_and_targets()
    if targets!=p['targets']:raise RuntimeError('Training target identity changed')
    if read(ROOT/'baseline/complete.json')['status']!='complete':raise RuntimeError('Common baseline required')
    model,view,optimizer,rng,moments,names=build(arm)
    guard=frozen_versions(model);bank=FeatureBank(p['training_feature_index'])
    out=ROOT/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Run already complete')
    start,trace,elapsed=0,[],0.
    if resume:
        saved=restore_checkpoint(out/'last.pt',view,optimizer,rng,'cpu')
        if saved['protocol_sha256']!=sha(ROOT/'protocol.json') or saved['arm']!=arm or saved['seed']!=seed:
            raise RuntimeError('Wrong recovery checkpoint')
        start,elapsed=saved['updates'],saved['training_seconds']
        restore_trace_prefix(out,start)
        trace=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
    else:
        if (out/'last.pt').exists() or (out/'trace.jsonl').exists():raise RuntimeError('Explicit recovery required')
        atomic_json(out/'baseline.json',dict(common_parent_Adam_sha256=moments,native_LoRA_modules=names,
                    trainable_parameters=sum(v.numel() for v in view.parameters()),
                    common_baseline_files={t['protein']:sha(ROOT/'baseline'/t['protein']/'coarse_nfe16.npz') for t in targets}))
    if sum(v.numel()for v in view.parameters())!=p['trainable_parameter_counts'][arm]:
        raise RuntimeError('Trainable parameter count changed')
    writer=CompactTrainingTrace(UPDATES,start);clock=time.perf_counter()
    for step in range(start,UPDATES):
        row=update(model,optimizer,cases,bank,seed,step);trace.append(row)
        if (step+1)%p['progress_every']==0:
            writer(out/'trace.json',trace)
            atomic_json(out/'progress.json',dict(phase='training',updates=step+1,total=UPDATES,
                        loss=row['loss'],lr=row['lr'],draw_step=row['draw_step'],
                        training_seconds=elapsed+time.perf_counter()-clock))
            print('CHIRAL_VELOCITY_PROGRESS',seed,arm,step+1,flush=True)
        if (step+1)%p['checkpoint_every']==0:
            assert_frozen(model,guard)
            save_checkpoint(out/'last.pt',view,optimizer,rng,updates=step+1,arm=arm,seed=seed,
                            protocol_sha256=sha(ROOT/'protocol.json'),
                            training_seconds=elapsed+time.perf_counter()-clock)
    training_seconds=elapsed+time.perf_counter()-clock
    for i,group in enumerate(optimizer.param_groups):
        expected=4096+UPDATES if i==0 else UPDATES
        if any(float(optimizer.state[v]['step'])!=expected for v in group['params']):
            raise RuntimeError('Incorrect Adam counters')
    atomic_json(out/'progress.json',dict(phase='sampling',updates=UPDATES,total=UPDATES))
    sample(p,model,bank,out/'step1024')
    assert_frozen(model,guard)
    digest=hashlib.sha256(json.dumps([paired_identity(row)for row in trace],sort_keys=True).encode()).hexdigest()
    atomic_json(out/'complete.json',dict(status='complete',arm=arm,seed=seed,updates=UPDATES,
                common_event_sha256=digest,training_seconds=training_seconds,host=platform.node(),
                official_base_parameters_unchanged=True,trainable_parameters=sum(v.numel()for v in view.parameters()),
                parent_Adam_preserved=True,new_development_or_test_coordinates_loaded=False))
    atomic_json(out/'progress.json',dict(phase='complete',updates=UPDATES,total=UPDATES))
    print('CHIRAL_VELOCITY_COMPLETE',seed,arm,flush=True)


def score_group(p,folder,cases):
    rows=[];count=0
    for case,target in zip(cases,p['targets']):
        with np.load(folder/target['protein']/'coarse_nfe16.npz') as z:
            values=z['coarse'];keys=z['seeds']
            if values.shape!=(64,23,3) or keys.tolist()!=p['evaluation_noise_keys']:
                raise RuntimeError('Sampling identities changed')
        count+=len(values)
        if not np.isfinite(values).all():
            rows.append(dict(metrics_available=False,protein=target['protein'],samples=64,
                             failed_samples=int((~np.isfinite(values).all(axis=(1,2))).sum())))
        else:
            rows.append(dict(metrics_available=True,protein=target['protein'],failed_samples=0,
                             **summary(values,case['coarse'].numpy(),90)))
    available=all(row['metrics_available']for row in rows)
    return dict(metrics_available=available,per_protein=rows,samples=count,
                macro_proper_RMSD_A=float(np.mean([r['mean_proper_RMSD_A']for r in rows])) if available else None,
                both_targets_mean_within_2A=available and all(r['mean_proper_RMSD_A']<=2 for r in rows))


def report(p):
    cases,_=cases_and_targets()
    initial=score_group(p,ROOT/'baseline',cases)
    effects={};events={};total=initial['samples'];records=[]
    for seed in SEEDS:
        scores={}
        for arm in ARMS:
            out=ROOT/arm/f'seed{seed}';complete=read(out/'complete.json')
            if complete['updates']!=UPDATES or not complete['official_base_parameters_unchanged']:
                raise RuntimeError('Incomplete or modified base model')
            trace=read(out/'trace.json')
            if len(trace)!=UPDATES or any(r['update']!=i+1 or r['draw_step']!=DRAW_OFFSET+i for i,r in enumerate(trace)):
                raise RuntimeError('Wrong local/draw step counters')
            if not all(np.isfinite(r['loss'])and np.isfinite(r['gradient_norm'])for r in trace):
                raise RuntimeError('Nonfinite training event')
            identity=[paired_identity(r)for r in trace]
            digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            if digest!=complete['common_event_sha256']:raise RuntimeError('Trace digest differs')
            if seed in events and identity!=events[seed]:raise RuntimeError('Unpaired training events')
            events[seed]=identity
            scores[arm]=score_group(p,out/'step1024',cases);total+=scores[arm]['samples']
            last=trace[-128:]
            records.append(dict(seed=seed,arm=arm,training_seconds=complete['training_seconds'],
                                trainable_parameters=complete['trainable_parameters'],
                                velocity_loss_last128=float(np.mean([r['loss']for r in last])),
                                gradient_clip_fraction_last128=float(np.mean([r['gradient_norm']>1 for r in last])),
                                mean_basis_RMS_last128=float(np.mean([r['residual_diagnostic']['basis_RMS']for r in last])) if arm!='native' else None,
                                mean_residual_RMS_nm_last128=float(np.mean([r['residual_diagnostic']['residual_RMS_nm']for r in last])) if arm!='native' else None))
        valid=all(s['metrics_available']for s in scores.values())
        effects[str(seed)]=dict(scores=scores,
                axial_minus_polar_A=scores['axial']['macro_proper_RMSD_A']-scores['polar']['macro_proper_RMSD_A'] if valid else None,
                axial_minus_native_A=scores['axial']['macro_proper_RMSD_A']-scores['native']['macro_proper_RMSD_A'] if valid else None)
    if events[SEEDS[0]]==events[SEEDS[1]]:raise RuntimeError('Training seeds did not change the random events')
    if total!=896:raise RuntimeError('Missing requested outputs')
    passed=all(e['axial_minus_polar_A']is not None and e['axial_minus_polar_A']<0 and
               e['axial_minus_native_A']<0 for e in effects.values())
    result=dict(status='complete_awaiting_independent_review',baseline=initial,effects=effects,
                directional_consistency_passed=passed,training_records=records,requested_outputs=896,
                scored_or_marked_unavailable_outputs=total,all_training_events_paired=True,
                protocol_sha256=sha(ROOT/'protocol.json'),model_default_replaced=False,
                independent_test_claim=False,new_development_predictions=0,R3_opened=False,
                scope=p['scope'],stop_rule=p['stop_rule'])
    atomic_json(ROOT/'report.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,seed in zip(axes,SEEDS):
        scores=effects[str(seed)]['scores']
        values=[scores[arm]['macro_proper_RMSD_A']for arm in ARMS]
        if all(v is not None for v in values):
            ax.bar(ARMS,values,color=['#777777','#4c78a8','#f58518'])
            for i,v in enumerate(values):ax.text(i,v+.05,f'{v:.3f}',ha='center')
        ax.set_title(f'Training-draw seed {seed}');ax.axhline(2,color='black',ls=':',lw=1)
        ax.set_ylabel('Proper coarse RMSD (Å)');ax.set_ylim(bottom=0)
    fig.suptitle('Two reused TRAIN structures; fixed pure-MSE architecture control')
    fig.tight_layout()
    directory=Path('paper/figures/par_chiral_velocity_v1');directory.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png','svg']:fig.savefig(directory/f'PAR_Chiral_Velocity.{ext}',dpi=180)
    plt.close(fig)
    state_update(status='complete_awaiting_independent_review',all_models_running=False,completed_runs=6,
                 report_status='complete',directional_consistency_passed=passed,
                 figure=str(directory/'PAR_Chiral_Velocity.pdf'),next_experiment_submitted=False)
    print('CHIRAL_VELOCITY_REPORTED',json.dumps(effects),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['preflight','prepare','baseline','train','report'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='preflight':preflight()
    elif args.stage=='prepare':prepare()
    else:
        p=read(ROOT/'protocol.json');verify(p)
        if args.stage=='baseline':baseline(p)
        elif args.stage=='report':report(p)
        else:
            if args.arm is None or args.seed is None:parser.error('Training requires --arm and --seed')
            train(p,args.seed,args.arm,args.resume)


if __name__=='__main__':main()
