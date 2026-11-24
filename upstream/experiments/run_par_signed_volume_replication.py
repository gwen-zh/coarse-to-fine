"""Two fixed training-noise replications and an old-checkpoint fresh-noise anchor."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint, sha
from par_coarse_backbone import sample_coarse_batch
from par_coarse_capacity import summary, learning_rate
from par_coarse_transfer import frozen_versions, assert_frozen
from par_signed_volume import ARMS, UPDATES, DRAW_OFFSET, update
from par_signed_volume_replication import SEEDS, evaluation_noise_keys
from par_two_protein_fit import identity
from run_par_coarse_capacity import warm_model
from run_par_coarse_transfer import FeatureBank, verify
from run_par_two_protein_fit import cases_and_targets


ROOT = Path('artifacts/par_signed_volume_replication_v1')
PARENT = Path('artifacts/par_signed_volume_v1')


def read(path):
    return json.loads(Path(path).read_text())


def state_update(**fields):
    f = ROOT / 'jobs.json'
    state = read(f) if f.is_file() else {}
    state.update(fields)
    atomic_json(f, state)
    f = Path('artifacts/run_state.json')
    overall = read(f)
    overall['par_signed_volume_replication_v1'] = state
    atomic_json(f, overall)


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Preflight precedes the frozen replication')
    parent = read(PARENT / 'protocol.json')
    if read(PARENT / 'review.json')['status'] != 'complete_reviewed_positive_training_pair_effect_requires_replication':
        raise RuntimeError('Reviewed parent study required')
    cases, targets = cases_and_targets()
    bank = FeatureBank(parent['training_feature_index'])
    outputs, records = [], []
    for arm in ARMS:
        model, view, optimizer, rng, moment_hash, _ = warm_model('adapter_flow_lora')
        guard = frozen_versions(model)
        before = [sample_coarse_batch(model, bank(t['sequence']), 90, evaluation_noise_keys()[:4], 16).numpy()
                  for t in targets]
        outputs.append(before)
        clock = time.perf_counter()
        row = update(model, optimizer, cases, bank, arm, 2027, 0, parent['signed_volume_weight'])
        seconds = time.perf_counter() - clock
        if row['draw_step'] != 5120 or row['gradient_norm'] <= 0:
            raise RuntimeError('Invalid replication update')
        assert_frozen(model, guard)
        records.append(dict(arm=arm, training_draw_seed=2027, parent_Adam_state_sha256=moment_hash,
                            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                            discarded_update_seconds=seconds, finite_gradient_norm=row['gradient_norm']))
        del model, view, optimizer, rng
    for a, b in zip(*outputs):
        np.testing.assert_array_equal(a, b)
    if len({r['parent_Adam_state_sha256'] for r in records}) != 1:
        raise RuntimeError('Different parent optimizer states')
    atomic_json(ROOT / 'preflight.json', dict(status='passed', runs=records,
                initialization_predictions_exact=True, fixed_weight=parent['signed_volume_weight'],
                discarded_updates=2, actual_replication_updates=0))
    state_update(status='real_weight_preflight_passed', real_weight_preflight_passed=True)


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Fresh replication namespace required')
    pre = read(ROOT / 'preflight.json')
    if pre['status'] != 'passed':
        raise RuntimeError('Real-weight preflight required')
    previous = read(PARENT / 'protocol.json')
    p = dict(seed_family=SEEDS, arms=ARMS, updates=1024, draw_offset=5120,
             signed_volume_weight=previous['signed_volume_weight'],
             targets=previous['targets'], training_feature_index=previous['training_feature_index'],
             evaluation_noise_keys=evaluation_noise_keys(), primary_steps=16, primary_samples=64,
             endpoints=[0, 1024], checkpoint_every=128, progress_every=32,
             initialization='Every new run reloads the SAME parent4096 AA adapter and Adam moments. Native last-two-block rank8 flow LoRA has the SAME fixed initialization in every run and both arms. Only training Gaussian/rotation/time/self-conditioning draws vary between seeds2027/2028;the label order and parameter initialization are held fixed. This is training-draw-seed replication,not independent pretrained-model initialization.',
             objective='Use the unchanged signed-volume updater and feature definition from the reviewed exploratory study. Same parameter counts,LR,gradient clipping and1024update budget for both objectives. Reuse lambda3.3399699750634673 exactly;no calibration or weight scan in the replication.',
             data='The SAME two fixed90-residue TRAIN structures. No extra proteins or MD frames. This tests repeatability on the training pair,not new-protein generalization or ensemble quality.',
             evaluation='64 newly fixed,unique Gaussian noise keys,disjoint from the original64. These keys are SHARED across both proteins,all training seeds,arms and endpoints. AA-only native16-step sampling,in batches32. Save every requested output;no rejection,reflection correction,reference-based selection or checkpoint ranking.',
             anchor='Re-evaluate both existing final2026 objective checkpoints on the SAME new64 noises. This tests evaluation-noise sensitivity of the selected exploratory checkpoints and is not an independent training replication. Do not use the anchor to decide whether to run2027/2028;all four new training runs are fixed.',
             primary='For each new training-draw seed,compare signed-volume minus velocity-only proper macro RMSD across the2proteins,each averaged over64samples. Report both per-protein effects and all seed results. Predeclared directional consistency means BOTH new seed macro effects are negative;it is not a statistical significance criterion. Do not pool individual conformations as independent proteins.',
             near_fit='Report whether each final arm has BOTH per-protein mean proper RMSDs<=2A. Keep all continuous values when the target fails.',
             stop_rule='Exactly2new seeds x2objectives x1024updates. Score only0/final endpoints and all fixed2026 anchors. No best-seed choice,adaptive extension,new weight calibration,extra outcome-triggered seeds or default promotion.',
             scope='Exploratory fixed replication on two repeatedly used training structures,from one common warm start. No independent test,R3,new-sequence,thermodynamic-fidelity or superiority-to-Hierarchical-ConfGen claim.',
             failure_policy='Retain all1280requested baseline/final/anchor samples. Any nonfinite metric group is unavailable;do not filter,resample or replace failing runs.')
    sources = dict(previous['source_sha256'])
    paths = [Path(__file__), Path('upstream/experiments/par_signed_volume_replication.py'),
             Path('upstream/tests/test_par_signed_volume_replication.py'),
             Path('upstream/scripts/par_signed_volume_replication.sbatch'), ROOT / 'preflight.json',
             PARENT / 'protocol.json', PARENT / 'report.json', PARENT / 'review.json']
    paths.extend(PARENT / arm / 'last.pt' for arm in ARMS)
    sources.update({str(path): sha(path) for path in paths})
    p['source_sha256'] = sources
    atomic_json(ROOT / 'protocol.json', p)
    state_update(status='protocol_frozen', frozen_inputs=len(sources), signed_volume_weight=p['signed_volume_weight'])


@torch.no_grad()
def sample(p, model, bank, folder):
    for target in p['targets']:
        clock = time.perf_counter()
        values = np.concatenate([sample_coarse_batch(model, bank(target['sequence']), 90,
                                 p['evaluation_noise_keys'][i:i+32], 16).numpy() for i in [0, 32]])
        out = folder / target['protein']
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'coarse_nfe16.npz', coarse=values,
                            valid=np.isfinite(values).all(axis=(1, 2)),
                            seeds=np.asarray(p['evaluation_noise_keys'], dtype=np.uint32))
        atomic_json(out / 'sampling.json', dict(seconds=time.perf_counter()-clock,
                    samples=64, reference_used_for_generation=False))
    model.nn.init_repr_factory.features = None


def anchor(p):
    bank = FeatureBank(p['training_feature_index'])
    for arm in ARMS:
        state_update(status='anchor_sampling', anchor_arm=arm)
        model, view, optimizer, rng, _, _ = warm_model('adapter_flow_lora')
        saved = restore_checkpoint(PARENT / arm / 'last.pt', view, optimizer, rng, 'cpu')
        if saved['protocol_sha256'] != sha(PARENT / 'protocol.json') or saved['updates'] != 1024 or saved['arm'] != arm:
            raise RuntimeError('Incorrect exploratory anchor checkpoint')
        guard = frozen_versions(model)
        sample(p, model, bank, ROOT / 'anchor2026' / arm)
        assert_frozen(model, guard)
        del model, view, optimizer, rng
    state_update(status='replication_arms_starting')


def train(p, seed, arm, resume):
    cases, _ = cases_and_targets()
    bank = FeatureBank(p['training_feature_index'])
    model, view, optimizer, rng, moment_hash, _ = warm_model('adapter_flow_lora')
    guard = frozen_versions(model)
    out = ROOT / arm / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise RuntimeError('Run already complete')
    start, trace, elapsed = 0, [], 0.
    if resume:
        saved = restore_checkpoint(out / 'last.pt', view, optimizer, rng, 'cpu')
        if saved['protocol_sha256'] != sha(ROOT / 'protocol.json') or saved['seed'] != seed or saved['arm'] != arm:
            raise RuntimeError('Wrong recovery checkpoint')
        start, elapsed = saved['updates'], saved['training_seconds']
        restore_trace_prefix(out, start)
        trace = [json.loads(line) for line in (out / 'trace.jsonl').read_text().splitlines()]
    else:
        if (out / 'last.pt').exists() or (out / 'trace.jsonl').exists():
            raise RuntimeError('Explicit recovery required')
        atomic_json(out / 'baseline.json', dict(parent_Adam_state_sha256=moment_hash,
                    trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
        atomic_json(out / 'progress.json', dict(phase='sampling', updates=0))
        sample(p, model, bank, out / 'step0')
    writer = CompactTrainingTrace(UPDATES, start)
    clock = time.perf_counter()
    for step in range(start, UPDATES):
        row = update(model, optimizer, cases, bank, arm, seed, step, p['signed_volume_weight'])
        trace.append(row)
        if (step+1) % p['progress_every'] == 0:
            writer(out / 'trace.json', trace)
            atomic_json(out / 'progress.json', dict(phase='training', updates=step+1,total=1024,
                        velocity_loss=row['velocity_loss'],volume_loss=row['volume_loss'],lr=row['lr']))
        if (step+1) % p['checkpoint_every'] == 0:
            assert_frozen(model, guard)
            save_checkpoint(out / 'last.pt', view, optimizer, rng, updates=step+1, seed=seed,arm=arm,
                            protocol_sha256=sha(ROOT / 'protocol.json'),training_seconds=elapsed+time.perf_counter()-clock)
    elapsed += time.perf_counter()-clock
    atomic_json(out / 'progress.json', dict(phase='sampling', updates=1024))
    sample(p, model, bank, out / 'step1024')
    assert_frozen(model, guard)
    for index, group in enumerate(optimizer.param_groups):
        expected = 5120 if index == 0 else 1024
        if any(float(optimizer.state[v]['step']) != expected for v in group['params']):
            raise RuntimeError('Incorrect optimizer counters')
    atomic_json(out / 'complete.json', dict(status='complete',seed=seed,arm=arm,updates=1024,
                official_base_parameters_unchanged=True,training_seconds=elapsed,host=platform.node()))
    atomic_json(out / 'progress.json', dict(phase='complete',updates=1024))
    print('VOLUME_REPLICATION_COMPLETE',seed,arm,flush=True)


def report(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases, targets = cases_and_targets()
    rows, traces, initial, fingerprints = [], {}, [], []
    for seed in SEEDS:
        for arm in ARMS:
            folder = ROOT / arm / f'seed{seed}'
            done = read(folder / 'complete.json')
            if done['updates'] != 1024 or not done['official_base_parameters_unchanged']:
                raise RuntimeError('Incomplete replication run')
            trace = read(folder / 'trace.json')
            if [r['draw_step'] for r in trace] != list(range(5120,6144)) or [r['update'] for r in trace] != list(range(1,1025)):
                raise RuntimeError('Changed draw sequence')
            if any(r['lr'] != learning_rate(i) for i,r in enumerate(trace)):
                raise RuntimeError('Changed LR')
            weight=p['signed_volume_weight'] if arm=='velocity_signed_volume' else 0.
            if any(r['signed_volume_weight']!=weight for r in trace):
                raise RuntimeError('Changed objective weight')
            if any(not np.isfinite([r['loss'],r['velocity_loss'],r['volume_loss'],r['gradient_norm']]).all() for r in trace):
                raise RuntimeError('Invalid training objective record')
            traces[seed,arm] = trace
            fingerprints.append(read(folder / 'baseline.json')['parent_Adam_state_sha256'])
    if len(set(fingerprints)) != 1:
        raise RuntimeError('Different parent Adam initialization')
    if any([identity(r) for r in traces[seed,ARMS[0]]] != [identity(r) for r in traces[seed,ARMS[1]]] for seed in SEEDS):
        raise RuntimeError('Unpaired objective draws')
    if [identity(r) for r in traces[SEEDS[0],ARMS[0]]] == [identity(r) for r in traces[SEEDS[1],ARMS[0]]]:
        raise RuntimeError('Training draw seeds did not change the realized draws')
    for seed in [2026]+SEEDS:
        for arm in ARMS:
            for endpoint in ([1024] if seed==2026 else [0,1024]):
                base = ROOT / 'anchor2026' / arm if seed==2026 else ROOT / arm / f'seed{seed}' / f'step{endpoint}'
                for case,target in zip(cases,targets):
                    with np.load(base / target['protein'] / 'coarse_nfe16.npz') as z:
                        values=z['coarse'].copy()
                        np.testing.assert_array_equal(z['seeds'],np.asarray(p['evaluation_noise_keys'],dtype=np.uint32))
                    if values.shape!=(64,23,3):raise RuntimeError('Requested samples missing')
                    if endpoint==0:initial.append((target['protein'],values))
                    bad=int((~np.isfinite(values).all(axis=(1,2))).sum())
                    metrics=dict(metrics_available=False,failed_samples=bad) if bad else dict(metrics_available=True,failed_samples=0,**summary(values,case['coarse'].numpy(),90))
                    rows.append(dict(seed=seed,arm=arm,endpoint=endpoint,protein=target['protein'],metrics=metrics))
    for target in targets:
        values=[v for name,v in initial if name==target['protein']]
        for other in values[1:]:np.testing.assert_array_equal(values[0],other)
    effects={}
    for seed in [2026]+SEEDS:
        scores={}
        for arm in ARMS:
            selected=[r['metrics'] for r in rows if r['seed']==seed and r['arm']==arm and r['endpoint']==1024]
            ok=all(v['metrics_available'] for v in selected)
            scores[arm]=dict(metrics_available=ok,per_protein=selected,
                macro_proper_RMSD_A=float(np.mean([v['mean_proper_RMSD_A'] for v in selected])) if ok else None,
                both_targets_within_2A=bool(max(v['mean_proper_RMSD_A'] for v in selected)<=2) if ok else None)
        ok=all(v['metrics_available'] for v in scores.values())
        effects[str(seed)]=dict(scores=scores,candidate_minus_control_A=scores[ARMS[1]]['macro_proper_RMSD_A']-scores[ARMS[0]]['macro_proper_RMSD_A'] if ok else None,
                               exploratory_anchor=seed==2026)
    dest=Path('paper/figures/par_signed_volume_replication_v1');dest.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for index,arm in enumerate(ARMS):
        xx=np.arange(3)+(index-.5)*.3
        axes[0].bar(xx,[effects[str(s)]['scores'][arm]['macro_proper_RMSD_A']
                       if effects[str(s)]['scores'][arm]['metrics_available'] else np.nan
                       for s in [2026]+SEEDS],width=.3,label=arm)
        axes[1].bar(xx,[sum(r.get('negative_orthogonal_fit_count',0) for r in effects[str(s)]['scores'][arm]['per_protein'])/128
                       if effects[str(s)]['scores'][arm]['metrics_available'] else np.nan for s in [2026]+SEEDS],width=.3)
    for ax in axes:ax.set_xticks([0,1,2],['2026 anchor','2027','2028']);ax.grid(alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean proper coarse RMSD (Å)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Negative orthogonal-fit fraction')
    fig.suptitle('Fresh common evaluation noise · same two TRAIN structures\n2027/2028 are training-draw seeds from one warm start;2026 is an exploratory anchor',fontsize=11)
    for ext in ['pdf','png','svg']:fig.savefig(dest/f'PAR_Signed_Volume_Replication.{ext}',dpi=170)
    plt.close(fig)
    result=dict(status='complete_awaiting_review',effects=effects,rows=rows,
                both_replication_seed_macro_effects_negative=all(effects[str(s)]['candidate_minus_control_A'] is not None and effects[str(s)]['candidate_minus_control_A']<0 for s in SEEDS),
                retained_predictions=sum(r['metrics'].get('samples',64) for r in rows),failed_samples=sum(r['metrics']['failed_samples'] for r in rows),
                audits=dict(paired_training_draws=True,identical_baseline_outputs=True,fixed_lambda=True,all_fixed_seeds_retained=True),
                scope=p['scope'],figure=str(dest/'PAR_Signed_Volume_Replication.pdf'),protocol_sha256=sha(ROOT/'protocol.json'),model_default_replaced=False,
                new_development_predictions=0,test_coordinates_opened=False,R3_opened=False)
    atomic_json(ROOT/'report.json',result);state_update(status=result['status'],all_models_running=False,report_status='complete',figure=result['figure'])
    print('SIGNED_VOLUME_REPLICATION_REPORT',effects,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','prepare','anchor','train','report'],required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='preflight':preflight()
    elif args.stage=='prepare':prepare()
    else:
        p=read(ROOT/'protocol.json');verify(p)
        if args.stage=='anchor':anchor(p)
        elif args.stage=='report':report(p)
        else:
            if args.seed is None or args.arm is None:raise ValueError('Seed and arm required')
            train(p,args.seed,args.arm,args.resume)


if __name__=='__main__':main()
