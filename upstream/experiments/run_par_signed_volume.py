"""Matched native velocity MSE versus clean-coarse signed-volume supervision."""
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
from par_coarse_capacity import evaluation_keys, learning_rate, summary
from par_coarse_transfer import assert_frozen, frozen_versions
from par_signed_volume import ARMS, UPDATES, DRAW_OFFSET, STRIDES, EDGE_FLOOR_NM, calibrate, update
from par_two_protein_fit import identity, sample_keys, update as original_update
from run_par_coarse_capacity import warm_model, sample_endpoint
from run_par_coarse_transfer import FeatureBank, verify
from run_par_two_protein_fit import cases_and_targets, denoising


ROOT = Path('artifacts/par_signed_volume_v1')
AUDIT = Path('artifacts/par_chirality_audit_v1')
CAPACITY = Path('artifacts/par_coarse_capacity_v1')


def read(path):
    return json.loads(Path(path).read_text())


def state_update(**fields):
    f = ROOT / 'jobs.json'
    state = read(f) if f.is_file() else {}
    state.update(fields)
    atomic_json(f, state)
    f = Path('artifacts/run_state.json')
    all_state = read(f)
    all_state['par_signed_volume_v1'] = state
    atomic_json(f, all_state)


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Preflight must precede frozen trial')
    if not read(AUDIT / 'review.json')['status'].startswith('complete_reviewed_'):
        raise RuntimeError('Reviewed diagnostic required')
    cases, targets = cases_and_targets()
    bank = FeatureBank(read(CAPACITY / 'protocol.json')['training_feature_index'])
    state_update(status='training_only_gradient_calibration')
    model, view, optimizer, rng, _, _ = warm_model('adapter_flow_lora')
    guard = frozen_versions(model)
    calibration = calibrate(model, cases, bank, 2026)
    assert_frozen(model, guard)
    if any(float(optimizer.state[v]['step']) != 4096 for v in optimizer.param_groups[0]['params']):
        raise RuntimeError('Calibration performed an optimizer update')
    weight = calibration['weight']
    atomic_json(ROOT / 'gradient_calibration.json', calibration)
    del model, view, optimizer, rng
    # One discarded native updater step checks that the new control preserves
    # the existing velocity objective and exact draw/optimizer semantics.
    model, view, optimizer, rng, _, _ = warm_model('adapter_flow_lora')
    original_update(model, optimizer, cases, bank, 'sequence_uniform', 2026, DRAW_OFFSET,
                    dict(lr=1e-4, min_lr=1e-4, updates=DRAW_OFFSET+UPDATES, gradient_clip=1.))
    expected = {name: value.detach().clone() for name, value in view.state_dict().items()}
    del model, view, optimizer, rng
    records, initial_outputs = [], []
    for arm in ARMS:
        model, view, optimizer, rng, moment_hash, names = warm_model('adapter_flow_lora')
        guard = frozen_versions(model)
        outputs = []
        for target in targets:
            values = sample_coarse_batch(model, bank(target['sequence']), 90, sample_keys(2026, 4), 16).numpy()
            previous = np.load(CAPACITY / 'adapter_only' / 'step0' / target['protein'] / 'coarse_nfe16.npz')['coarse'][:4]
            np.testing.assert_allclose(values, previous, atol=1e-3, rtol=1e-5)
            outputs.append(values)
        initial_outputs.append(outputs)
        clock = time.perf_counter()
        row = update(model, optimizer, cases, bank, arm, 2026, 0, weight)
        elapsed = time.perf_counter() - clock
        assert_frozen(model, guard)
        replay = None
        if arm == 'velocity_only':
            replay = max(float((value-expected[name]).abs().max()) for name, value in view.state_dict().items())
            for name, value in view.state_dict().items():
                torch.testing.assert_close(value, expected[name], atol=1e-7, rtol=1e-5)
        if row['draw_step'] != DRAW_OFFSET or row['gradient_norm'] <= 0:
            raise RuntimeError('Wrong draw offset or missing objective gradient')
        records.append(dict(arm=arm, parent_Adam_state_sha256=moment_hash,
                            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                            active_volume_weight=row['signed_volume_weight'], gradient_norm=row['gradient_norm'],
                            velocity_loss=row['velocity_loss'], volume_loss=row['volume_loss'],
                            discarded_update_seconds=elapsed, native_control_update_max_abs=replay,
                            official_base_parameters_unchanged=True, lora_modules=names))
        del model, view, optimizer, rng
    for first, second in zip(*initial_outputs):
        np.testing.assert_array_equal(first, second)
    if len({r['trainable_parameters'] for r in records}) != 1 or len({r['parent_Adam_state_sha256'] for r in records}) != 1:
        raise RuntimeError('Objective controls have different parameter count or parent Adam moments')
    atomic_json(ROOT / 'preflight.json', dict(status='passed', runs=records,
                signed_volume_weight=weight, zero_step_predictions_bitwise_equal=True,
                native_MSE_control_one_update_replay_passed=True, calibration_updates=0,
                discarded_updates=3, actual_trial_updates=0))
    state_update(status='real_weight_preflight_passed', signed_volume_weight=weight,
                 real_weight_preflight_passed=True)
    print('SIGNED_VOLUME_PREFLIGHT_PASSED', weight, records, flush=True)


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Fresh signed-volume namespace required')
    pre = read(ROOT / 'preflight.json')
    if pre['status'] != 'passed':
        raise RuntimeError('Actual-weight preflight required')
    previous = read(AUDIT / 'protocol.json')
    p = dict(seed=2026, arms=ARMS, updates=UPDATES, parent_adapter_updates=4096,
             final_adapter_updates=5120, draw_offset=DRAW_OFFSET, final_draw_step=DRAW_OFFSET+UPDATES-1,
             endpoints=[0, 256, 1024], progress_every=32, checkpoint_every=128,
             targets=previous['targets'], training_feature_index=previous['training_feature_index'],
             primary=dict(steps=16, samples=64, batch_size=32),
             signed_volume_weight=pre['signed_volume_weight'], strides=list(STRIDES), edge_floor_nm=EDGE_FLOOR_NM,
             trainable_parameters_per_arm=pre['runs'][0]['trainable_parameters'],
             initialization='Both arms restore the SAME uniform-restart4096 adapter and its4096-step Adam moments,then install IDENTICAL fixed-seed rank8 QKV/output LoRA on the last TWO native flow blocks. New LoRA Adam moments start empty in BOTH arms. All pretrained base parameters and buffers remain frozen. Three preflight updates are discarded;calibration makes zero updates.',
             dataset='Same two90-residue training proteins,one previously used MD frame each. Balanced4draws per update,two per protein. This is a fixed-structure learnability experiment,not ensemble training.',
             objective='Control:native velocity MSE. Candidate:the same MSE plus lambda times MSE of normalized signed scalar triple products in predicted CLEAN coarse coordinates x_t+(1-t)v,compared with the correctly paired clean training target x_t+(1-t)v_true. Features use consecutive edges at fixed strides1/2/4 and0.1nm edge-norm floors. They are translation/proper-rotation invariant and change sign under reflection. No STE,VQ,decoder loss,reference-based inference correction or rejection.',
             weight_calibration='Before trial sampling or training,compute gradient norms on8fixed TRAIN batches with draw steps2100000..2100007. lambda=clip(median(||gradMSE||/||gradVolume||),0.01,100). Record both norms,cosines and achieved scales;do not scan lambda by generated quality. The measured lambda is frozen in this protocol.',
             training='1024additional updates,paired data,proper rotations,OT/Gaussian noises,uniform times,self-conditioning coins and1e-4->1e-5 cosine LR. Actual draw indices5120..6143 avoid repeating the preceding capacity-control draws. The restored adapter optimizer has4096steps and reaches5120;new LoRA moments reach1024. Draw index and optimizer-step count are intentionally distinct and reported separately.',
             evaluation='Unchanged native AA-only16-step coarse sampler,ALL64fixed noise draws per protein from the previous capacity experiment.32historical+32fresh keys are retained;all64 are PRIMARY. Report all0/256/1024endpoints,proper per-protein/macro RMSD,negative-determinant fits and thresholded mirror-like counts separately,reflection-permitted shape error,within2A counts,and wall time. No checkpoint selection,filtering,resampling or reference use in generation.',
             operational_fit_target='BOTH final per-protein mean proper weighted coarse RMSDs<=2A overALL64samples. This is a training-pair sanity target,not a significance,ensemble or generalization criterion.',
             stop_rule='Exactly1024additional updates per arm. No adaptive extension,weight scan or default promotion. Report either direction;decreasing volume loss alone is not success.',
             scope='Exploratory one-seed,two-fixed-training-structure matched-objective intervention after posthoc mirror diagnosis. No independent test,R3,new-sequence,ensemble-fidelity or published-model-superiority claim.',
             failure_policy='Retain all768requested coarse outputs and all endpoints. Any nonfinite group is unavailable;do not filter or gate by reference quality. Unexpected replay,shape or freeze errors stop the run.')
    sources = dict(previous['source_sha256'])
    paths = [Path(__file__), Path('upstream/experiments/par_signed_volume.py'),
             Path('upstream/tests/test_par_signed_volume.py'), Path('upstream/scripts/par_signed_volume.sbatch'),
             ROOT / 'preflight.json', ROOT / 'gradient_calibration.json',
             AUDIT / 'protocol.json', AUDIT / 'report.json', AUDIT / 'review.json']
    sources.update({str(path): sha(path) for path in paths})
    p['source_sha256'] = sources
    atomic_json(ROOT / 'protocol.json', p)
    state_update(status='objective_arms_starting', signed_volume_weight=p['signed_volume_weight'],
                 frozen_inputs=len(sources), actual_trainable_parameters_per_arm=p['trainable_parameters_per_arm'])


def train(p, arm, resume):
    cases, _ = cases_and_targets()
    bank = FeatureBank(p['training_feature_index'])
    model, view, optimizer, rng, parent_hash, _ = warm_model('adapter_flow_lora')
    guard = frozen_versions(model)
    out = ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise RuntimeError('Arm already complete')
    start, trace, elapsed = 0, [], 0.
    if resume:
        saved = restore_checkpoint(out / 'last.pt', view, optimizer, rng, 'cpu')
        if saved['protocol_sha256'] != sha(ROOT / 'protocol.json') or saved['arm'] != arm:
            raise RuntimeError('Incorrect recovery checkpoint')
        start, elapsed = saved['updates'], saved['training_seconds']
        restore_trace_prefix(out, start)
        trace = [json.loads(row) for row in (out / 'trace.jsonl').read_text().splitlines()]
        probes = read(out / 'native_probes.json')
    else:
        if (out / 'last.pt').exists() or (out / 'trace.jsonl').exists():
            raise RuntimeError('Explicit recovery required')
        atomic_json(out / 'baseline.json', dict(parent_Adam_state_sha256=parent_hash,
                    trainable_parameters=p['trainable_parameters_per_arm']))
        sample_endpoint(p, model, bank, out, 0)
        probes = {'0': denoising(model, cases, bank, 'sequence_uniform', p['seed'])}
        atomic_json(out / 'native_probes.json', probes)
    writer = CompactTrainingTrace(UPDATES, start)
    clock = time.perf_counter()
    for step in range(start, UPDATES):
        row = update(model, optimizer, cases, bank, arm, p['seed'], step, p['signed_volume_weight'])
        trace.append(row)
        if (step+1) % p['progress_every'] == 0:
            writer(out / 'trace.json', trace)
            atomic_json(out / 'progress.json', dict(phase='training', updates=step+1, total=UPDATES,
                        draw_step=row['draw_step'], loss=row['loss'], velocity_loss=row['velocity_loss'],
                        volume_loss=row['volume_loss'], lr=row['lr']))
        if step+1 in p['endpoints']:
            elapsed += time.perf_counter()-clock
            sample_endpoint(p, model, bank, out, step+1)
            probes[str(step+1)] = denoising(model, cases, bank, 'sequence_uniform', p['seed'])
            atomic_json(out / 'native_probes.json', probes)
            clock = time.perf_counter()
        if (step+1) % p['checkpoint_every'] == 0:
            assert_frozen(model, guard)
            save_checkpoint(out / 'last.pt', view, optimizer, rng, updates=step+1,
                            total_adapter_updates=4096+step+1, draw_step=DRAW_OFFSET+step, arm=arm,
                            protocol_sha256=sha(ROOT / 'protocol.json'),
                            training_seconds=elapsed+time.perf_counter()-clock)
    for index, group in enumerate(optimizer.param_groups):
        expected = 5120 if index == 0 else 1024
        if any(float(optimizer.state[v]['step']) != expected for v in group['params']):
            raise RuntimeError('Optimizer step counters changed')
    assert_frozen(model, guard)
    atomic_json(out / 'complete.json', dict(status='complete', arm=arm, updates=UPDATES,
                total_adapter_updates=5120, last_draw_step=DRAW_OFFSET+UPDATES-1,
                parent_Adam_moments_preserved=True, official_base_parameters_unchanged=True,
                trainable_parameters=p['trainable_parameters_per_arm'],
                training_seconds=elapsed+time.perf_counter()-clock, host=platform.node()))
    atomic_json(out / 'progress.json', dict(phase='complete', updates=UPDATES))
    print('SIGNED_VOLUME_ARM_COMPLETE', arm, flush=True)


def report(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases, targets = cases_and_targets()
    rows, curves, traces, baseline_hashes, arrays = [], {}, {}, [], {}
    failures = 0
    for arm in ARMS:
        done = read(ROOT / arm / 'complete.json')
        if done['updates'] != UPDATES or not done['official_base_parameters_unchanged']:
            raise RuntimeError('Incomplete objective arm')
        trace = read(ROOT / arm / 'trace.json')
        if [r['update'] for r in trace] != list(range(1, 1025)) or [r['draw_step'] for r in trace] != list(range(5120, 6144)):
            raise RuntimeError('Incorrect trace or noise-draw offset')
        if any(r['lr'] != learning_rate(i) for i, r in enumerate(trace)):
            raise RuntimeError('Changed LR policy')
        expected_weight = p['signed_volume_weight'] if arm == 'velocity_signed_volume' else 0.
        if any(r['signed_volume_weight'] != expected_weight for r in trace):
            raise RuntimeError('Changed objective coefficient')
        traces[arm] = trace
        baseline_hashes.append(read(ROOT / arm / 'baseline.json')['parent_Adam_state_sha256'])
        curves[arm] = {}
        for ep in p['endpoints']:
            records = []
            for case, target in zip(cases, targets):
                with np.load(ROOT / arm / f'step{ep}' / target['protein'] / 'coarse_nfe16.npz') as z:
                    values = z['coarse'].copy()
                    np.testing.assert_array_equal(z['seeds'], np.asarray(evaluation_keys(p['seed']), dtype=np.uint32))
                if values.shape != (64, 23, 3):
                    raise RuntimeError('Requested outputs missing')
                arrays[arm, ep, target['protein']] = values
                bad = int((~np.isfinite(values).all(axis=(1, 2))).sum())
                failures += bad
                metrics = dict(metrics_available=False, failed_samples=bad) if bad else dict(metrics_available=True, failed_samples=0, **summary(values, case['coarse'].numpy(), 90))
                records.append(metrics)
                rows.append(dict(arm=arm, additional_updates=ep, protein=target['protein'], metrics=metrics))
            ok = all(r['metrics_available'] for r in records)
            curves[arm][str(ep)] = dict(per_protein=records, metrics_available=ok,
                macro_proper_RMSD_A=float(np.mean([r['mean_proper_RMSD_A'] for r in records])) if ok else None,
                both_targets_within_2A=bool(max(r['mean_proper_RMSD_A'] for r in records) <= 2) if ok else None)
    if len(set(baseline_hashes)) != 1 or [identity(r) for r in traces[ARMS[0]]] != [identity(r) for r in traces[ARMS[1]]]:
        raise RuntimeError('Objective arms were not initialized and trained on paired controls')
    for target in targets:
        np.testing.assert_array_equal(arrays[ARMS[0], 0, target['protein']], arrays[ARMS[1], 0, target['protein']])
    dest = Path('paper/figures/par_signed_volume_v1')
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, label, color in zip(ARMS, ['Velocity MSE', 'Velocity MSE + signed volume'], ['#3577a7', '#b36945']):
        axes[0].plot(p['endpoints'], [curves[arm][str(ep)]['macro_proper_RMSD_A'] for ep in p['endpoints']], marker='o', label=label, color=color)
        axes[1].plot(p['endpoints'], [sum(r.get('negative_orthogonal_fit_count', 0) for r in curves[arm][str(ep)]['per_protein'])/128
                                    if curves[arm][str(ep)]['metrics_available'] else np.nan for ep in p['endpoints']], marker='o', color=color)
    axes[0].set(xlabel='Additional optimizer updates', ylabel='Mean proper coarse RMSD (Å)')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Additional optimizer updates', ylabel='Negative orthogonal-fit fraction (diagnostic)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Two fixed training structures · one seed · same adapter + native-flow LoRA\n64 draws/protein · matched objective intervention · no ensemble-quality claim', fontsize=11)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(dest / f'PAR_Signed_Volume.{ext}', dpi=170)
    plt.close(fig)
    result = dict(status='complete_awaiting_review', primary_curve=curves, rows=rows,
                  signed_volume_weight=p['signed_volume_weight'], failed_samples=failures,
                  retained_predictions=sum(len(v) for v in arrays.values()),
                  audits=dict(all1024_training_events_paired=True, matched_parameters_and_Adam_initialization=True,
                              initial_predictions_exact=True, fixed_loss_weight_and_LR=True,
                              official_base_parameters_unchanged=True), scope=p['scope'],
                  figure=str(dest / 'PAR_Signed_Volume.pdf'), protocol_sha256=sha(ROOT / 'protocol.json'),
                  model_default_replaced=False, new_development_predictions=0,test_coordinates_opened=False,R3_opened=False)
    atomic_json(ROOT / 'report.json', result)
    state_update(status=result['status'], all_models_running=False, report_status='complete', figure=result['figure'])
    print('SIGNED_VOLUME_REPORT_COMPLETE', {a: curves[a]['1024']['macro_proper_RMSD_A'] for a in ARMS}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['preflight', 'prepare', 'train', 'report'])
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'preflight':
        preflight()
    elif args.stage == 'prepare':
        prepare()
    else:
        p = read(ROOT / 'protocol.json')
        verify(p)
        if args.stage == 'report':
            report(p)
        else:
            if args.arm is None:
                raise ValueError('Arm required')
            train(p, args.arm, args.resume)


if __name__ == '__main__':
    main()
