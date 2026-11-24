"""Replay coarse trajectories, then compare adapter-only versus native-flow LoRA."""
import argparse
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
from par_coarse_backbone import sample_coarse_batch
from par_coarse_capacity import (ARMS, PARENT_UPDATES, UPDATES, evaluation_keys, geometry_diagnostic,
                                install_flow_lora, summary, trajectory_sample, update)
from par_coarse_transfer import assert_frozen, frozen_versions
from par_two_protein_fit import identity, sample_keys
from run_par_coarse_transfer import FeatureBank, model_for, verify
from run_par_two_protein_continuation import optimizer_digest
from run_par_two_protein_fit import cases_and_targets, denoising


ROOT = Path('artifacts/par_coarse_capacity_v1')
PARENT = Path('artifacts/par_two_protein_continuation_v1')


def read(path):
    return json.loads(Path(path).read_text())


def state_update(**fields):
    path = ROOT / 'jobs.json'
    state = read(path) if path.is_file() else {}
    state.update(fields)
    atomic_json(path, state)
    path = Path('artifacts/run_state.json')
    all_state = read(path) if path.is_file() else {}
    all_state['par_coarse_capacity_v1'] = state
    atomic_json(path, all_state)


def warm_model(arm, parent='uniform_restart'):
    p = read(PARENT / 'protocol.json')
    model, factory, view = model_for(p['seed'], 'cpu')
    optimizer = torch.optim.AdamW(view.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(p['seed'])
    saved = restore_checkpoint(PARENT / parent / 'last.pt', view, optimizer, rng, 'cpu')
    if saved['global_updates'] != PARENT_UPDATES or saved['arm'] != parent or saved['protocol_sha256'] != sha(PARENT / 'protocol.json'):
        raise RuntimeError('Incorrect capacity-control parent')
    if any(float(v['step']) != PARENT_UPDATES for v in optimizer.state.values()):
        raise RuntimeError('Parent Adam moments are not at4096')
    moment_hash = optimizer_digest(optimizer.state_dict())
    if moment_hash != optimizer_digest(saved['optimizer']):
        raise RuntimeError('Parent Adam state differs after restoration')
    names = []
    if arm == 'adapter_flow_lora':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261109)
            names = install_flow_lora(model, view, rank=8, last_blocks=2)
        optimizer.add_param_group(dict(params=list(view.flow_lora.parameters())))
    elif arm != 'adapter_only':
        raise ValueError('Declared arm required')
    return model, view, optimizer, rng, moment_hash, names


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Preflight must precede the frozen capacity trial')
    if read(PARENT / 'review.json')['status'] != 'complete_reviewed_training_pair_improved_2A_target_not_met':
        raise RuntimeError('Reviewed parent required')
    cases, targets = cases_and_targets()
    bank = FeatureBank(read(PARENT / 'protocol.json')['training_feature_index'])
    records, initial = [], []
    for arm in ARMS:
        model, view, optimizer, rng, moment_hash, names = warm_model(arm)
        guard = frozen_versions(model)
        outputs, maxima = [], []
        for target in targets:
            values = sample_coarse_batch(model, bank(target['sequence']), 90, sample_keys(2026, 4), 16).numpy()
            traced = trajectory_sample(model, bank(target['sequence']), 90, sample_keys(2026, 4), 16)
            np.testing.assert_array_equal(traced['coarse'], values)
            expected = np.load(PARENT / 'uniform_restart' / 'step4096' / target['protein'] / 'coarse_nfe16.npz')['coarse'][:4]
            np.testing.assert_allclose(values, expected, atol=1e-3, rtol=1e-5)
            outputs.append(values)
            maxima.append(float(np.max(np.abs(values - expected))))
        initial.append(outputs)
        clock = time.perf_counter()
        row = update(model, optimizer, cases, bank, 2026, 0)
        elapsed = time.perf_counter() - clock
        assert_frozen(model, guard)
        if row['global_update'] != 4097 or row['gradient_norm'] <= 0:
            raise RuntimeError('Missing warm-start gradient or wrong draw offset')
        lora_gradient = 0.
        if names:
            lora_gradient = sum(float(v.grad.square().sum()) for name, v in view.flow_lora.named_parameters()
                                if name.endswith('.b') and v.grad is not None) ** .5
            if not np.isfinite(lora_gradient) or lora_gradient <= 0:
                raise RuntimeError('Native flow LoRA receives no finite gradient')
        records.append(dict(arm=arm, parent_optimizer_state_sha256=moment_hash,
                            parent_Adam_moments_preserved=True, replay_max_abs_A=max(maxima),
                            discarded_update_seconds=elapsed, finite_gradient_norm=row['gradient_norm'],
                            lora_B_gradient_norm=lora_gradient,
                            trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                            lora_modules=names, official_base_parameters_unchanged=True))
        del model, view, optimizer, rng
    for first, second in zip(*initial):
        np.testing.assert_array_equal(first, second)
    if records[0]['parent_optimizer_state_sha256'] != records[1]['parent_optimizer_state_sha256']:
        raise RuntimeError('Different initial adapter optimizer states')
    atomic_json(ROOT / 'preflight.json', dict(status='passed', runs=records,
                zero_LoRA_initial_predictions_bitwise_equal=True, discarded_updates=2,
                traced_sampler_matches_original_bitwise=True,
                trial_updates=0, replay_atol_A=1e-3, replay_rtol=1e-5, host=platform.node()))
    state_update(status='real_weight_preflight_passed', real_weight_preflight_passed=True)
    print('CAPACITY_PREFLIGHT_PASSED', json.dumps(records), flush=True)


def prepare():
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Fresh protocol namespace required')
    pre = read(ROOT / 'preflight.json')
    if pre['status'] != 'passed':
        raise RuntimeError('Actual-weight replay and gradient checks required')
    old = read(PARENT / 'protocol.json')
    _, targets = cases_and_targets()
    p = dict(seed=2026, arms=ARMS, parent='uniform_restart', parent_updates=4096,
             updates=UPDATES, global_final_updates=5120, endpoints=[0, 256, 1024],
             primary=dict(steps=16, samples=64, batch_size=32), gradient_clip=1., weight_decay=1e-4,
             checkpoint_every=128, progress_every=32, targets=targets,
             training_feature_index=old['training_feature_index'],
             initialization='Both arms restore the same reviewed uniform-restart4096 AA adapter and AdamW moments. The additional native-flow LoRA is rank8,scale1,with zero B and fixed-seed random A. Only its new Adam states start empty. Two preflight updates are discarded.',
             adaptation='Control:464512 AA-adapter parameters. Candidate:the same adapter plus49152 native-flow LoRA parameters on QKV/output projections in the LAST TWO flow-attention blocks. Original117107160 pretrained parameters and all buffers remain frozen. AR trunk,coordinate output head,own fine model and codec are unchanged. Parameter counts differ;this is an adaptation-location/capacity control,not an equal-parameter causal isolation.',
             dataset='Same two previously used training proteins of length90,one fixed MD frame each,balanced2examples per protein per update. This is a memorization control,not ensemble training.',
             training='Exactly1024 additional AdamW updates;shared uniform time,Gaussian/proper-rotation OT noise,self-conditioning coins,data order,and1e-4->1e-5 cosine LR. Draw indices4096..5119 advance beyond all prior continuation draws. Native velocity MSE only;no new geometric,chiral,reference-selection or ensemble objective.',
             generation='AA-only coarse generation with frozen evaluation noise keys:32 historical keys followed by32 fresh keys. Sample in two fixed32-size batches;all64 samples are the PRIMARY. Historical32 prefix reported separately only for continuity. Same64 keys for both proteins,arms,endpoints. No sample rejection,reflection correction,resampling or best-step selection.',
             trajectory_diagnostic='Before training,replay ALL32 historical16-step trajectories for BOTH completed beta_restart and uniform_restart parents,for both training proteins. Save every noisy state and clean prediction. Their final outputs must replay existing arrays within1e-3A/1e-5. Score afterwards under proper rotations and reflection-permitting O(3) alignment. A mirror-like diagnostic means properRMSD>2A,reflection-permittedRMSD<=2A,and negative orthogonal-fit determinant. The2A descriptive threshold follows the previous near-fit convention;this is a posthoc diagnostic,not a registered mechanism test.',
             primary_metric='Proper weighted-aligned block-center mean RMSD over ALL64 samples,first per protein then macro-average. Operational near-fit requires BOTH per-protein means<=2A at final1024additional updates. Reflection-permitting metrics NEVER replace the primary. All fixed endpoints are reported.',
             secondary='Historical32 and fresh32 mean errors;mirror-like counts;median,p90 and within2A counts;fixed native denoising probes at t=.05/.5/.9;actual training and sampling time. No independent significance or thermodynamic claim.',
             scope='Exploratory one-parent,one-training-draw-seed,two-fixed-training-structure control. Parent choice followed the previous training-pair result. No new development,test or R3 coordinates;no overall published-model superiority claim or automatic promotion.',
             stop_rule='Stop all arms at exactly1024additional updates and report all endpoints. No adaptive extension,checkpoint ranking or automatic transition to full MD training. If fitting remains poor,record the negative result and diagnose rather than extend this frozen study.',
             failure_policy='Retain all requested outputs. Nonfinite arrays invalidate the corresponding metric group;never filter failed samples. Preserve recovery checkpoints and stop on unexpected shapes or replay failures.')
    # Keep the relevant frozen source closure, not unrelated feature/test data.
    sources = {name: digest for name, digest in old['source_sha256'].items()
               if name.endswith('.py') or name.startswith('third_party/')
               or name.endswith('/train_labels.pt') or name.endswith('.ckpt')}
    for name, digest in sources.items():
        if sha(name) != digest:
            raise RuntimeError('Inherited frozen input changed: ' + name)
    paths = [Path(__file__), Path('upstream/experiments/par_coarse_capacity.py'),
             Path('upstream/experiments/coarse_lora.py'),
             Path('upstream/experiments/report_par_coarse_capacity.py'),
             Path('upstream/tests/test_par_coarse_capacity.py'), Path('upstream/scripts/par_coarse_capacity.sbatch'),
             ROOT / 'preflight.json', PARENT / 'protocol.json', PARENT / 'report.json', PARENT / 'review.json',
             Path(p['training_feature_index'])]
    index = read(p['training_feature_index'])
    paths.extend(Path(index[t['sequence']]['path']) for t in targets)
    for parent in ['beta_restart', 'uniform_restart']:
        paths.extend([PARENT / parent / 'last.pt', PARENT / parent / 'complete.json'])
        paths.extend(PARENT / parent / 'step4096' / t['protein'] / 'coarse_nfe16.npz' for t in targets)
    sources.update({str(path): sha(path) for path in paths})
    p['source_sha256'] = sources
    atomic_json(ROOT / 'protocol.json', p)
    state_update(status='protocol_frozen', frozen_inputs=len(sources), parent='uniform_restart',
                 updates=UPDATES, primary_samples_per_protein=64, model_default_replaced=False)
    print('CAPACITY_PROTOCOL_FROZEN', len(sources), flush=True)


def diagnose(p):
    cases, targets = cases_and_targets()
    bank = FeatureBank(p['training_feature_index'])
    records = []
    for parent in ['beta_restart', 'uniform_restart']:
        state_update(status='diagnostic_running', diagnostic_parent=parent)
        model, view, optimizer, rng, _, _ = warm_model('adapter_only', parent)
        guard = frozen_versions(model)
        for case, target in zip(cases, targets):
            traced = trajectory_sample(model, bank(target['sequence']), 90, sample_keys(2026, 32), 16)
            expected = np.load(PARENT / parent / 'step4096' / target['protein'] / 'coarse_nfe16.npz')['coarse']
            np.testing.assert_allclose(traced['coarse'], expected, atol=1e-3, rtol=1e-5)
            folder = ROOT / 'diagnostic' / parent / target['protein']
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'trajectories.npz', **traced)
            reference = case['coarse'].numpy()
            states = [summary(v * 10., reference, 90) for v in traced['states_nm']]
            clean = [summary(v * 10., reference, 90) for v in traced['clean_predictions_nm']]
            final = geometry_diagnostic(traced['coarse'], reference, 90)
            records.append(dict(parent=parent, protein=target['protein'], state_times=np.linspace(0, 1, 17).tolist(),
                                clean_prediction_times=(np.arange(16)/16).tolist(), states=states, clean_predictions=clean,
                                final_sample_metrics={name: value.tolist() for name, value in final.items()},
                                final_replay_max_abs_A=float(np.max(np.abs(traced['coarse'] - expected)))))
        assert_frozen(model, guard)
        del model, view, optimizer, rng
    atomic_json(ROOT / 'trajectory_diagnostic.json', dict(status='complete', records=records,
                all128_parent_trajectories_retained=True, references_used_only_for_scoring=True,
                reflection_permitting_metrics_are_diagnostic_only=True, model_updates=0,
                scope='Posthoc diagnosis of two fixed training structures;no deployment correction or mechanism proof.'))
    state_update(status='trajectory_diagnostic_complete', diagnostic_trajectories=128)
    print('CAPACITY_TRAJECTORY_DIAGNOSTIC_COMPLETE', flush=True)


def sample_endpoint(p, model, bank, out, endpoint):
    keys = evaluation_keys(p['seed'])
    for target in p['targets']:
        state_update_current = dict(phase='sampling', updates=endpoint, protein=target['protein'])
        atomic_json(out / 'progress.json', state_update_current)
        clock = time.perf_counter()
        values = np.concatenate([sample_coarse_batch(model, bank(target['sequence']), 90, keys[i:i+32], 16).numpy()
                                 for i in [0, 32]])
        folder = out / f'step{endpoint}' / target['protein']
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'coarse_nfe16.npz', coarse=values,
                            valid=np.isfinite(values).all(axis=(1, 2)), seeds=np.asarray(keys, dtype=np.uint32))
        atomic_json(folder / 'sampling.json', dict(seconds=time.perf_counter()-clock, samples=64,
                    steps=16, historical_prefix=32, fresh_suffix=32, reference_used_for_generation=False))
    model.nn.init_repr_factory.features = None


def train(p, arm, resume):
    cases, _ = cases_and_targets()
    bank = FeatureBank(p['training_feature_index'])
    model, view, optimizer, rng, parent_hash, names = warm_model(arm)
    guard = frozen_versions(model)
    out = ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        raise RuntimeError('Arm already complete')
    start, trace, elapsed = 0, [], 0.
    if resume:
        saved = restore_checkpoint(out / 'last.pt', view, optimizer, rng, 'cpu')
        if saved['protocol_sha256'] != sha(ROOT / 'protocol.json') or saved['arm'] != arm:
            raise RuntimeError('Wrong recovery checkpoint')
        start, elapsed = saved['updates'], saved['training_seconds']
        restore_trace_prefix(out, start)
        trace = [json.loads(line) for line in (out / 'trace.jsonl').read_text().splitlines()]
        probes = read(out / 'native_probes.json')
    else:
        if (out / 'last.pt').exists() or (out / 'trace.jsonl').exists():
            raise RuntimeError('Explicit recovery required')
        atomic_json(out / 'baseline.json', dict(parent_checkpoint_sha256=sha(PARENT / p['parent'] / 'last.pt'),
                    parent_Adam_state_sha256=parent_hash, lora_modules=names,
                    trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad)))
        sample_endpoint(p, model, bank, out, 0)
        probes = {'0': denoising(model, cases, bank, 'sequence_uniform', p['seed'])}
        atomic_json(out / 'native_probes.json', probes)
    writer = CompactTrainingTrace(UPDATES, start)
    clock = time.perf_counter()
    for step in range(start, UPDATES):
        row = update(model, optimizer, cases, bank, p['seed'], step)
        trace.append(row)
        if (step + 1) % p['progress_every'] == 0:
            writer(out / 'trace.json', trace)
            atomic_json(out / 'progress.json', dict(phase='training', updates=step+1, total=UPDATES,
                        global_updates=PARENT_UPDATES+step+1, loss=row['loss'], lr=row['lr']))
        if step + 1 in p['endpoints']:
            elapsed += time.perf_counter() - clock
            sample_endpoint(p, model, bank, out, step+1)
            probes[str(step+1)] = denoising(model, cases, bank, 'sequence_uniform', p['seed'])
            atomic_json(out / 'native_probes.json', probes)
            clock = time.perf_counter()
        if (step+1) % p['checkpoint_every'] == 0:
            assert_frozen(model, guard)
            save_checkpoint(out / 'last.pt', view, optimizer, rng, updates=step+1,
                            global_updates=PARENT_UPDATES+step+1, arm=arm,
                            protocol_sha256=sha(ROOT / 'protocol.json'),
                            training_seconds=elapsed+time.perf_counter()-clock)
    assert_frozen(model, guard)
    for index, group in enumerate(optimizer.param_groups):
        expected = PARENT_UPDATES + UPDATES if index == 0 else UPDATES
        if any(float(optimizer.state[v]['step']) != expected for v in group['params']):
            raise RuntimeError('Adam step counters do not follow declared initialization')
    digest = hashlib.sha256(json.dumps([identity(row) for row in trace], sort_keys=True).encode()).hexdigest()
    atomic_json(out / 'complete.json', dict(status='complete', arm=arm, updates=UPDATES,
                global_updates=PARENT_UPDATES+UPDATES, parent_Adam_moments_preserved=True,
                common_event_sha256=digest, official_base_parameters_unchanged=True,
                trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                training_seconds=elapsed+time.perf_counter()-clock, host=platform.node(),
                new_development_or_test_coordinates_loaded=False))
    atomic_json(out / 'progress.json', dict(phase='complete', updates=UPDATES))
    print('CAPACITY_ARM_COMPLETE', arm, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['preflight', 'prepare', 'diagnose', 'train'])
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
        if args.stage == 'diagnose':
            diagnose(p)
        else:
            if args.arm is None:
                raise ValueError('Arm required')
            train(p, args.arm, args.resume)


if __name__ == '__main__':
    main()
