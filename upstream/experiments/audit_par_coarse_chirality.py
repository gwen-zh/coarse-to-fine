"""No-training target-construction and rotation/reflection field audit.

This does not correct outputs or establish a causal mechanism. It probes the
same saved generated states under fixed coordinate transformations, including
the matching transformation of the previous self-conditioning prediction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import restore_checkpoint, sha
from par_coarse_backbone import ar_condition, flow_batch
from par_coarse_capacity import geometry_diagnostic
from par_coarse_transfer import assert_frozen, frozen_versions
from par_two_protein_fit import identity, matched_draw
from run_par_coarse_capacity import ROOT as CAPACITY, read, warm_model
from run_par_coarse_transfer import FeatureBank, verify
from run_par_two_protein_fit import cases_and_targets


ROOT = Path('artifacts/par_chirality_audit_v1')
TRANSFORMS = dict(
    rotate_z90=[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
    rotate_cyclic=[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
    reflect_x=[[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])


def defect(original, transformed, matrix):
    """Row-vector convention: compare v(xR) with v(x)R, sample by sample."""
    expected = original @ matrix
    absolute = (transformed - expected).square().mean(dim=(1, 2)).sqrt()
    scale = expected.square().mean(dim=(1, 2)).sqrt()
    return absolute, absolute / scale.clamp_min(1e-8)


def state_update(**fields):
    path = ROOT / 'jobs.json'
    state = read(path) if path.is_file() else {}
    state.update(fields)
    atomic_json(path, state)
    path = Path('artifacts/run_state.json')
    all_state = read(path)
    all_state['par_chirality_audit_v1'] = state
    atomic_json(path, all_state)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'protocol.json').exists():
        raise RuntimeError('Fresh audit namespace required')
    review = read(CAPACITY / 'review.json')
    if review['status'] != 'complete_reviewed_no_material_incremental_LoRA_gain_mirror_gap_unresolved':
        raise RuntimeError('Completed capacity review required')
    previous = read(CAPACITY / 'protocol.json')
    p = dict(seed=2026, models=['parent4096', 'adapter5120', 'flow_lora5120'],
             transforms=TRANSFORMS, state_steps=[2, 8, 14], steps=16, samples_per_protein=32,
             targets=previous['targets'], training_feature_index=previous['training_feature_index'],
             target_audit='Reconstruct all4096 shared training targets from the last1024 actual updates:state+(1-t)*velocity. Verify event digests,times,and self-conditioning coins;score each recovered target against its own unaugmented training structure with proper weighted alignment. Tolerance1e-3A. This checks target augmentation/coupling,not learned sample quality.',
             field_audit='At ALL32 saved parent-uniform noise trajectories for each of2 training sequences,probe fixed times2/16,8/16,14/16. Evaluate each of3 fixed models at the same state,then at2proper rotations and1reflection. Transform the previous self-conditioning prediction by the same matrix. AA-derived conditioning,positions,time and model weights remain unchanged. No new trajectories or reference-based selection.',
             metric='Primary diagnostic:remove each predicted velocity point-mean,as the native sampler centers its state after every Euler step;compute per-sample RMS of centered-v(xR)-centered-v(x)R in native velocity units,divided by RMS of centered-v(x)R with floor1e-8. Also retain the uncentered raw-field defects. Previous self-conditioning predictions are transformed without additional centering,matching the native interface. Retain all1728 comparisons and report every protein/time/transform. There is no prespecified symmetry threshold or causal pass criterion.',
             scope='No training or model promotion. Exploratory diagnosis on two reused fixed training structures and their generated states. Approximate reflection behavior would be consistent with,but would not prove,a mirror-mode mechanism. Large defects also do not prove that a chirality-aware intervention will work. No independent ensemble-quality evidence.',
             stop_rule='One fixed audit only;no transform/time/seed scan after results,no checkpoint selection or adaptive training.')
    sources = dict(previous['source_sha256'])
    paths = [Path(__file__), Path('upstream/tests/test_par_chirality_audit.py'),
             Path('upstream/scripts/par_chirality_audit.sbatch'), CAPACITY / 'protocol.json',
             CAPACITY / 'report.json', CAPACITY / 'review.json', CAPACITY / 'protocol_clarification.json']
    paths.extend(CAPACITY / arm / 'last.pt' for arm in ['adapter_only', 'adapter_flow_lora'])
    paths.extend(CAPACITY / 'diagnostic' / 'uniform_restart' / t['protein'] / 'trajectories.npz' for t in p['targets'])
    paths.extend(CAPACITY / arm / 'trace.json' for arm in ['adapter_only', 'adapter_flow_lora'])
    sources.update({str(path): sha(path) for path in paths})
    p['source_sha256'] = sources
    atomic_json(ROOT / 'protocol.json', p)
    state_update(status='protocol_frozen', frozen_inputs=len(sources), model_updates=0)


def target_audit(p):
    cases, _ = cases_and_targets()
    traces = [read(CAPACITY / arm / 'trace.json') for arm in ['adapter_only', 'adapter_flow_lora']]
    if [identity(r) for r in traces[0]] != [identity(r) for r in traces[1]]:
        raise RuntimeError('The audited training draws were not shared')
    recovered = {case['id']: [] for case in cases}
    for row in traces[0]:
        step = row['global_update'] - 1
        for slot, event in enumerate(row['batch']):
            case = cases[(step + slot) % 2]
            if case['id'] != event['case_id']:
                raise RuntimeError('Training case order changed')
            state, velocity, t, sc, meta = matched_draw(case['coarse'], p['seed'], step, slot, 'uniform')
            if meta['noise_rotation_sha256'] != event['noise_rotation_sha256'] or float(t) != event['time'] or sc != event['self_condition']:
                raise RuntimeError('Training draw cannot be replayed')
            recovered[case['id']].append(((state + (1-t) * velocity) * 10).numpy())
    results = []
    for case in cases:
        values = np.stack(recovered[case['id']])
        metric = geometry_diagnostic(values, case['coarse'].numpy(), 90)
        maximum = float(metric['proper_RMSD_A'].max())
        results.append(dict(protein=case['protein'], draws=len(values),
                            max_proper_target_replay_RMSD_A=maximum,
                            reflected_target_count=int((metric['orthogonal_fit_determinant'] < 0).sum()),
                            passes_declared_1e_3A_tolerance=maximum < 1e-3))
    atomic_json(ROOT / 'target_audit.json', dict(records=results, total_draws=sum(r['draws'] for r in results),
                all_event_digests_replayed=True, training_updates=0))
    return results


@torch.no_grad()
def field_audit(p):
    bank = FeatureBank(p['training_feature_index'])
    rows = []
    for name in p['models']:
        arm = 'adapter_flow_lora' if name == 'flow_lora5120' else 'adapter_only'
        model, view, optimizer, rng, _, _ = warm_model(arm)
        if name != 'parent4096':
            saved = restore_checkpoint(CAPACITY / arm / 'last.pt', view, optimizer, rng, 'cpu')
            if saved['global_updates'] != 5120 or saved['arm'] != arm:
                raise RuntimeError('Wrong fixed model for field audit')
        model.eval()
        guard = frozen_versions(model)
        versions = {key: value._version for key, value in model.named_parameters()}
        for target in p['targets']:
            state_update(status='field_audit_running', model=name, protein=target['protein'])
            with np.load(CAPACITY / 'diagnostic' / 'uniform_restart' / target['protein'] / 'trajectories.npz') as z:
                states, clean = z['states_nm'].copy(), z['clean_predictions_nm'].copy()
            batch, condition = ar_condition(model, bank(target['sequence']), 90, p['samples_per_protein'])
            for step in p['state_steps']:
                t = step / p['steps']
                state = torch.from_numpy(states[step])
                sc = torch.from_numpy(clean[step-1])
                original = model.atom_diffusion.nn(flow_batch(batch, condition, state, t, sc))['coors_pred']
                for label, values in p['transforms'].items():
                    matrix = state.new_tensor(values)
                    transformed = model.atom_diffusion.nn(flow_batch(batch, condition, state @ matrix, t, sc @ matrix))['coors_pred']
                    raw_absolute, raw_relative = defect(original, transformed, matrix)
                    centered_original = original - original.mean(1, keepdim=True)
                    centered_transformed = transformed - transformed.mean(1, keepdim=True)
                    absolute, relative = defect(centered_original, centered_transformed, matrix)
                    if not torch.isfinite(absolute).all() or not torch.isfinite(relative).all():
                        raise RuntimeError('Nonfinite field comparison')
                    rows.append(dict(model=name, protein=target['protein'], time=t, transform=label,
                                     determinant=float(torch.linalg.det(matrix)),
                                     absolute_RMS=absolute.numpy().tolist(), relative_RMS=relative.numpy().tolist(),
                                     raw_absolute_RMS=raw_absolute.numpy().tolist(), raw_relative_RMS=raw_relative.numpy().tolist(),
                                     median_relative_RMS=float(relative.median()), mean_absolute_RMS=float(absolute.mean())))
        assert_frozen(model, guard)
        if versions != {key: value._version for key, value in model.named_parameters()}:
            raise RuntimeError('A parameter changed during the no-training audit')
        atomic_json(ROOT / 'field_rows.json', rows)
        del model, view, optimizer, rng
    return rows


def audit(p):
    verify(p)
    targets = target_audit(p)
    rows = field_audit(p)
    if len(rows) != 54 or sum(len(r['relative_RMS']) for r in rows) != 1728:
        raise RuntimeError('Fixed field comparisons incomplete')
    aggregates = {}
    for model in p['models']:
        aggregates[model] = {}
        for transform in p['transforms']:
            selected = [r for r in rows if r['model'] == model and r['transform'] == transform]
            relative = np.concatenate([r['relative_RMS'] for r in selected])
            absolute = np.concatenate([r['absolute_RMS'] for r in selected])
            raw_relative = np.concatenate([r['raw_relative_RMS'] for r in selected])
            aggregates[model][transform] = dict(comparisons=len(relative), median_relative_RMS=float(np.median(relative)),
                                               mean_relative_RMS=float(relative.mean()), mean_absolute_RMS=float(absolute.mean()),
                                               raw_median_relative_RMS=float(np.median(raw_relative)))
    report = dict(status='complete_awaiting_review', target_audit=targets, field_aggregates=aggregates,
                  field_rows=rows, total_field_comparisons=1728, model_updates=0,
                  scope=p['scope'], protocol_sha256=sha(ROOT / 'protocol.json'),
                  model_default_replaced=False, new_development_predictions=0,
                  test_coordinates_opened=False, R3_opened=False)
    atomic_json(ROOT / 'report.json', report)
    state_update(status=report['status'], model_updates=0, comparisons=1728)
    print('CHIRALITY_AUDIT_COMPLETE', json.dumps(aggregates), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'audit'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == 'prepare':
        prepare()
    else:
        audit(read(ROOT / 'protocol.json'))


if __name__ == '__main__':
    main()
