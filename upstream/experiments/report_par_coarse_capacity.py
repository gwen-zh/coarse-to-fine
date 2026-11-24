"""Report every fixed capacity-control endpoint, keeping reflection diagnostic."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from par_coarse_capacity import ARMS, evaluation_keys, learning_rate, summary
from par_two_protein_fit import identity
from run_par_coarse_capacity import ROOT, PARENT, read, state_update
from run_par_coarse_transfer import verify
from run_par_two_protein_fit import cases_and_targets


def score(values, reference):
    finite = np.isfinite(values).all(axis=(1, 2))
    if not finite.all():
        return dict(metrics_available=False, failed_samples=int((~finite).sum()), samples=len(values))
    return dict(metrics_available=True, failed_samples=0, **summary(values, reference, 90))


def main():
    p = read(ROOT / 'protocol.json')
    verify(p)
    cases, targets = cases_and_targets()
    rows, runs, curves, native, traces, arrays, baseline_hashes = [], [], {}, {}, {}, {}, []
    for arm in ARMS:
        folder = ROOT / arm
        done = read(folder / 'complete.json')
        if done['updates'] != 1024 or done['global_updates'] != 5120 or not done['official_base_parameters_unchanged']:
            raise RuntimeError('An arm did not complete the fixed budget')
        trace = read(folder / 'trace.json')
        if len(trace) != 1024 or [r['global_update'] for r in trace] != list(range(4097, 5121)):
            raise RuntimeError('Incorrect fixed noise-draw sequence')
        if [r['update'] for r in trace] != list(range(1, 1025)):
            raise RuntimeError('Incorrect local checkpoint indices')
        if any(r['lr'] != learning_rate(i) for i, r in enumerate(trace)):
            raise RuntimeError('Declared LR schedule changed')
        if any(abs(e['time'] - e['uniform_time_draw']) > 1e-7 for row in trace for e in row['batch']):
            raise RuntimeError('Uniform time rule changed')
        traces[arm] = trace
        runs.append(done)
        baseline_hashes.append(read(folder / 'baseline.json')['parent_Adam_state_sha256'])
        probes = read(folder / 'native_probes.json')
        curves[arm], native[arm] = {}, {}
        for endpoint in p['endpoints']:
            curve = dict(per_protein=[], historical32_per_protein=[], fresh32_per_protein=[])
            for case, target in zip(cases, targets):
                file = folder / f'step{endpoint}' / target['protein'] / 'coarse_nfe16.npz'
                with np.load(file) as z:
                    values = z['coarse'].copy()
                    np.testing.assert_array_equal(z['seeds'], np.asarray(evaluation_keys(p['seed']), dtype=np.uint32))
                    if not np.array_equal(z['valid'], np.isfinite(values).all(axis=(1, 2))):
                        raise RuntimeError('Invalid sample flags')
                if values.shape != (64, 23, 3):
                    raise RuntimeError('Requested primary outputs missing')
                reference = case['coarse'].numpy()
                full, historical, fresh = [score(v, reference) for v in [values, values[:32], values[32:]]]
                arrays[arm, endpoint, target['protein']] = values
                if endpoint == 0:
                    previous = np.load(PARENT / p['parent'] / 'step4096' / target['protein'] / 'coarse_nfe16.npz')['coarse']
                    np.testing.assert_allclose(values[:32], previous, atol=1e-3, rtol=1e-5)
                rows.append(dict(arm=arm, additional_updates=endpoint, protein=target['protein'],
                                 primary64=full, historical32=historical, fresh32=fresh))
                curve['per_protein'].append(full)
                curve['historical32_per_protein'].append(historical)
                curve['fresh32_per_protein'].append(fresh)
            available = all(v['metrics_available'] for v in curve['per_protein'])
            curve['metrics_available'] = available
            curve['macro_proper_RMSD_A'] = float(np.mean([v['mean_proper_RMSD_A'] for v in curve['per_protein']])) if available else None
            curve['both_training_targets_mean_within_2A'] = bool(max(v['mean_proper_RMSD_A'] for v in curve['per_protein']) <= 2) if available else None
            curves[arm][str(endpoint)] = curve
            native[arm][str(endpoint)] = {str(t): float(np.mean([r['velocity_MSE_nm2'] for r in probes[str(endpoint)] if r['time'] == t]))
                                        for t in [.05, .5, .9]}
    if len(set(baseline_hashes)) != 1:
        raise RuntimeError('Parent optimizer moments differ across arms')
    if [identity(r) for r in traces[ARMS[0]]] != [identity(r) for r in traces[ARMS[1]]]:
        raise RuntimeError('Training data/noise/time/LR are not paired')
    for target in targets:
        np.testing.assert_array_equal(arrays[ARMS[0], 0, target['protein']], arrays[ARMS[1], 0, target['protein']])
    diagnosis = read(ROOT / 'trajectory_diagnostic.json')
    if diagnosis['status'] != 'complete' or len(diagnosis['records']) != 4:
        raise RuntimeError('Parent trajectory diagnosis incomplete')

    destination = Path('paper/figures/par_coarse_capacity_v1')
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, record in zip(axes.flat, diagnosis['records']):
        for key, label, color in [('mean_proper_RMSD_A', 'Proper rotation only', '#3577a7'),
                                  ('mean_reflection_permitted_RMSD_A', 'Reflection permitted (diagnostic)', '#b36945')]:
            ax.plot(record['clean_prediction_times'], [v[key] for v in record['clean_predictions']], color=color, label=label)
        final = record['final_sample_metrics']
        ax.set(title=f"{record['parent']} · {record['protein']}\nFinal mirror-like draws: {sum(final['mirror_like'])}/32",
               xlabel='Integration time t (clean prediction)', ylabel='Mean training-target error (Å)')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Saved parent trajectories · all 32 historical draws\nPosthoc reflection diagnostic; no correction, filtering or model update', fontsize=11)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(destination / f'PAR_Coarse_Trajectory_Diagnostic.{extension}', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ['#3577a7', '#b36945']
    labels = ['AA adapter only', 'AA adapter + flow LoRA']
    for j, arm in enumerate(ARMS):
        values = [curves[arm][str(ep)]['macro_proper_RMSD_A'] for ep in p['endpoints']]
        axes[0, 0].plot(p['endpoints'], values, marker='o', color=colors[j], label=labels[j])
        final = curves[arm]['1024']['per_protein']
        axes[0, 1].bar(np.arange(2) + (j-.5)*.3, [v.get('mean_proper_RMSD_A', np.nan) for v in final],
                       width=.3, color=colors[j], label=labels[j])
        axes[1, 0].plot(p['endpoints'], [sum(v['mirror_like_count'] for v in curves[arm][str(ep)]['per_protein'])/128
                                       if curves[arm][str(ep)]['metrics_available'] else np.nan
                                       for ep in p['endpoints']], marker='o', color=colors[j])
        axes[1, 1].plot(p['endpoints'], [native[arm][str(ep)]['0.05'] for ep in p['endpoints']], marker='o', color=colors[j])
    axes[0, 0].set(xlabel='Additional optimizer updates', ylabel='Mean proper coarse RMSD (Å)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xticks([0, 1], [t['protein'] for t in targets])
    axes[0, 1].set_ylabel('Final per-protein mean proper RMSD (Å)')
    axes[0, 1].axhline(2, color='#888888', ls=':', label='Predeclared 2 Å near-fit target')
    axes[1, 0].set(xlabel='Additional optimizer updates', ylabel='Mirror-like fraction (diagnostic)')
    axes[1, 1].set(xlabel='Additional optimizer updates', ylabel='Native t=.05 velocity MSE (nm²)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Two fixed training structures · one seed · 64 noise draws/protein\nSame warm start, Adam moments, draws and LR · 1024 updates · no ensemble-quality claim', fontsize=11)
    for extension in ['pdf', 'png', 'svg']:
        fig.savefig(destination / f'PAR_Coarse_Capacity.{extension}', dpi=170)
    plt.close(fig)
    retained = sum(len(v) for v in arrays.values())
    report = dict(status='complete_awaiting_review', primary_curve=curves, native_probe_MSE=native,
                  rows=rows, runs=runs, retained_predictions=retained,
                  failed_samples=sum(row['primary64']['failed_samples'] for row in rows),
                  audits=dict(all1024_training_events_paired=True, identical_initial_predictions=True,
                              parent_Adam_moments_preserved=True, official_base_parameters_unchanged=True,
                              LR_and_global_draw_offset_passed=True, all_primary64_outputs_retained=retained == 768),
                  primary_definition=p['primary_metric'], scope=p['scope'],
                  figure=str(destination / 'PAR_Coarse_Capacity.pdf'),
                  trajectory_figure=str(destination / 'PAR_Coarse_Trajectory_Diagnostic.pdf'),
                  protocol_sha256=sha(ROOT / 'protocol.json'), model_default_replaced=False,
                  new_development_predictions=0, independent_test_claim=False, test_coordinates_opened=False, R3_opened=False)
    atomic_json(ROOT / 'report.json', report)
    state_update(status=report['status'], all_models_running=False, report_status='complete', figure=report['figure'])
    print('CAPACITY_REPORT_COMPLETE', {arm: curves[arm]['1024']['macro_proper_RMSD_A'] for arm in ARMS}, flush=True)


if __name__ == '__main__':
    main()
