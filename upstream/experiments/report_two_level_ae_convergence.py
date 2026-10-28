"""Report every frozen checkpoint; do not select a best checkpoint or seed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json

RUN = Path('artifacts/development/two_level_autoencoder_convergence_v1')


def main():
    pp = RUN / 'protocol.json'
    p = json.loads(pp.read_text())
    digest = hashlib.sha256(pp.read_bytes()).hexdigest()
    for source, expected in p['sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen source changed: ' + source)
    parent = json.loads((Path(p['parent_run']) / 'protocol.json').read_text())
    expected = {r['id']: r['subset'] for r in parent['targets']}
    records = []
    for seed in parent['seeds']:
        directory = RUN / f'seed{seed}'
        complete = json.loads((directory / 'complete.json').read_text())
        replay = json.loads((directory / 'replay.json').read_text())
        if complete['protocol_sha256'] != digest or complete['status'] != 'complete' or not replay['all_tensors_bitwise_equal']:
            raise RuntimeError('Run incomplete or replay unverified')
        for step in [parent['steps']] + p['checkpoints']:
            data = json.loads((directory / f'step{step}/evaluation.json').read_text())
            if data['step'] != step or data['status'] != 'complete':
                raise RuntimeError('Evaluation incomplete')
            rows = data['targets']
            if len(rows) != len(expected) or {r['target']: r['subset'] for r in rows} != expected:
                raise RuntimeError('Incomplete protein pairing')
            subsets = {}
            for subset in ['train', 'validation']:
                selected = [r for r in rows if r['subset'] == subset]
                methods = {}
                for mode in selected[0]['metrics']:
                    methods[mode] = {}
                    for key in ['rmsd', 'cn_rate', 'clashes_per_residue', 'within_rate', 'boundary_rate']:
                        values = [r['metrics'][mode][key] for r in selected]
                        if any(v is None or not np.isfinite(v) for v in values):
                            raise RuntimeError('Undefined evaluation; do not average successful proteins only')
                        methods[mode][key] = float(np.mean(values))
                groups = dict(proteins=len(selected), methods=methods)
                for key in ['coarse_rmse', 'zero_coarse_rmse', 'fine_mse', 'zero_fine_mse', 'fixed_loss']:
                    groups[key] = float(np.mean([r[key] for r in selected]))
                groups['loss_terms'] = {key: float(np.mean([r['loss_terms'][key] for r in selected]))
                                        for key in selected[0]['loss_terms']}
                groups['max_centroid_drift'] = max(r['centroid_invariance_max_error'] for r in selected)
                subsets[subset] = groups
            t, v = subsets['train'], subsets['validation']
            criteria = dict(train_rmsd_below1A=t['methods']['full']['rmsd'] <= 1.,
                            train_cn_below5pct=t['methods']['full']['cn_rate'] <= .05,
                            validation_rmsd_below2A=v['methods']['full']['rmsd'] <= 2.,
                            validation_cn_below10pct=v['methods']['full']['cn_rate'] <= .10,
                            coarse_used=all(x['zero_coarse_rmse'] >= 1.05*x['coarse_rmse'] for x in [t, v]),
                            fine_used=all(x['zero_fine_mse'] >= 1.05*x['fine_mse'] for x in [t, v]),
                            fine_cannot_move_centers=max(t['max_centroid_drift'], v['max_centroid_drift']) <= 2e-5)
            records.append(dict(seed=seed, step=step, subsets=subsets, criteria=criteria,
                                original_feasibility_pass=all(criteria.values())))
    changes = []
    for seed in parent['seeds']:
        first = next(r for r in records if r['seed'] == seed and r['step'] == parent['steps'])
        last = next(r for r in records if r['seed'] == seed and r['step'] == max(p['checkpoints']))
        for subset in ['train', 'validation']:
            a, b = first['subsets'][subset], last['subsets'][subset]
            changes.append(dict(seed=seed, subset=subset,
                           **{key+'_delta': b['methods']['full'][key]-a['methods']['full'][key]
                              for key in ['rmsd', 'cn_rate', 'within_rate', 'boundary_rate']},
                           fixed_loss_delta=b['fixed_loss']-a['fixed_loss']))
    final_records = [r for r in records if r['step'] == max(p['checkpoints'])]
    result = dict(status='complete', protocol_sha256=digest, records=records, changes_2000_to16000=changes,
                  both_seeds_final_original_feasibility_pass=all(r['original_feasibility_pass'] for r in final_records),
                  scope=p['scope'], interpretation=p['interpretation'], no_checkpoint_selection=True,
                  next='No automatic additional updates, architecture/loss changes, larger dataset or generator training.')
    atomic_json(RUN / 'comparison.json', result)
    lines = ['两层autoencoder固定训练步数诊断', '='*40, '', p['scope'], '',
             '所有行均为按蛋白等权均值。CN阈值沿用1.0–1.8Å；块内和跨块分母不同。',
             'seed / subset / steps / CA RMSD(Å) / CN(%) / within-CN(%) / boundary-CN(%) / fixed loss', '']
    for record in records:
        for subset, group in record['subsets'].items():
            m = group['methods']['full']
            lines.append(f"{record['seed']} / {subset} / {record['step']} / {m['rmsd']:.4f} / {100*m['cn_rate']:.2f} / {100*m['within_rate']:.2f} / {100*m['boundary_rate']:.2f} / {group['fixed_loss']:.6f}")
    lines += ['', '固定最终16000步，两种子均通过首轮原门槛：'+str(result['both_seeds_final_original_feasibility_pass']), '',
              '解释限制：'+p['interpretation'], '',
              '原2000步结果保留不变。本实验为失败后新增的开发诊断，不是预注册首轮成功、独立测试或优于ESM的证据。',
              '中间点完整展示；不选择最低开发误差的checkpoint。两种子/8个训练蛋白不能证明收敛或泛化。',
              '零latent与oracle仅定位误差；后者含真实结构信息，不能部署。', '',
              '曲线：convergence.png / convergence.pdf。逐蛋白记录和分项loss见各step/evaluation.json。']
    (RUN / 'results_zh.rst').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    specifications = [('rmsd', 'CA RMSD (Å)', 1.), ('cn_rate', 'C–N outliers (%)', 100.),
                      ('within_rate', 'Within-block C–N (%)', 100.), ('boundary_rate', 'Boundary C–N (%)', 100.),
                      ('fixed_loss', 'Fixed-orientation objective', 1.), ('coarse_rmse', 'Coarse center RMSE (Å)', 1.)]
    for ax, (key, label, scale) in zip(axes.flat, specifications):
        for seed, color in zip(parent['seeds'], ['#2166ac', '#b2182b']):
            selected = [r for r in records if r['seed'] == seed]
            for subset, style in [('train', '--'), ('validation', '-')]:
                values = [r['subsets'][subset][key] if key in ['fixed_loss', 'coarse_rmse']
                          else r['subsets'][subset]['methods']['full'][key] for r in selected]
                ax.plot([r['step'] for r in selected], np.array(values)*scale,
                        marker='o', linestyle=style, color=color, label=f'{seed} {subset}')
        ax.set_xlabel('Total optimizer updates')
        ax.set_ylabel(label)
        ax.set_xticks([2000, 4000, 8000, 16000], ['2k', '4k', '8k', '16k'])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed two-level AE convergence diagnostic: 8 train / 8 reused development proteins')
    fig.savefig(RUN / 'convergence.png', dpi=180)
    fig.savefig(RUN / 'convergence.pdf')
    plt.close(fig)
    print('AE_CONVERGENCE_REPORT_COMPLETE', json.dumps(dict(final_gate=result['both_seeds_final_original_feasibility_pass'], changes=changes)), flush=True)


if __name__ == '__main__':
    main()
