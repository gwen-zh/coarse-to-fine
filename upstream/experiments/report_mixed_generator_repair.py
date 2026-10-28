"""Report every final composition; descriptive reused-development evidence only."""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from train_mixed_generator_repair import RUN, protocol


def delta(candidate, baseline):
    old = {r['target']: r for r in baseline['targets']}
    if set(old) != {r['target'] for r in candidate['targets']}:
        raise RuntimeError('Unpaired report cohorts')
    changes = {}
    for key in ['ca_rmsd', 'cn_rate', 'ca_clashes_per_residue', 'peptide_planarity_deviation_degrees', 'spread']:
        values = []
        for row in candidate['targets']:
            reference = old[row['target']]
            a = row['spread'] if key == 'spread' else np.mean([s['metrics'][key] for s in row['samples']])
            b = reference['spread'] if key == 'spread' else np.mean([s['metrics'][key] for s in reference['samples']])
            values.append(float(a-b))
        rng = np.random.default_rng(20261027)
        bootstrap = np.mean(np.asarray(values)[rng.integers(len(values), size=(5000, len(values)))], axis=1)
        changes[key] = dict(mean=float(np.mean(values)), descriptive95ci=np.quantile(bootstrap, [.025, .975]).tolist(),
                            protein_lower=sum(v < 0 for v in values), protein_higher=sum(v > 0 for v in values),
                            unit='fraction' if key == 'cn_rate' else 'as_named')
    return changes


def main():
    p = protocol()
    rows = []
    lines = ['生成器泛化修正：固定预算开发实验', '='*44, '', p['scope'], '',
             '所有配置同16开发蛋白、每蛋白8样本、同随机种子键、同冻结tokenizer。无几何接受规则或参考挑选。',
             '分别报告两个训练种子；不把同一蛋白的多个样本当成独立蛋白。', '',
             'seed / coarse / fine / CA RMSD(Å) / CN(%) / CA clashes per residue / spread(Å)', '']
    for seed in p['seeds']:
        base_path = Path(p['parent_run'])/'fine'/f'seed{seed}'/'final_samples/evaluation.json'
        baseline = json.loads(base_path.read_text())
        coarse = json.loads((RUN/'coarse'/f'seed{seed}'/'complete.json').read_text())
        variants = [('original', 'original', baseline),
                    ('repaired', 'original', coarse['evaluations']['repaired_coarse_original_fine'])]
        for head in p['fine_heads']:
            path = RUN/f'fine_{head}'/f'seed{seed}'/'complete.json'
            complete = json.loads(path.read_text())
            variants.extend((source, head, complete['evaluations'][source]) for source in ['original', 'repaired'])
        for source, head, result in variants:
            m = result['metrics']
            rows.append(dict(seed=seed, coarse=source, fine=head, metrics=m, spread=result['spread'],
                             paired_deltas=delta(result, baseline), targets=result['targets']))
            lines.append(f"{seed} / {source} / {head} / {m['ca_rmsd']:.4f} / {100*m['cn_rate']:.2f} / {m['ca_clashes_per_residue']:.5f} / {result['spread']:.4f}")
    limitations = ('Reused development cohort, exploratory multiple configurations. Bootstrap intervals are descriptive, '
                   'not confirmatory significance. Fixed final checkpoints, all arms and both seeds are reported. '
                   'Regularization/capacity/condition jitter are a joint recipe; metric head versus CE head is a '
                   'same-schedule comparison with different output parameter counts. The old60epoch versus '
                   'new20coarse/12fine epochs comparison is not matched training compute. No author-matched '
                   'superiority or thermodynamic coverage claim;8sample spread is a limited diversity proxy.')
    lines += ['', '原始失败实验保留；本轮不回写原门槛。没有依据验证集反复挑最佳epoch。', limitations,
              '', '逐蛋白结果及配对差见comparison.json；多个开发配置的区间不作为独立确认。']
    atomic_json(RUN/'comparison.json', dict(status='complete', rows=rows, limitations=limitations))
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    state_path = Path('artifacts/run_state.json')
    state = json.loads(state_path.read_text())
    state['mixed_generator_repair_v1'].update(status='complete', report=str(RUN/'results_zh.rst'))
    atomic_json(state_path, state)
    print('GENERATOR_REPAIR_REPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
