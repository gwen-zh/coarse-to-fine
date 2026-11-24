"""Plot every sample from the frozen two-native-conformer capacity probe."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN = Path('artifacts/hierarchical_ensemble_flow_v1')
OUT = Path('paper/figures/hierarchical_ensemble_capacity_v1')


def main():
    protocol = json.loads((RUN/'protocol.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed frozen source: '+path)
    with np.load(RUN/'paired_native_labels.npz', allow_pickle=False) as data:
        native = data['coarse'].reshape(2, -1)
    direction = native[1] - native[0]
    colors = ['#23658B', '#B96920']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
    inputs = [RUN/'protocol.json', RUN/'report.json', RUN/'paired_native_labels.npz']
    for j, seed in enumerate(protocol['seeds']):
        root = RUN/'pair'/f'seed{seed}'
        result = json.loads((root/'evaluation.json').read_text())
        records = result['samples']
        expected = protocol['pair']['samples']
        if len(records) != expected or [r['sample'] for r in records] != list(range(expected)):
            raise RuntimeError('Incomplete or reordered sample set')
        projections = []
        for i in range(expected):
            path = root/f'sample{i}.npz'
            with np.load(path, allow_pickle=False) as data:
                vector = data['coarse'].reshape(-1)
            projections.append(float((vector-native[0])@direction/(direction@direction)))
            inputs.append(path)
        axes[0, 0].scatter(projections, np.full(expected, j), s=22, alpha=.4, color=colors[j], edgecolor='none')
        counts = result['cases'][0]['near_native_mode_counts']
        x = np.arange(2)+(j-.5)*.32
        axes[0, 1].bar(x, np.array(counts)/expected, width=.3, label=f'Seed {seed}', color=colors[j])
        for xx, value in zip(x, counts):
            axes[0, 1].text(xx, value/expected+.025, f'{value}/{expected}', ha='center', fontsize=9)
        rmsd = [r['nearest_reference_rmsd_A'] for r in records]
        bp = axes[1, 0].boxplot([rmsd], positions=[j], widths=.45, patch_artist=True, showfliers=False)
        bp['boxes'][0].set(facecolor=colors[j], alpha=.3)
        offsets = np.linspace(-.16, .16, expected)
        axes[1, 0].scatter(j+offsets, rmsd, s=12, color=colors[j], alpha=.5, edgecolor='none')
        mean = float(np.mean(rmsd))
        axes[1, 0].text(j, max(rmsd)+.1, f'Mean {mean:.2f}', ha='center', color=colors[j])
        geom = result['cases'][0]['geometry']
        axes[1, 1].bar(np.arange(3)+(j-.5)*.32, [100*geom[k] for k in ['cn_rate','within_cn_rate','boundary_cn_rate']], width=.3, color=colors[j], label=f'Seed {seed}')
        inputs.append(root/'evaluation.json')
    for x in [0, 1]:
        axes[0, 0].axvline(x, color='.4', lw=1, ls='--', zorder=0)
    axes[0, 0].set(yticks=[0, 1], yticklabels=['Seed 2026', 'Seed 2027'], xlabel='Coarse projection: native 1 = 0, native 2 = 1', ylim=(-.6, 1.6), title='A  Sampled coarse latents')
    axes[0, 1].axhline(.5, color='.45', ls='--', lw=1, label='50/50 training weights')
    axes[0, 1].set(xticks=[0, 1], xticklabels=['Native 1', 'Native 2'], ylabel='Fraction of 64 raw samples', ylim=(0, 1.02), title='B  Both modes found; proportions remain biased')
    axes[0, 1].legend(frameon=False, fontsize=8)
    for j, value in enumerate(protocol['pair']['native_reconstruction_ca_rmsd_A']):
        axes[1, 0].axhline(value, ls=':', color='.45', lw=1)
    axes[1, 0].set(xticks=[0, 1], xticklabels=['Seed 2026', 'Seed 2027'], ylabel='CA RMSD to closest training reference (Å)', ylim=(0, 3.3), title='C  Training-state fit after fine-code snap')
    axes[1, 0].text(.5, .1, 'Dotted: native encode/decode errors', ha='center', fontsize=8, color='.4')
    axes[1, 1].set(xticks=[0, 1, 2], xticklabels=['All links', 'Within block', 'Across blocks'], ylabel='C–N outliers (%)', title='D  Cross-block geometry still needs work')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('AA-only hierarchical flow: two-conformer capacity check', fontsize=15, y=.99)
    fig.text(.5, .015, '1JFJ / 1JFK, 134 residues; both conformers used for fitting. All 64 samples per seed; no filtering.\nThis checks fitted capacity, not unseen-protein accuracy or thermodynamic populations.', ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'png', 'svg']:
        fig.savefig(OUT/f'Hierarchical_Ensemble_Capacity.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    atomic_json(RUN/'plot_manifest.json', dict(figure=str(OUT/'Hierarchical_Ensemble_Capacity.pdf'), samples_per_seed=64, selection='All samples; no filtering or representative-example choice', inputs_sha256={str(p): sha(p) for p in inputs}, plot_source_sha256=sha(__file__)))
    print('CAPACITY_FIGURE', OUT/'Hierarchical_Ensemble_Capacity.pdf')


if __name__ == '__main__':
    main()
