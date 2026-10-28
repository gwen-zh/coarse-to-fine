"""Measure teacher/predicted-prefix mismatch, without retraining or test data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Hierarchical_VQ_VAE'))
from esm.models.residual_bridge import load_residual_bridge
from train_residual_vq import atomic_json


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    targets = [t for t in json.loads((args.tokens / 'manifest.json').read_text())['targets']
               if t['subset'] == 'validation']
    methods = {}
    for mode in ['residual', 'parallel']:
        bridge, _ = load_residual_bridge(args.tokens / f'{mode}_bridge.pt')
        summary = json.loads((args.run / mode / 'summary.json').read_text())
        rows = []
        for t in targets:
            with np.load(args.cache / (t['id'] + '.npz')) as a:
                z = torch.from_numpy(a['latent'].copy())
            with np.load(args.tokens / t['tokens']) as a:
                native = torch.from_numpy(a[mode + '_tokens'].copy()).long()
            for sample in range(summary['protocol']['samples']):
                with np.load(args.run / mode / 'predictions' / f"{t['id']}_sample{sample}.npz") as a:
                    predicted = torch.from_numpy(a['tokens'].copy()).long()
                levels = []
                for level in range(len(bridge.sizes)):
                    oracle_prefix = bridge.prefix_latent(native[:, :level+1])
                    predicted_prefix = bridge.prefix_latent(predicted[:, :level+1])
                    source = z
                    oracle_source = z
                    if mode == 'residual' and level > 0:
                        source = z - bridge.prefix_latent(predicted[:, :level])
                        oracle_source = z - bridge.prefix_latent(native[:, :level])
                    table = getattr(bridge, f'centers_{level}')
                    oracle_labels = bridge.nearest(oracle_source, table)
                    assert torch.equal(oracle_labels, native[:, level])
                    conditional_labels = bridge.nearest(source, table)
                    levels.append(dict(
                        original_label_accuracy=(predicted[:, level] == native[:, level]).float().mean().item(),
                        conditional_label_change_fraction=(conditional_labels != native[:, level]).float().mean().item(),
                        conditional_label_accuracy=(predicted[:, level] == conditional_labels).float().mean().item(),
                        oracle_prefix_latent_mse=F.mse_loss(oracle_prefix, z).item(),
                        predicted_prefix_latent_mse=F.mse_loss(predicted_prefix, z).item(),
                        oracle_residual_mean_square=oracle_source.square().mean().item(),
                        generated_prefix_residual_mean_square=source.square().mean().item()))
                rows.append(dict(target=t['id'], sample=sample, levels=levels))
        means = [{key: float(np.mean([r['levels'][i][key] for r in rows]))
                  for key in rows[0]['levels'][i]} for i in range(len(bridge.sizes))]
        methods[mode] = dict(prefix_means=means, samples=len(rows), targets=rows)
    output = dict(note='Development diagnostic against one reference latent per sequence. '
                       'Conditional label means nearest residual code after subtracting generated prefix. '
                       'Label changes measure target inconsistency, not a causal demonstration that '
                       'relabeling improves generation or that a sampled alternate conformation is wrong.',
                  proteins=len(targets), methods=methods)
    atomic_json(args.run / 'prefix_diagnostic.json', output)
    print(json.dumps({k: dict(prefix_means=v['prefix_means']) for k, v in methods.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
