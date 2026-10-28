"""Compare external coarse continuation with the SAME fixed refiner controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_residual_vq import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--initial-control', type=Path, required=True)
    p.add_argument('--adaptive-control', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--control-kind', choices=['tiny', 'untrained_pretrained'], default='tiny')
    args = p.parse_args()
    comparisons = {}
    for label, control in [('initial', args.initial_control), ('adaptive', args.adaptive_control)]:
        current = json.loads((args.run / label / 'structure_summary.json').read_text())
        previous = json.loads((control / 'structure_summary.json').read_text())
        new = {r['target']: r for r in current['targets']}
        old = {r['target']: r for r in previous['targets']}
        if set(new) != set(old):
            raise ValueError('Target sets do not match')
        delta, released_delta = [], []
        for target in sorted(new):
            a, b = new[target]['samples'], old[target]['samples']
            if [r['sample'] for r in a] != [r['sample'] for r in b]:
                raise ValueError('Sample sets do not match')
            delta.append(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in a]) -
                         np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in b]))
            released_delta.append(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in a]) -
                                  new[target]['baseline']['closest_available_reference_ca_rmsd'])
        delta = np.asarray(delta)
        released_delta = np.asarray(released_delta)
        rng = np.random.default_rng(0)
        indices = rng.integers(0, len(delta), (5000, len(delta)))
        interval = np.quantile(delta[indices].mean(1), [.025, .975])
        comparisons[label] = dict(proteins=len(delta), pretrained_coarse=current['methods']['residual'],
            control_coarse=previous['methods']['residual'],
            released_generator=current['methods']['baseline'],
            rmsd_difference=float(delta.mean()), protein_bootstrap_95ci=interval.tolist(),
            proteins_lower_rmsd=int((delta < 0).sum()),
            rmsd_difference_from_released=float(released_delta.mean()),
            vs_released_protein_bootstrap_95ci=np.quantile(released_delta[indices].mean(1), [.025, .975]).tolist())
    note = ('Same refiner, same validation proteins, coarse source replaced. '
            'External pretrained LoRA coarse vs earlier random-initialized tiny coarse. '
            'Architectures and coarse samplers/cost differ; this does not isolate LoRA training. '
            'No sampled unadapted-pretrained coarse control yet.')
    if args.control_kind == 'untrained_pretrained':
        note = ('Same pretrained initialization, coarse sampler, seeds and frozen downstream refiner; '
                'coarse LoRA training vs no training. Control must reproduce original epoch-zero CE '
                'before sampling. Adapters selected on the same development validation data.')
    result = dict(status='complete', control_kind=args.control_kind, comparisons=comparisons,
                  note=note + ' Development diagnostics only, not an independent ensemble benchmark. '
                              'CA clash is a geometric proxy.')
    atomic_json(args.output or args.run / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
