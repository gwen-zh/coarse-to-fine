import argparse
import json
from pathlib import Path

import numpy as np
from train_residual_vq import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    methods, missing = {}, []
    baseline = json.loads((args.source / 'structure_summary.json').read_text())
    methods['initial_rvq_generator'] = baseline['methods']['residual']
    methods['published_generator'] = baseline['methods']['baseline']
    records = {}
    for variant in ['teacher', 'fixed', 'adaptive']:
        path = args.run / variant / 'structure_summary.json'
        if not path.exists():
            missing.append(variant)
            continue
        structure = json.loads(path.read_text())
        training = json.loads((args.run / variant / 'summary.json').read_text())
        methods[variant] = dict(**structure['methods']['residual'],
                               selected_epoch=training['selected_epoch'],
                               selected_validation_latent_mse=training['selected_validation_latent_mse'],
                               coarse_mismatches=training['coarse_mismatches'])
        records[variant] = {r['target']: r for r in structure['targets']}
    oracle = args.run / 'oracle' / 'summary.json'
    if oracle.exists():
        s = json.loads(oracle.read_text())
        methods['reference_assisted_greedy_oracle'] = {k: v for k, v in s.items() if k != 'targets'}
    else:
        missing.append('oracle')
    comparisons = {}
    if 'adaptive' in records:
        rng = np.random.default_rng(0)
        for control in ['teacher', 'fixed']:
            if control not in records:
                continue
            a, b = records['adaptive'], records[control]
            if set(a) != set(b):
                raise ValueError('Validation targets differ across refinements')
            delta = []
            for name in sorted(a):
                values = []
                for r in [a[name], b[name]]:
                    values.append(np.mean([x['metrics']['closest_available_reference_ca_rmsd'] for x in r['samples']]))
                delta.append(values[0] - values[1])
            delta = np.asarray(delta)
            ci = np.quantile(delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1), [0.025, 0.975])
            comparisons['adaptive_minus_' + control] = dict(reference_ca_rmsd=float(delta.mean()),
                                                           protein_bootstrap_95ci=ci.tolist())
    result = dict(status='incomplete' if missing else 'complete', missing=missing,
                  methods=methods, paired_development_comparisons=comparisons,
                  note='Development mechanism test, same frozen coarse policy, same starting refiner, '
                       'training steps and sampled-prefix cache. Fixed/adaptive use previous-policy '
                       'prefixes. Epoch zero eligible. Selection uses validation generated latent MSE; '
                       'coordinate quality reported separately. Oracle uses reference latent and is '
                       'not a deployable generator or global bound. Tiny data/model; no independent-test claim.')
    atomic_json(args.run / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    if missing:
        raise SystemExit('Incomplete refinement experiments: ' + ', '.join(missing))


if __name__ == '__main__':
    main()
