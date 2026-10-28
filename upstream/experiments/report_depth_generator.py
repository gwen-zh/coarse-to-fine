"""Assemble completed generation pilot arms without treating samples as proteins."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_residual_vq import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    summaries, missing = {}, []
    for mode in ['residual', 'parallel', 'direct']:
        path = args.run / mode / 'structure_summary.json'
        if not path.exists():
            missing.append(mode)
        else:
            summaries[mode] = json.loads(path.read_text())
    if missing:
        atomic_json(args.run / 'comparison.json', dict(status='incomplete', missing=missing))
        raise SystemExit('Generator arms incomplete: ' + ', '.join(missing))
    names = [{r['target'] for r in s['targets']} for s in summaries.values()]
    if any(n != names[0] for n in names):
        raise ValueError('Different validation proteins across arms')
    methods = {'legacy_baseline': summaries['residual']['methods']['baseline']}
    for mode, s in summaries.items():
        training = json.loads((args.run / mode / 'summary.json').read_text())
        methods[mode] = dict(**s['methods'][mode], parameters=training['parameters'],
                             mean_pairwise_ca_rmsd=s['mean_pairwise_ca_rmsd'],
                             adapter_network_calls_per_sample=training['samples'][0]['adapter_network_calls'],
                             selected_epoch=training['selected_epoch'],
                             recorded_train_sample_seconds=training['elapsed_seconds'])
    residual = {r['target']: r for r in summaries['residual']['targets']}
    parallel = {r['target']: r for r in summaries['parallel']['targets']}
    differences = []
    for name in sorted(residual):
        a = [r['metrics'].get('closest_available_reference_ca_rmsd') for r in residual[name]['samples']]
        b = [r['metrics'].get('closest_available_reference_ca_rmsd') for r in parallel[name]['samples']]
        if all(v is not None for v in a + b):
            differences.append(float(np.mean(a) - np.mean(b)))
    delta = np.asarray(differences)
    rng = np.random.default_rng(0)
    ci = np.quantile(delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(1), [0.025, 0.975]) if len(delta) else [None, None]
    result = dict(status='complete', proteins=len(names[0]), methods=methods,
        rvq_minus_parallel_reference_rmsd=dict(mean=float(delta.mean()) if len(delta) else None,
                                             protein_bootstrap_95ci=list(ci)),
        note='Development-only fully predicted-prefix pilot. All arms use cached sequence-derived '
             'context from a frozen published generator (7 large-model calls per protein), excluded '
             'from recorded adapter runtime. RVQ/parallel share architectures, vocabularies and '
             'sampling calls. Direct fine has fewer parameters/calls and a different token budget. '
             'One upstream scaffold and two downstream samples per protein, one generator seed; '
             'not 80 independent proteins, homology-separated generalization, or matched final compute.')
    atomic_json(args.run / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
