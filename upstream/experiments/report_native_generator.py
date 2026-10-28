"""Paired native-token baseline/adaptation comparison on development diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    structures={k:json.loads((args.run/k/'structure_summary.json').read_text()) for k in ['baseline','adapted']}
    targets={k:{r['target']:r for r in s['targets']} for k,s in structures.items()}
    if set(targets['baseline'])!=set(targets['adapted']):
        raise ValueError('Diagnostic targets differ')
    differences=[]
    for name in sorted(targets['baseline']):
        rows=[targets[k][name]['samples'] for k in ['baseline','adapted']]
        if [r['sample'] for r in rows[0]] != [r['sample'] for r in rows[1]]:
            raise ValueError('Diagnostic sample sets differ')
        differences.append(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in rows[1]])-
                           np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in rows[0]]))
    d=np.asarray(differences)
    rng=np.random.default_rng(0)
    result=dict(status='complete',methods={k:s['methods']['direct'] for k,s in structures.items()},
        paired_rmsd_difference=float(d.mean()),
        protein_bootstrap_95ci=np.quantile(d[rng.integers(0,len(d),(5000,len(d)))].mean(1),[.025,.975]).tolist(),
        note='Same native token semantics, pretrained initialization, sampler, seeds and model calls. '
             'Training/selection use new screened corpus; previous 40 development proteins are diagnostics. '
             'Single-reference RMSD and CA clashes are not full ensemble/physical validity measures. '
             'Independent holdout not evaluated.')
    atomic_json(args.run/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
