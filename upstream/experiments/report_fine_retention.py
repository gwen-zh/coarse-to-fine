"""Paired diagnosis after selection strictly on the new validation corpus."""
import argparse
import json
from pathlib import Path

import numpy as np
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    args=p.parse_args()
    base=json.loads((args.baseline/'structure_summary.json').read_text())
    baseline={r['target']:r for r in base['targets']}
    methods={'baseline':dict(base['methods']['direct'],mean_pairwise_ca_rmsd=base['mean_pairwise_ca_rmsd'])}
    comparisons={}
    for variant in ['plain','retained']:
        directory=args.run/variant
        structure=json.loads((directory/'structure_summary.json').read_text())
        selection=json.loads((directory/'summary.json').read_text())
        current={r['target']:r for r in structure['targets']}
        if set(current)!=set(baseline):raise ValueError('Diagnostic target mismatch')
        differences=[]
        for name in sorted(baseline):
            old,new=baseline[name]['samples'],current[name]['samples']
            if [r['sample'] for r in old]!=[r['sample'] for r in new]:raise ValueError('Sample mismatch')
            for row in old:
                sample=row['sample']
                with np.load(args.baseline/'predictions'/f'{name}_sample{sample}.npz') as a:expected=a['coarse'].copy()
                with np.load(directory/'predictions'/f'{name}_sample{sample}.npz') as a:actual=a['coarse'].copy()
                if not np.array_equal(expected,actual):raise ValueError('Coarse changed')
            differences.append(np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in new])-
                               np.mean([r['metrics']['closest_available_reference_ca_rmsd'] for r in old]))
        delta=np.asarray(differences)
        rng=np.random.default_rng(0)
        methods[variant]=dict(structure['methods']['direct'],selected_epoch=selection['selected_epoch'],
                              mean_pairwise_ca_rmsd=structure['mean_pairwise_ca_rmsd'])
        comparisons[variant]=dict(rmsd_difference=float(delta.mean()),
            protein_bootstrap_95ci=np.quantile(delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1),[.025,.975]).tolist(),
            coarse_mismatches=0,eligible_epochs=selection['eligible_epochs'])
    result=dict(status='complete',methods=methods,comparisons=comparisons,
        note='Fixed original coarse/backbone; tiny final-linear LoRA, matched lr/steps/rank with/without '
             'teacher retention. Geometry gate uses 24 new validation proteins, CE uses all 101 new validation '
             'proteins; epoch zero eligible. Old 164 development TRAIN sequences provide sequence-only KL '
             'anchors for retained variant. Forty repeatedly-used development proteins are diagnostics, '
             'not independent tests; geometry proxies are not full ensemble/physical validation.')
    atomic_json(args.run/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
