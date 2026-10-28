"""Four-combination stage intervention, paired by protein; development only."""
import argparse
import json
from pathlib import Path

import numpy as np
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    variants=['base_base','adapt_base','base_adapt','adapt_adapt']
    structures={k:json.loads((args.run/k/'structure_summary.json').read_text()) for k in variants}
    records={k:{r['target']:r for r in v['targets']} for k,v in structures.items()}
    names=sorted(records['base_base'])
    if any(set(records[k])!=set(names) for k in variants):
        raise ValueError('Cross target sets differ')
    outcomes={}
    for k in variants:
        outcomes[k]={}
        for metric in ['closest_available_reference_ca_rmsd','upstream_ca_clash_valid','backbone_cn_outliers']:
            outcomes[k][metric]=np.asarray([np.mean([r['metrics'][metric] for r in records[k][t]['samples']]) for t in names])
    contrasts={}
    pairs=[('coarse_on_base_fine','adapt_base','base_base'),
           ('fine_on_base_coarse','base_adapt','base_base'),
           ('coarse_on_adapted_fine','adapt_adapt','base_adapt'),
           ('fine_on_adapted_coarse','adapt_adapt','adapt_base'),
           ('joint','adapt_adapt','base_base')]
    rng=np.random.default_rng(0)
    indices=rng.integers(0,len(names),(5000,len(names)))
    for name,a,b in pairs:
        contrasts[name]={}
        for metric in outcomes[a]:
            delta=outcomes[a][metric]-outcomes[b][metric]
            contrasts[name][metric]=dict(mean_difference=float(delta.mean()),
                protein_bootstrap_95ci=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
    interaction={}
    for metric in outcomes['base_base']:
        delta=outcomes['adapt_adapt'][metric]-outcomes['adapt_base'][metric]-outcomes['base_adapt'][metric]+outcomes['base_base'][metric]
        interaction[metric]=dict(mean=float(delta.mean()),protein_bootstrap_95ci=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
    result=dict(status='complete',proteins=len(names),methods={k:v['methods']['direct'] for k,v in structures.items()},
        paired_contrasts=contrasts,stage_interaction=interaction,
        note='Same-device fine recomputation, fixed paired coarse caches. Stage-wise use of shared jointly '
             'trained weights does not predict performance of separately trained stage-only models. '
             'Post-hoc development diagnosis; CA/C-N metrics are geometric proxies, not full physical or ensemble quality.')
    atomic_json(args.run/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
