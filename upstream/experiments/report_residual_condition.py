"""Matched residual/parallel/prediction-shuffle diagnostic on original scaffold."""
import argparse
import json
from pathlib import Path

import numpy as np
from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--variant',choices=['residual','geometry'],default='residual')
    args=p.parse_args()
    baseline=json.loads((args.baseline/'structure_summary.json').read_text())
    structures={'baseline':baseline}
    first,second=('residual','parallel') if args.variant=='residual' else ('geometry','no_geometry')
    labels=[first,second,first+'_shuffled',second+'_shuffled']
    for label in labels:
        structures[label]=json.loads((args.run/label/'structure_summary.json').read_text())
    records={k:{r['target']:r for r in s['targets']} for k,s in structures.items()}
    names=sorted(records['baseline'])
    if any(set(r)!=set(names) for r in records.values()):raise ValueError('Target mismatch')
    methods={}
    for label,s in structures.items():
        methods[label]=dict(s['methods']['direct'],mean_pairwise_ca_rmsd=s['mean_pairwise_ca_rmsd'])
        if label!='baseline':
            selection=json.loads((args.run/label/'summary.json').read_text())
            methods[label]['selected_epoch']=selection['selected_epoch']
    for label in [first,second]:
        for name in names:
            for sample in range(2):
                with np.load(args.baseline/'predictions'/f'{name}_sample{sample}.npz') as a:expected=a['coarse'].copy()
                with np.load(args.run/label/'predictions'/f'{name}_sample{sample}.npz') as a:actual=a['coarse'].copy()
                if not np.array_equal(expected,actual):raise ValueError('Coarse scaffold changed')
    pairs=[(first+'_minus_baseline',first,'baseline'),(second+'_minus_baseline',second,'baseline'),
           (first+'_minus_'+second,first,second),(first+'_minus_shuffled',first,first+'_shuffled')]
    rng=np.random.default_rng(0);idx=rng.integers(0,len(names),(5000,len(names)))
    contrasts={}
    for name,a,b in pairs:
        contrasts[name]={}
        for metric in ['closest_available_reference_ca_rmsd','upstream_ca_clash_valid','backbone_cn_outliers']:
            delta=np.asarray([np.mean([r['metrics'][metric] for r in records[a][t]['samples']])-
                              np.mean([r['metrics'][metric] for r in records[b][t]['samples']]) for t in names])
            contrasts[name][metric]=dict(mean_difference=float(delta.mean()),
                protein_bootstrap_95ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
    result=dict(status='complete',proteins=len(names),methods=methods,paired_contrasts=contrasts,coarse_mismatches=0,
        note='Frozen original coarse/fine weights; new predicted mid conditions zero-initialized adapter before final4 original fine layers. '
             'Residual/parallel same 512 codes, architecture and steps; shuffled control reorders per-protein predicted mid conditions. '
             'Selection only on new validation CE and preselected 24 new validation geometry targets; epoch0 eligible. '
             'Forty reused development proteins are diagnostics, not independent tests. Additional original coarse-context '
             'forward makes runtime larger than original baseline; no matched-compute superiority or full ensemble claim.')
    if args.variant=='geometry':
        result['note']='64 fixed-order new TRAIN proteins, 3 epochs; same fixed residual mid and zero-initialized condition architecture in both arms. Geometry arm adds biased hard-token decoder surrogate gradients; control computes and discards same decoder gradients. Original generator and decoder frozen. Selection uses original 101 new validation CE and 24 geometry targets, epoch0 eligible. Forty reused development proteins remain diagnostics; holdout untouched. Not an independent or equal-inference-compute superiority claim.'
    atomic_json(args.run/'comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
