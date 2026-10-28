"""Paired exploratory comparison; no generator or independent-test claim."""
import argparse, json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def paired(left,right):
    d=np.asarray(left)-np.asarray(right)
    rng=np.random.default_rng(20261026)
    boots=d[rng.integers(len(d),size=(10000,len(d)))].mean(-1)
    return dict(mean_delta=float(d.mean()),bootstrap95=np.quantile(boots,[.025,.975]).tolist(),proteins=len(d),improved=int((d<0).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text())
    data={arm:json.loads((a.run/arm/'evaluation.json').read_text()) for arm in protocol['arms']}
    names=[[r['target'] for r in data[arm]['targets']] for arm in protocol['arms']]
    if names[0]!=names[1]:raise RuntimeError('Paired proteins differ')
    comparisons={}
    def values(arm,method,metric):return [r['methods'][method][metric] for r in data[arm]['targets']]
    for metric in ('ca_rmsd','cn_rate','long_range_distance_mae','ca_clashes_per_residue'):
        np.testing.assert_allclose(values('global','original',metric),values('local','original',metric),atol=1e-5,rtol=1e-5)
        comparisons[metric]={
            'global_minus_local':paired(values('global','full',metric),values('local','full',metric)),
            'global_minus_original':paired(values('global','full',metric),values('global','original',metric)),
            'global_minus_zero_coarse':paired(values('global','full',metric),values('global','zero_coarse',metric)),
            'global_minus_shuffled_coarse':paired(values('global','full',metric),values('global','shuffled_coarse',metric))}
    g=data['global']['methods'];l=data['local']['methods']
    checks=dict(global_rmsd_5percent_better_original=g['full']['ca_rmsd']<=.95*g['original']['ca_rmsd'],
        global_rmsd_5percent_better_local=g['full']['ca_rmsd']<=.95*l['full']['ca_rmsd'],
        global_long_range_no_worse_local=g['full']['long_range_distance_mae']<=l['full']['long_range_distance_mae'],
        cn_within_half_pp_original=g['full']['cn_rate']<=g['original']['cn_rate']+.005,
        at_least_8_codes=data['global']['code_usage']>=8,
        intact_better_shuffled=g['full']['ca_rmsd']<g['shuffled_coarse']['ca_rmsd'])
    promoted=all(checks.values())
    decision=('Promising development probe; eligible to discuss fixed multi-seed replication, not generator training or independent test.' if promoted else 'Stop this fixed configuration. Do not infer global encoding is impossible; no automatic variants, extended training or generator training.')
    result=dict(status='complete',scope=protocol['scope'],means={k:v['methods'] for k,v in data.items()},paired=comparisons,gate=checks,gate_passed=promoted,decision=decision,code_usage={k:v['code_usage'] for k,v in data.items()})
    atomic_json(a.run/'comparison.json',result)
    lines=['Two-scale reconstruction probe','==============================','',protocol['scope'],'',decision,'']
    for arm in data:
        lines.append(arm+': '+json.dumps(data[arm]['methods'],ensure_ascii=False));lines.append('')
    lines.extend(['Paired exploratory intervals',json.dumps(comparisons,indent=2),'','Gate',json.dumps(checks,indent=2),'','Limitations: one seed, reused development cohort, true-coordinate encoder, extra5bits/residue vs original. Local mask permits2hop propagation. No sequence generator trained. Ablation masking/shuffling may be off-distribution; reliance is not unique global semantic disentanglement.'])
    (a.run/'results.rst').write_text('\n'.join(lines)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
