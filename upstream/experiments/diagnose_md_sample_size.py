"""Paired subsampling and metric-Monte-Carlo sensitivity on existing32 samples."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
sys.path.insert(0,str(Path('upstream/bioemu-benchmarks').resolve()))
from bioemu_benchmarks.eval.md_emulation.state_metric import DistributionMetrics2D


def main():
    run=Path('artifacts/md_emulation_pilot_v1/sensitivity_v1');p=json.loads((run/'protocol.json').read_text());source=Path(p['source'])
    for path,digest in p['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed frozen input '+path)
    with np.load(source/'original_projections.npz') as a:original={k:a[k].copy() for k in a.files}
    with np.load(source/'selected_projections.npz') as a:selected={k:a[k].copy() for k in a.files}
    with np.load(p['references']) as a:refs={k:a[k].copy() for k in a.files}
    archived=json.loads((source/'comparison.json').read_text());records=[]
    for target in sorted(original):
        if original[target].shape!=(32,2) or selected[target].shape!=(32,2):raise RuntimeError('Expected paired32')
        metric=DistributionMetrics2D(refs[target],random_seed=2026)
        def score(indices,count,rep,family,metric_seed,scorer):
            values={}
            for arm,data in [('original',original),('selected',selected)]:
                try:
                    result=scorer.score_nonzero(data[target][indices])
                    if not np.isfinite(result).all():raise ValueError('Nonfinite official result')
                    values[arm]=dict(zip(['mae','rmse','coverage'],map(float,result)))
                except ValueError as error:
                    values[arm]=dict(mae=None,rmse=None,coverage=None,error=type(error).__name__+': '+str(error))
            row=dict(target=target,count=count,replicate=rep,family=family,metric_seed=metric_seed,indices=list(map(int,indices)),arms=values,delta={k:values['selected'][k]-values['original'][k] if values['selected'][k] is not None and values['original'][k] is not None else None for k in ['mae','rmse','coverage']})
            records.append(row);return row
        baseline=score(np.arange(32),32,0,'subset',2026,metric)
        for arm in ['original','selected']:
            for key,value in baseline['arms'][arm].items():
                if not np.isclose(value,archived['arms'][arm]['md_metrics'][target][key],rtol=1e-10,atol=1e-10):raise RuntimeError('Archived full32 metric replay failed')
        for rep in range(p['subset_repeats']):
            rng=np.random.default_rng(int(hashlib.sha256(('md-subset-v1:'+target+':'+str(rep)).encode()).hexdigest()[:8],16));order=rng.permutation(32)
            for count in [8,16,24]:score(np.sort(order[:count]),count,rep,'subset',2026,metric)
        for metric_seed in [2027,2028]:
            score(np.arange(32),32,0,'metric_noise',metric_seed,DistributionMetrics2D(refs[target],random_seed=metric_seed))
        atomic_json(run/'progress.json',dict(status='running',completed_targets=len(set(r['target'] for r in records)),records=records));print('MD_SENSITIVITY_TARGET',target,flush=True)
    means=[]
    groups=sorted(set((r['family'],r['count'],r['replicate'],r['metric_seed']) for r in records))
    for family,count,rep,metric_seed in groups:
        rows=[r for r in records if (r['family'],r['count'],r['replicate'],r['metric_seed'])==(family,count,rep,metric_seed)]
        if len(rows)!=5:raise RuntimeError('Incomplete protein-mean group')
        means.append(dict(family=family,count=count,replicate=rep,metric_seed=metric_seed,undefined_pairs=sum(r['delta']['mae'] is None for r in rows),delta={k:float(np.mean([r['delta'][k] for r in rows])) if all(r['delta'][k] is not None for r in rows) else None for k in ['mae','rmse','coverage']}))
    atomic_json(run/'comparison.json',dict(status='complete',records=records,protein_means=means,scope=p['scope'],full32_replay=True,undefined_official_scores=[dict(target=r['target'],count=r['count'],replicate=r['replicate'],family=r['family'],metric_seed=r['metric_seed'],arm=arm,error=v['error']) for r in records for arm,v in r['arms'].items() if 'error' in v]))
    atomic_json(run/'progress.json',dict(status='complete',completed_targets=5))
    print('MD_SENSITIVITY_COMPLETE',flush=True)

if __name__=='__main__':main()
