import argparse,json
from pathlib import Path
import numpy as np
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'protocol.json').read_text());per={};summaries={}
    for arm in ['ce','repa']:
        rows=[json.loads((a.run/arm/'decoded'/(name+'.json')).read_text()) for name in protocol['geometry_targets']];assert len(rows)==24;per[arm]={}
        for row in rows:
            n=row['length'];record={}
            for key in ['fine','native_fine']:
                samples=[s[key] for s in row['samples']];assert len(samples)==8
                values=dict(reference_ca_rmsd=float(np.mean([s['metrics']['closest_available_reference_ca_rmsd'] for s in samples])),cn_rate=float(np.mean([s['metrics']['backbone_cn_outliers']/(n-1) for s in samples])),ca_clash_density=float(np.mean([s['metrics']['upstream_ca_clashes']/n for s in samples])),spread=row['spread'][key],quality_errors=sum(s['quality_error'] is not None for s in samples))
                for metric,label in [('reference_ca_c_n_angle_mae_degrees','ca_c_n_angle_error'),('reference_c_n_ca_angle_mae_degrees','c_n_ca_angle_error')]:values[label]=float(np.mean([s['quality'][metric] for s in samples])) if all(s['quality'] is not None for s in samples) else None
                record[key]=values
            per[arm][row['target']]=record
        summaries[arm]={key:{metric:float(np.mean([r[key][metric] for r in per[arm].values()])) if all(r[key][metric] is not None for r in per[arm].values()) else None for metric in next(iter(per[arm].values()))[key]} for key in ['fine','native_fine']}
    # Both arms must start from identical original coarse/fine token samples.
    for name in protocol['geometry_targets']:
        for sample in range(8):
            with np.load(a.run/'ce'/'predictions'/f'{name}_sample{sample}.npz') as x,np.load(a.run/'repa'/'predictions'/f'{name}_sample{sample}.npz') as y:
                for key in ['coarse','native_fine']:assert np.array_equal(x[key],y[key]),'Paired original draws must match'
    comparisons={}
    for title,left,lk,right,rk in [('repa_minus_ce','repa','fine','ce','fine'),('ce_minus_native','ce','fine','ce','native_fine'),('repa_minus_native','repa','fine','repa','native_fine')]:
        comparisons[title]={}
        for metric in next(iter(per[left].values()))[lk]:
            values=[per[left][name][lk][metric]-per[right][name][rk][metric] if per[left][name][lk][metric] is not None and per[right][name][rk][metric] is not None else None for name in protocol['geometry_targets']];comparisons[title][metric]=interval(values) if all(v is not None for v in values) else dict(mean=None,undefined_proteins=sum(v is None for v in values))
    result=dict(status='complete',scope=protocol['scope'],methods=summaries,paired_comparisons=comparisons,per_protein=per,training={arm:json.loads((a.run/arm/'summary.json').read_text()) for arm in ['ce','repa']},limitations='One seed, five epochs, one alignment layer and lambda0.1, repeatedly used validation cohort. Eight samples quantify spread, not thermodynamic coverage. Same original coarse draws, fine argmax, frozen codebooks/decoder; adapters invoked only in fine stage. CE-based checkpoint selection includes epoch0. Neither latent distance nor cosine guarantees decoded geometric similarity. No independent confirmation or overall superiority claim.')
    atomic_json(a.run/'comparison.json',result);print(json.dumps(dict(methods=summaries,comparisons=comparisons),indent=2))


if __name__=='__main__':main()
