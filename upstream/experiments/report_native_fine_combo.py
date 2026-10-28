"""Exploratory protein-paired native fine-LoRA/search comparison."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from structural_metrics import aligned_rmsd
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'protocol.json').read_text());records=[];arrays={}
    for item in protocol['targets']:
        d=json.loads((a.run/item['target']/'summary.json').read_text());assert d['status']=='complete'
        for parent in protocol['parents']:
            x=d['parents'][parent];assert x['decoder_calls']==protocol['decoder_budget']
            with np.load(a.run/item['target']/(parent+'.npz')) as z:
                for phase in ['original','searched']:arrays[(item['target'],parent,phase)]=z[phase+'_backbone'][:,1].copy()
            for phase in ['original','searched']:
                m=x[phase+'_metrics'];q=x[phase+'_quality'];assert q is not None, x[phase+'_quality_error']
                records.append(dict(protein=item['protein'],target=item['target'],method=parent+'_'+phase,reference_ca_rmsd=m['closest_available_reference_ca_rmsd'],cn_count=m['backbone_cn_outliers'],cn_rate=m['backbone_cn_outliers']/(d['length']-1),ca_valid=m['upstream_ca_clash_valid'],ca_clash_density=m['upstream_ca_clashes']/d['length'],ca_c_n_angle_error=q['reference_ca_c_n_angle_mae_degrees'],c_n_ca_angle_error=q['reference_c_n_ca_angle_mae_degrees'],search_seconds=x['search_seconds'] if phase=='searched' else 0.,decoder_calls=x['decoder_calls'] if phase=='searched' else 0,padding_calls=x['padding_calls'] if phase=='searched' else 0,returned_original=x['returned_original'] if phase=='searched' else False))
    assert len(records)==480;proteins=sorted(set(r['protein'] for r in records));assert len(proteins)==40
    methods=[parent+'_'+phase for parent in protocol['parents'] for phase in ['original','searched']];per={};summary={}
    keys=[k for k in records[0] if k not in ['protein','target','method']]
    for method in methods:
        parent,phase=method.rsplit('_',1);per[method]={}
        for protein in proteins:
            rows=[r for r in records if r['method']==method and r['protein']==protein];assert len(rows)==2
            row={k:float(np.mean([r[k] for r in rows])) for k in keys};row['two_sample_spread']=aligned_rmsd(*[arrays[(r['target'],parent,phase)] for r in rows]);assert row['two_sample_spread'] is not None;per[method][protein]=row
        summary[method]={k:float(np.mean([r[k] for r in per[method].values()])) for k in next(iter(per[method].values()))}
    pairs=[('plain_original','native_original'),('retained_original','native_original'),('native_searched','native_original')]
    for parent in ['plain','retained']:pairs.extend([(parent+'_searched',parent+'_original'),(parent+'_searched','native_searched'),(parent+'_searched','native_original')])
    comparisons={left+'_minus_'+right:{k:interval([per[left][protein][k]-per[right][protein][k] for protein in proteins]) for k in next(iter(per[left].values()))} for left,right in pairs}
    result=dict(status='complete',proteins=40,contexts=80,outputs=480,scope=protocol['scope'],methods=summary,paired_comparisons=comparisons,per_protein=per,limitations='Repeatedly used development cohort: all intervals exploratory, no independent superiority confirmation. CN/CA gains within each parent are enforced by acceptance. Two-sample spread is a diversity warning, not coverage. References used only after search. Own-parent displacement bound does not guarantee reference RMSD preservation. Search runtime excludes generator, initial decode and model loading; not end-to-end overhead.')
    atomic_json(a.run/'comparison.json',result)
    with (a.run/'per_sample.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    text=['Original-path fine LoRA plus fixed-budget search','===============================================','',result['limitations'],'']
    for method,vals in summary.items():text.append(f"{method}: RMSD={vals['reference_ca_rmsd']:.6f} A; mean CN count={vals['cn_count']:.4f}; CN rate={100*vals['cn_rate']:.4f}%; two-sample spread={vals['two_sample_spread']:.6f} A")
    (a.run/'results.rst').write_text('\n'.join(text)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
