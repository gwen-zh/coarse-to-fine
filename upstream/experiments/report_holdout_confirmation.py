"""Report both fixed holdout endpoints including negative/nonfinite outcomes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from diagnostic_io import atomic_json
from holdout_confirmation import load_locked,paired_summary


def method_summary(rows,label,coarse=False):
    rmsds=[];cn=[];ca=[];clashes=[]
    for row in rows:
        values=row['methods'][label]['samples'] if coarse else [row['methods'][label]]
        rmsds.append(None if any(v['ca_rmsd'] is None or not v['all_finite'] for v in values) else float(np.mean([v['ca_rmsd'] for v in values])))
        metrics=[v['metrics'] for v in values]
        cn.append(float(np.mean([m['backbone_cn_outliers']/max(1,row['length']-1) for m in metrics])) if all('backbone_cn_outliers' in m for m in metrics) else None)
        clashes.append(float(np.mean([m['upstream_ca_clashes']/row['length'] for m in metrics])) if all('upstream_ca_clashes' in m for m in metrics) else None)
        ca.append(float(np.mean([m.get('upstream_ca_clash_valid',False) for m in metrics])))
    complete=all(v is not None and np.isfinite(v) for v in rmsds)
    mean=lambda values:float(np.mean(values)) if all(v is not None for v in values) else None
    return dict(proteins=58,missing_or_nonfinite_rmsd=sum(v is None or not np.isfinite(v) for v in rmsds),mean_ca_rmsd=mean(rmsds) if complete else None,mean_cn_rate=mean(cn),ca_validity=mean(ca),ca_clashes_per_residue=mean(clashes)),rmsds


def endpoint(run,protocol,digest,kind):
    folder=run/'medium' if kind=='medium' else run/'coarse/decoded'
    rows=[json.loads((folder/(target['id']+'.json')).read_text()) for target in protocol['targets']]
    if any(row['protocol_sha256']!=digest or row['target']!=target['id'] for row,target in zip(rows,protocol['targets'])):raise ValueError('Holdout cohort/protocol mismatch')
    labels=['original_medium','repaired_medium','original_fine'] if kind=='medium' else ['control','adapted']
    summaries={};values={}
    for label in labels:summaries[label],values[label]=method_summary(rows,label,kind=='coarse')
    before,after=labels[:2];complete=all(v is not None and np.isfinite(v) for name in [before,after] for v in values[name])
    result=dict(status='complete' if complete else 'nonfinite_primary_no_paired_claim',protocol_sha256=digest,kind=kind,methods=summaries,primary=paired_summary(np.asarray(values[after])-np.asarray(values[before])) if complete else None,note=protocol['scope'],statistics=protocol['statistics'])
    if kind=='coarse':result['mean_two_sample_spread']={label:float(np.mean([r['methods'][label]['mean_pairwise_ca_rmsd'] for r in rows])) if all(r['methods'][label]['mean_pairwise_ca_rmsd'] is not None for r in rows) else None for label in labels}
    atomic_json(run/(kind+'_summary.json'),result)
    return result,values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--part',choices=['medium','all'],required=True);args=parser.parse_args()
    protocol,digest=load_locked(args.run);kinds=['medium'] if args.part=='medium' else ['medium','coarse'];results={}
    figure,axes=plt.subplots(1,len(kinds),figsize=(5*len(kinds),4.6),squeeze=False)
    for ax,kind in zip(axes[0],kinds):
        result,values=endpoint(args.run,protocol,digest,kind);results[kind]=result
        if result['status']=='complete':
            before,after=('original_medium','repaired_medium') if kind=='medium' else ('control','adapted')
            ax.scatter(values[before],values[after],s=20,alpha=.7);limit=max(max(values[before]),max(values[after]))*1.05
            ax.plot([0,limit],[0,limit],color='gray',linestyle='--');ax.set(xlim=(0,limit),ylim=(0,limit),xlabel=before+' CA RMSD (A)',ylabel=after+' CA RMSD (A)',title=kind+':58 fixed holdout proteins')
        else:ax.text(.1,.5,'Nonfinite primary outcomes; no paired claim')
        print('HOLDOUT_ENDPOINT_REPORT',kind,json.dumps(result),flush=True)
    figure.tight_layout();name='medium_confirmation' if args.part=='medium' else 'holdout_confirmation'
    figure.savefig(args.run/(name+'.png'),dpi=180);figure.savefig(args.run/(name+'.svg'));plt.close(figure)
    if args.part=='all':atomic_json(args.run/'summary.json',dict(status='complete' if all(r['status']=='complete' for r in results.values()) else 'nonfinite_primary',protocol_sha256=digest,endpoints=results,no_model_selection_on_holdout=True))


if __name__=='__main__':main()
