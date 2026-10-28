"""Descriptive latent/quantization audit of frozen outputs; no model fitting."""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def token_stats(ids):
    _,counts=np.unique(ids,return_counts=True);p=counts/counts.sum()
    return dict(used_codes=len(counts),perplexity=float(np.exp(-np.sum(p*np.log(p)))),largest_code_fraction=float(p.max()))


def main():
    run=Path('artifacts/development/continuous_diagnostic_v1');p=json.loads((run/'protocol.json').read_text());source=Path(p['source']);old=json.loads((source/'protocol.json').read_text())
    with np.load(old['calibration']) as a:codes=a['fine_codes'].astype(np.float64);scale=float(a['centered_rms'])
    records=[]
    for seed in old['seeds']:
        for target in old['targets']:
            with np.load(Path(old['tokens'])/(target['id']+'.npz')) as a:true=a['latent'].astype(np.float64);truth=a['fine'].copy()
            for step in [0,200]:
                with np.load(source/f'seed{seed}/step{step}'/(target['id']+'.npz')) as a:pred=a['latent'].astype(np.float64);ids=a['fine'].copy();native=a['native_fine'].copy()
                for sample,(z,s) in enumerate(zip(pred,ids)):
                    quant=codes[s];aerr=z-true;displacement=quant-z;berr=quant-true
                    before=float(np.mean(aerr*aerr));move=float(np.mean(displacement*displacement));cross=float(2*np.mean(aerr*displacement));after=float(np.mean(berr*berr))
                    if not np.isclose(after,before+move+cross,rtol=1e-10):raise RuntimeError('MSE accounting mismatch')
                    records.append(dict(seed=seed,target=target['id'],subset=target['subset'],step=step,sample=sample,generation_mse=before,snap_displacement_mse=move,cross_term=cross,snapped_target_mse=after,oracle_quantization_mse=float(np.mean((codes[truth]-true)**2)),native_code_target_mse=float(np.mean((codes[native]-true)**2)),fine_accuracy=float(np.mean(s==truth)),pred_token_stats=token_stats(s),native_token_stats=token_stats(native),oracle_token_stats=token_stats(truth),generated_centered_variance=float(np.mean((z-z.mean(0))**2))/scale**2,truth_centered_variance=float(np.mean((true-true.mean(0))**2))/scale**2))
    summaries=[]
    keys=['generation_mse','snap_displacement_mse','cross_term','snapped_target_mse','oracle_quantization_mse','native_code_target_mse','fine_accuracy','generated_centered_variance','truth_centered_variance']
    for seed in old['seeds']:
        for subset in ['train','validation']:
            for step in [0,200]:
                rows=[r for r in records if r['seed']==seed and r['subset']==subset and r['step']==step]
                summary=dict(seed=seed,subset=subset,step=step,**{k:float(np.mean([r[k] for r in rows])) for k in keys})
                for source_key in ['pred_token_stats','native_token_stats','oracle_token_stats']:
                    summary[source_key]={k:float(np.mean([r[source_key][k] for r in rows])) for k in rows[0][source_key]}
                summaries.append(summary)
    atomic_json(run/'latent_audit.json',dict(status='complete',records=records,summaries=summaries,interpretation='Latent MSE accounting includes cross term, not additive coordinate-error decomposition. Frozen decoder has no validated continuous input here. Low oracle error rules out a poor oracle ceiling for these chains, but cannot assign generated-coordinate error causally to quantization or decoder.'))
    print('CACHED_LATENT_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
