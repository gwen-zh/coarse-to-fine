"""Resume scoring saved paired outputs without the optional CLI import chain.

Reads precisely the three frozen npz assets used by official utils.py. No
sampling, candidate, metric, target selection or filtering changes.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from md_projection_adapter import ca_trajectory
from structural_metrics import summarize_backbone,ensemble_summary


def main():
    run=Path('artifacts/md_emulation_pilot_v1');protocol=json.loads((run/'sampling_protocol.json').read_text())
    for name,digest in protocol['sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed frozen input '+name)
    progress=json.loads((run/'progress.json').read_text());records=progress['records']
    expected={(t['id'],i) for t in protocol['targets'] for i in range(protocol['samples_per_protein'])}
    if {(r['target'],r['sample']) for r in records}!=expected or len(records)!=len(expected):raise RuntimeError('Incomplete or duplicate samples')
    sys.path.insert(0,str(Path('upstream/bioemu-benchmarks').resolve()))
    from bioemu_benchmarks.eval.md_emulation.projections import project_samples,ProjectionParameters
    from bioemu_benchmarks.eval.md_emulation.state_metric import compute_state_metrics
    base=Path('upstream/bioemu-benchmarks/bioemu_benchmarks/assets/md_emulation_benchmark_0.1/md_emulation')
    with np.load(base/'projections_mean.npz') as means,np.load(base/'projections_sqrt_inv_cov.npz') as matrices,np.load(base/'reference_projections.npz') as refs:
        parameters={k:ProjectionParameters(mean=means[k].copy(),sqrt_inv_cov=matrices[k].copy()) for k in means.files}
        references={k:refs[k].copy() for k in refs.files}
    geometry=[];summaries={}
    for arm in ['original','selected']:
        trajectories={};geometry_by_target=[]
        for target in protocol['targets']:
            bbs=[];raw=[]
            for sample in range(protocol['samples_per_protein']):
                with np.load(run/'outputs'/(target['id']+'_'+str(sample)+'.npz')) as a:bb=a[arm].copy()
                if bb.shape!=(target['length'],3,3) or not np.isfinite(bb).all():raise RuntimeError('Invalid sample; do not silently omit')
                metrics=summarize_backbone(bb);raw.append(metrics);bbs.append(bb)
                geometry.append(dict(arm=arm,target=target['id'],sample=sample,metrics=metrics))
            trajectories[target['id']]=ca_trajectory(target['sequence'],np.stack(bbs)[:,:,1,:])
            geometry_by_target.append(dict(target=target['id'],cn_rate=float(np.mean([m['backbone_cn_outliers']/(target['length']-1) for m in raw])),ca_clash_invalid_fraction=float(np.mean([not m['upstream_ca_clash_valid'] for m in raw])),spread=ensemble_summary([b[:,1] for b in bbs])['mean_pairwise_ca_rmsd']))
        projections=project_samples(trajectories,parameters);np.savez_compressed(run/(arm+'_projections.npz'),**projections)
        metrics=compute_state_metrics(projections,references,temperature_K=300.,random_seed=2026)
        metrics.to_csv(run/(arm+'_md_metrics.csv'))
        summaries[arm]=dict(md_metrics=metrics.to_dict(orient='index'),geometry_by_target=geometry_by_target,protein_mean_cn_rate=float(np.mean([x['cn_rate'] for x in geometry_by_target])),protein_mean_spread=float(np.mean([x['spread'] for x in geometry_by_target])))
        print('MD_SCORED',arm,json.dumps(metrics.loc['mean'].to_dict()),flush=True)
    atomic_json(run/'raw_geometry.json',geometry)
    atomic_json(run/'comparison.json',dict(status='complete',targets=len(protocol['targets']),samples_per_target=protocol['samples_per_protein'],arms=summaries,scope=protocol['scope'],limitations='Five proteins,32 paired samples,one seed family: exploratory and sample-size limited. Raw unfiltered outputs; CA projection only. No reference RMSD, kinetic or thermodynamic superiority claim. Gated geometry reductions are not independent evidence.'))
    progress.update(status='complete',scoring_resume='Direct frozen-asset loading avoids missing fire CLI dependency; identical official projection/metric functions and settings')
    atomic_json(run/'progress.json',progress)
    print('MD_PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
