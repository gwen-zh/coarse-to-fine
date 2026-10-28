"""Nested32/64/128 projection scores for3 independent generation seed families."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from md_projection_adapter import ca_trajectory
from structural_metrics import summarize_backbone,ensemble_summary
sys.path.insert(0,str(Path('upstream/bioemu-benchmarks').resolve()))
from bioemu_benchmarks.eval.md_emulation.projections import ProjectionParameters,project_samples
from bioemu_benchmarks.eval.md_emulation.state_metric import DistributionMetrics2D


def main():
    run=Path('artifacts/md_emulation_expansion_v1');p=json.loads((run/'protocol.json').read_text());evaluation=json.loads((run/'evaluation_protocol.json').read_text());ph=hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest()
    for source,h in evaluation['sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=h:raise RuntimeError('Changed evaluator asset '+source)
    base=Path(evaluation['reference_dir'])
    with np.load(base/'projections_mean.npz') as a,np.load(base/'projections_sqrt_inv_cov.npz') as b,np.load(base/'reference_projections.npz') as c:
        parameters={k:ProjectionParameters(mean=a[k].copy(),sqrt_inv_cov=b[k].copy()) for k in a.files};refs={k:c[k].copy() for k in c.files}
    parent=json.loads((Path(p['parent'])/'comparison.json').read_text());records=[];raw_geometry=[];replay_checks=0
    for family in p['families']:
        for target in p['targets']:
            name=target['id'];directory=run/f'family{family}'/name;manifest=json.loads((directory/'manifest.json').read_text())
            if manifest['status']!='complete' or manifest['protocol_sha256']!=ph or [r['sample'] for r in manifest['records']]!=list(range(128)):raise RuntimeError('Incomplete sample manifest')
            bb={a:[] for a in ['original','selected']};tokens={a:[] for a in bb}
            for row in manifest['records']:
                path=Path(row['source'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Output changed')
                with np.load(path) as a:
                    for arm in bb:
                        xyz=a[arm].copy()
                        if xyz.shape!=(target['length'],3,3) or not np.isfinite(xyz).all():raise RuntimeError('Invalid sample, no filtering')
                        bb[arm].append(xyz);tokens[arm].append(a['original_fine' if arm=='original' else 'selected_fine'].copy());raw_geometry.append(dict(family=family,target=name,sample=row['sample'],arm=arm,metrics=summarize_backbone(xyz)))
            projections={arm:project_samples({name:ca_trajectory(target['sequence'],np.stack(values)[:,:,1,:])},{name:parameters[name]})[name] for arm,values in bb.items()}
            np.savez_compressed(directory/'projections.npz',**projections)
            metric=DistributionMetrics2D(refs[name],random_seed=p['metric_seed'])
            for count in p['evaluate_counts']:
                values={};geometry={}
                for arm in bb:
                    try:
                        v=metric.score_nonzero(projections[arm][:count])
                        if not np.isfinite(v).all():raise ValueError('Nonfinite official score')
                        values[arm]=dict(zip(['mae','rmse','coverage'],map(float,v)))
                    except ValueError as error:values[arm]=dict(mae=None,rmse=None,coverage=None,error=str(error))
                    if family==2026 and count==32:
                        for k in ['mae','rmse','coverage']:
                            if values[arm][k] is None or not np.isclose(values[arm][k],parent['arms'][arm]['md_metrics'][name][k],rtol=1e-8,atol=1e-8):raise RuntimeError('Parent32 score replay mismatch')
                        replay_checks+=1
                    ms=[summarize_backbone(x) for x in bb[arm][:count]]
                    geometry[arm]=dict(cn_rate=float(np.mean([x['backbone_cn_outliers']/(target['length']-1) for x in ms])),clash_invalid_fraction=float(np.mean([not x['upstream_ca_clash_valid'] for x in ms])),spread=ensemble_summary([x[:,1] for x in bb[arm][:count]])['mean_pairwise_ca_rmsd'],unique_fine_sequences=len({x.tobytes() for x in tokens[arm][:count]}))
                records.append(dict(family=family,target=name,count=count,arms=values,geometry=geometry,delta={k:values['selected'][k]-values['original'][k] if values['selected'][k] is not None and values['original'][k] is not None else None for k in ['mae','rmse','coverage']}))
            atomic_json(run/'scoring_progress.json',dict(status='running',records=records));print('MD_EXPANSION_SCORED',family,name,flush=True)
    means=[]
    for family in p['families']:
        for count in p['evaluate_counts']:
            rows=[r for r in records if r['family']==family and r['count']==count]
            if len(rows)!=5:raise RuntimeError('Missing target')
            means.append(dict(family=family,count=count,undefined_pairs=sum(r['delta']['mae'] is None for r in rows),delta={k:float(np.mean([r['delta'][k] for r in rows])) if all(r['delta'][k] is not None for r in rows) else None for k in ['mae','rmse','coverage']},arms={arm:{k:float(np.mean([r['arms'][arm][k] for r in rows])) if all(r['arms'][arm][k] is not None for r in rows) else None for k in ['mae','rmse','coverage']} for arm in ['original','selected']},geometry={arm:{k:float(np.mean([r['geometry'][arm][k] for r in rows])) for k in ['cn_rate','clash_invalid_fraction','spread','unique_fine_sequences']} for arm in ['original','selected']}))
    atomic_json(run/'raw_geometry.json',raw_geometry);atomic_json(run/'comparison.json',dict(status='complete',records=records,protein_means=means,parent32_replay_checks=replay_checks,scope=p['scope'],limitations='Five reused MD targets;3seed families and nested sample counts do not create15 independent proteins. Report all arms including undefined official scores; no scorer patches or search retuning. No automatic512 expansion.'))
    lines=['MD扩样评估：3种子族、每蛋白128样本','='*48,'',p['scope'],'']
    for row in means:lines.append(f"family{row['family']} n={row['count']}: selected-original {row['delta']}; 未定义配对蛋白数{row['undefined_pairs']}")
    (run/'results_zh.rst').write_text('\n'.join(lines)+'\n');atomic_json(run/'scoring_progress.json',dict(status='complete',paired_outputs=1920));print('MD_EXPANSION_COMPLETE',flush=True)

if __name__=='__main__':main()
