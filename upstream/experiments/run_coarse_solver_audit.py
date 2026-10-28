"""Frozen coarse numerical convergence audit; no new training or fine selection."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from mixed_hierarchical_io import data,tokenizer,model,sample_seed
from mixed_hierarchical_generator import pooled_sequence
from coarse_solver_audit import integrate,weighted_rmsd
RUN=Path('artifacts/coarse_solver_convergence_v1')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen solver audit input changed '+path)
    seed=p['seeds'][args.index];torch.set_num_threads(4);torch.manual_seed(seed)
    parent=json.loads(Path(p['parent_protocol']).read_text());items,stats=data(parent,seed)
    generator=model(parent,stats,'cpu');saved=torch.load(p['generator'][str(seed)],map_location='cpu',weights_only=False);generator.load_state_dict(saved['model']);generator.eval().requires_grad_(False)
    decoder=tokenizer(parent,seed);selected=[x for x in items if x['row']['id'] in p['train_probe_ids']+p['development_probe_ids']]
    if len(selected)!=len(p['train_probe_ids'])+len(p['development_probe_ids']):raise RuntimeError('Incomplete convergence cohort')
    out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Solver audit immutable')
    rows=[];parity_max=0.;started=time.perf_counter()
    for item in selected:
        name=item['row']['id'];n=len(item['features']);condition=pooled_sequence(item['features'])[None]
        valid=torch.ones(condition.shape[:2],dtype=torch.bool);weights=np.minimum(4,n-4*np.arange(len(condition[0])))
        target=item['centers'].numpy();native=decoder.coarse_centers(item['coarse'],n).numpy();draws=[]
        for sample in range(p['samples_per_protein']):
            key=sample_seed(seed,name,sample);rng=torch.Generator().manual_seed(key)
            initial=torch.randn((1,len(condition[0]),len(generator.mean)),generator=rng)
            variants={}
            for solver in p['solvers']:
                clock=time.perf_counter();state=integrate(generator.coarse_flow,initial,condition,valid,solver['steps'],solver['method']);seconds=time.perf_counter()-clock
                coarse=generator.denormalize(state[0]);centers=decoder.coarse_centers(coarse,n).numpy()
                if solver['name']=='euler16' and item['row']['subset']=='validation':
                    path=Path(p['parent_run'])/'fine'/f'seed{seed}'/'final_samples'/f'{name}_sample{sample}.npz'
                    with np.load(path) as z:cached=z['coarse']
                    difference=float(np.max(np.abs(coarse.numpy()-cached)));parity_max=max(parity_max,difference)
                    if difference>1e-4:raise RuntimeError('Original16step coarse not reproduced')
                variants[solver['name']]=dict(coarse=coarse.numpy(),state=state[0].numpy(),centers=centers,integration_seconds=seconds,
                    weighted_block_center_rmsd=weighted_rmsd(centers,target,weights),normalized_native_latent_mse=float((state[0]-generator.normalize(item['coarse'])).square().mean()))
                directory=out/solver['name'];directory.mkdir(exist_ok=True)
                np.savez_compressed(directory/f'{name}_sample{sample}.npz',coarse=coarse.numpy(),centers=centers)
            finest=variants['euler128'];records={}
            for name_solver,r in variants.items():
                records[name_solver]={k:r[k] for k in ['integration_seconds','weighted_block_center_rmsd','normalized_native_latent_mse']}
                records[name_solver].update(normalized_endpoint_mse_to_euler128=float(np.mean((r['state']-finest['state'])**2)),
                     weighted_center_rmsd_to_euler128=weighted_rmsd(r['centers'],finest['centers'],weights))
            draws.append(dict(sample=sample,seed=key,solvers=records))
        rows.append(dict(target=name,subset=item['row']['subset'],native_encoder_decoder_center_rmsd=weighted_rmsd(native,target,weights),samples=draws))
        atomic_json(out/'progress.json',dict(completed=len(rows),total=len(selected),last_target=name,seconds=time.perf_counter()-started));print('COARSE_SOLVER_PROGRESS',seed,name,len(rows),len(selected),flush=True)
    groups={}
    for subset in ['train','validation']:
        group=[r for r in rows if r['subset']==subset];summaries={}
        for solver in p['solvers']:
            name=solver['name'];keys=group[0]['samples'][0]['solvers'][name]
            summary={k:float(np.mean([np.mean([s['solvers'][name][k] for s in r['samples']]) for r in group])) for k in keys}
            changes=np.asarray([np.mean([s['solvers'][name]['weighted_block_center_rmsd']-s['solvers']['euler16']['weighted_block_center_rmsd'] for s in r['samples']]) for r in group])
            rng=np.random.default_rng(20261027);boot=changes[rng.integers(len(changes),size=(5000,len(changes)))].mean(1)
            summary['delta_to_euler16']=dict(mean=float(changes.mean()),descriptive95ci=np.quantile(boot,[.025,.975]).tolist())
            summary['nfe']=solver['steps']*(2 if solver['method']=='midpoint' else 1);summaries[name]=summary
        groups[subset]=dict(proteins=len(group),solvers=summaries,native_encoder_decoder_center_rmsd=float(np.mean([r['native_encoder_decoder_center_rmsd'] for r in group])))
    result=dict(status='complete',seed=seed,groups=groups,targets=rows,euler16_cache_parity_max_abs=parity_max,seconds=time.perf_counter()-started,scope=p['scope'])
    atomic_json(out/'complete.json',result)
    print('COARSE_SOLVER_COMPLETE',seed,json.dumps(groups),flush=True)


if __name__=='__main__':main()
