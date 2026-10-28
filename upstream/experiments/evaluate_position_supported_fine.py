"""Fixed original coarse, original/new fine with train-only position support."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from mixed_hierarchical_io import data,tokenizer,model as original_model,sample_seed
from regularized_mixed_generator import RegularizedMixedGenerator
from position_supported_fine import sample_supported,sample_position_prior
from compare_two_level_ae_author import metrics
from structural_metrics import ensemble_summary

RUN=Path('artifacts/mixed_generator_position_support_v1')
REPAIR=Path('artifacts/mixed_generator_repair_v1')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,required=True);args=parser.parse_args()
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen position support input changed '+path)
    seed=p['seeds'][args.index];torch.set_num_threads(4);torch.manual_seed(seed)
    parent=json.loads(Path(p['parent_protocol']).read_text());items,stats=data(parent,seed);decoder=tokenizer(parent,seed)
    counts=np.load(RUN/f'train_counts_seed{seed}.npy');recount=np.zeros_like(counts)
    for item in items:
        if item['row']['subset']=='train':np.add.at(recount,(item['fine'].numpy(),np.arange(len(item['fine']))%4),1)
    if not np.array_equal(counts,recount):raise RuntimeError('Position counts are not exclusively training-derived')
    out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Position experiment complete')
    oldroot=Path(p['parent_run'])/'fine'/f'seed{seed}'
    old_checkpoint=oldroot/'final.pt';old=original_model(parent,stats,'cpu')
    old.load_state_dict(torch.load(old_checkpoint,map_location='cpu',weights_only=False)['model']);old.eval().requires_grad_(False)
    models={'original':old};provenance={'original':sha(old_checkpoint)}
    for head in ['ce','metric']:
        root=REPAIR/f'fine_{head}'/f'seed{seed}'
        if json.loads((root/'complete.json').read_text())['status']!='complete':raise RuntimeError('Repair training incomplete')
        checkpoint=root/'final.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['protocol_sha256']!=p['repair_protocol_sha256']:raise RuntimeError('Repair checkpoint protocol mismatch')
        m=RegularizedMixedGenerator(stats['mean'],stats['std'],**saved['model_config'],codes=decoder.model.fine_book.codes)
        m.load_state_dict(saved['model']);m.eval().requires_grad_(False);models[head]=m;provenance[head]=sha(checkpoint)
    atomic_json(out/'source_checkpoints.json',provenance)
    results={}
    selected=[x for x in items if x['row']['id'] in p['generation_probe_ids']]
    for arm in ['original','ce','metric','position_prior']:
        arm_dir=out/arm;arm_dir.mkdir(exist_ok=True);rows=[];started=time.perf_counter()
        for item in selected:
            name=item['row']['id'];records=[];coords=[]
            for sample in range(p['samples_per_protein']):
                key=sample_seed(seed,name,sample);fine_seed=(key+104729)%(2**32)
                with np.load(oldroot/'final_samples'/f'{name}_sample{sample}.npz') as z:c=torch.from_numpy(z['coarse'].copy()).float()
                if arm=='position_prior':f=sample_position_prior(counts,len(item['features']),fine_seed)
                else:f=sample_supported(models[arm].fine,item['features'],models[arm].normalize(c),counts,fine_seed,16)
                bb=decoder.decode(c,f).numpy();record=dict(sample=sample,seed=key,metrics=metrics(bb,item['bb'].numpy()))
                records.append(record);coords.append(bb[:,1]);np.savez_compressed(arm_dir/f'{name}_sample{sample}.npz',coarse=c.numpy(),fine=f.numpy(),backbone=bb)
            rows.append(dict(target=name,samples=records,spread=ensemble_summary(coords)['mean_pairwise_ca_rmsd']))
            atomic_json(out/'progress.json',dict(arm=arm,completed=len(rows),total=len(selected),seconds=time.perf_counter()-started))
        aggregate={k:float(np.mean([np.mean([s['metrics'][k] for s in r['samples']]) for r in rows])) for k in rows[0]['samples'][0]['metrics']}
        result=dict(status='complete',metrics=aggregate,spread=float(np.mean([r['spread'] for r in rows])),targets=rows,seconds=time.perf_counter()-started,
                    fine_forward_calls_per_sample=0 if arm=='position_prior' else 16,
                    scope='Fixed original generated coarse;TRAIN-only position support;unfiltered outputs,no reference acceptance.')
        atomic_json(arm_dir/'evaluation.json',result);results[arm]=result;print('POSITION_SUPPORT_RESULT',seed,arm,json.dumps(aggregate),'SPREAD',result['spread'],flush=True)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arms=results,source_checkpoints=provenance))


if __name__=='__main__':main()
