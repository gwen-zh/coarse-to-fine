"""Frozen sequence-conditioning stress tests for the existing coarse flow.

No weights or sampling settings change. Wrong/zero/permuted features are
sensitivity controls, not calibrated unconditional models or deployment methods.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from hierarchical_latent_contract import pool_ca
from mixed_hierarchical_io import data,tokenizer,model,sample_seed,resampled_other_features
from coarse_solver_audit import weighted_rmsd

RUN=Path('artifacts/coarse_conditioning_audit_v1')
ARMS=['correct_sequence','other_sequence','permuted_feature_order','zero_features']

@torch.no_grad()
def execute(p,seed):
    parent=json.loads(Path(p['parent_protocol']).read_text());items,stats=data(parent,seed)
    generator=model(parent,stats,'cpu');saved=torch.load(p['generator'][str(seed)],map_location='cpu',weights_only=False)
    generator.load_state_dict(saved['model']);generator.eval().requires_grad_(False);decoder=tokenizer(parent,seed)
    selected=[x for x in items if x['row']['id'] in p['train_probe_ids']+p['development_probe_ids']]
    out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Condition audit already complete')
    rows=[];parity_max=0.;started=time.perf_counter()
    for item in selected:
        name=item['row']['id'];subset=item['row']['subset'];n=len(item['features']);m=(n+3)//4;weights=np.minimum(4,n-4*np.arange(m))
        cohort=[x for x in selected if x['row']['subset']==subset];index=next(i for i,x in enumerate(cohort) if x['row']['id']==name);other=cohort[(index+1)%len(cohort)]
        permutation=torch.randperm(n,generator=torch.Generator().manual_seed(sample_seed(seed,name,9900)))
        inputs=dict(correct_sequence=item['features'],other_sequence=resampled_other_features(other['features'],n),
                    permuted_feature_order=item['features'][permutation],zero_features=torch.zeros_like(item['features']))
        target=pool_ca(item['bb']).numpy();encoded=decoder.coarse_centers(item['coarse'],n).numpy()
        mean_centers=decoder.coarse_centers(generator.mean.expand(m,-1),n).numpy();draws=[]
        for j in range(p['samples']):
            key=sample_seed(seed,name,j);variants={};correct=None
            for arm in ARMS:
                t0=time.perf_counter();c=generator.sample_coarse(inputs[arm],key,p['steps']);centers=decoder.coarse_centers(c,n).numpy()
                if arm=='correct_sequence':
                    correct=centers
                    if subset=='validation':
                        cache=Path(p['parent_run'])/'fine'/f'seed{seed}'/'final_samples'/f'{name}_sample{j}.npz'
                        if sha(cache)!=p['cache_sha256'][str(cache)]:raise RuntimeError('Original coarse cache changed')
                        with np.load(cache) as z:difference=float(np.max(np.abs(c.numpy()-z['coarse'])))
                        parity_max=max(parity_max,difference)
                        if difference>1e-4:raise RuntimeError('Original coarse cache not reproduced')
                variants[arm]=dict(true_center_rmsd=weighted_rmsd(centers,target,weights),
                    encoded_center_rmsd=weighted_rmsd(centers,encoded,weights),
                    shape_change_vs_correct=weighted_rmsd(centers,correct,weights),
                    native_latent_mse=float((generator.normalize(c)-generator.normalize(item['coarse'])).square().mean()),
                    seconds=time.perf_counter()-t0)
            draws.append(dict(sample=j,seed=key,arms=variants))
        rows.append(dict(target=name,subset=subset,other_sequence=other['row']['id'],native_encoder_center_rmsd=weighted_rmsd(encoded,target,weights),
                         train_mean_latent_center_rmsd=weighted_rmsd(mean_centers,target,weights),samples=draws))
        atomic_json(out/'progress.json',dict(completed=len(rows),total=len(selected),last_target=name,seconds=time.perf_counter()-started))
        print('COARSE_CONDITION_PROGRESS',seed,len(rows),len(selected),flush=True)
    groups={}
    for subset in ['train','validation']:
        cohort=[r for r in rows if r['subset']==subset];summary={}
        for arm in ARMS:
            keys=cohort[0]['samples'][0]['arms'][arm]
            values={k:float(np.mean([np.mean([s['arms'][arm][k] for s in r['samples']]) for r in cohort])) for k in keys}
            delta=[np.mean([s['arms'][arm]['true_center_rmsd']-s['arms']['correct_sequence']['true_center_rmsd'] for s in r['samples']]) for r in cohort]
            values.update(mean_delta_vs_correct=float(np.mean(delta)),proteins_worse_than_correct=int(np.sum(np.asarray(delta)>0)))
            summary[arm]=values
        groups[subset]=dict(proteins=len(cohort),arms=summary,native_encoder_center_rmsd=float(np.mean([r['native_encoder_center_rmsd'] for r in cohort])),
                           train_mean_latent_center_rmsd=float(np.mean([r['train_mean_latent_center_rmsd'] for r in cohort])))
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,groups=groups,targets=rows,cache_parity_max_abs=parity_max,seconds=time.perf_counter()-started,scope=p['scope']))
    print('COARSE_CONDITION_COMPLETE',seed,flush=True)


def report(p):
    summaries=[]
    for seed in p['seeds']:summaries.append(json.loads((RUN/f'seed{seed}'/'complete.json').read_text()))
    atomic_json(RUN/'comparison.json',dict(status='complete',seeds=summaries,scope=p['scope']))
    lines=['冻结coarse生成器：序列条件有效性检查','======================================','','sameGaussian、固定16Euler步、冻结全部权重；按块残基数加权proper Kabsch对齐，主要参照真实骨架CA块中心。','','seed / subset / arm / true center RMSD(Å) / delta vs correct']
    for r in summaries:
        for subset,g in r['groups'].items():
            for arm,v in g['arms'].items():lines.append(f"{r['seed']} / {subset} / {arm} / {v['true_center_rmsd']:.6f} / {v['mean_delta_vs_correct']:+.6f}")
    lines+=['','控制条件不是训练过的无条件生成器。other-sequence采用其他蛋白纯AA特征重采样至相同长度；permuted只是打乱缓存特征顺序，没有重新编码打乱的AA序列；zero是分布外敏感性压力测试。','不因某个控制结果好而把它当作新采样策略。条件敏感性不证明学到正确序列-构象分布，也不证明泛化或因果机制。8train/16复用开发、每蛋白2相同噪声draw，两个种子；没有fine采样、独立测试或新训练。','额外保留相对native编码后decoder块中心的误差，以衔接旧solver审计；主要true_center指标直接对真实参考CA块均值，不能与完整CA RMSD或旧未对齐RMSE混用。']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    print('COARSE_CONDITION_REPORT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int,default=0);parser.add_argument('--report',action='store_true');args=parser.parse_args()
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen conditioning input changed '+path)
    torch.set_num_threads(4)
    if args.report:report(p)
    else:torch.manual_seed(p['seeds'][args.index]);execute(p,p['seeds'][args.index])

if __name__=='__main__':main()
