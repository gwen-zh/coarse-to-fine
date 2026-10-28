"""Full899 coarse objective comparison, reusing the fixed small-set protocol machinery."""
import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from hierarchical_latent_contract import pool_ca
from mixed_hierarchical_io import data,model,tokenizer,sample_seed
from coarse_solver_audit import weighted_rmsd
from coarse_smallset import ARMS,probe_loss,predict,state_hash
from compare_two_level_ae_author import metrics
from structural_metrics import ensemble_summary

RUN=Path('artifacts/coarse_generalization_v1')


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen smallset input changed '+path)
    return p


def load(p,seed):
    parent=json.loads(Path(p['parent_protocol']).read_text());items,stats=data(parent,seed)
    selected=[x for x in items if x['row']['id'] in p['train_ids']+p['development_ids']]
    if len(selected)!=len(p['train_ids'])+len(p['development_ids']):raise RuntimeError('Smallset cohort incomplete')
    generator=model(parent,stats,'cpu');saved=torch.load(p['warmstart'][str(seed)],map_location='cpu',weights_only=False)
    generator.load_state_dict(saved['model'])
    fine_saved=torch.load(p['frozen_fine'][str(seed)],map_location='cpu',weights_only=False)['model']
    generator.fine.load_state_dict({k[5:]:v for k,v in fine_saved.items() if k.startswith('fine.')})
    generator.eval();generator.fine.requires_grad_(False);generator.coarse_flow.requires_grad_(True)
    return generator,tokenizer(parent,seed),selected


def update(generator,decoder,item,arm,opt,p):
    generator.eval();generator.coarse_flow.train();opt.zero_grad(set_to_none=True)
    loss,terms=probe_loss(generator,decoder,item['features'],item['coarse'],arm,p['geometry_weight'])
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(generator.coarse_flow.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite probe gradient')
    opt.step()
    return dict(loss=float(loss.detach()),**{k:float(v) if torch.is_tensor(v) else v for k,v in terms.items()})


@torch.no_grad()
def evaluate(generator,decoder,items,p,seed,arm,directory,initial=False):
    generator.eval();directory.mkdir(parents=True,exist_ok=True);rows=[];parity=0.
    for item in items:
        name=item['row']['id'];n=len(item['features']);m=(n+3)//4;weights=np.minimum(4,n-4*np.arange(m))
        target=pool_ca(item['bb']).numpy();native=decoder.coarse_centers(item['coarse'],n).numpy();samples=[];positions=[];cached_point=None
        for j in range(p['samples']):
            key=sample_seed(seed,name,j)
            if arm=='regression' and cached_point is not None:c=cached_point
            else:c=predict(generator,item['features'],arm,key,p['sampling_steps'])
            if arm=='regression':cached_point=c
            centers=decoder.coarse_centers(c,n).numpy();positions.append(centers)
            if initial and arm=='flow' and item['row']['subset']=='validation':
                cache=Path(p['coarse_parent_run'])/'fine'/f'seed{seed}'/'final_samples'/f'{name}_sample{j}.npz'
                if sha(cache)!=p['cache_sha256'][str(cache)]:raise RuntimeError('Original coarse cache changed')
                with np.load(cache) as z:delta=float(np.max(np.abs(c.numpy()-z['coarse'])))
                parity=max(parity,delta)
                if delta>1e-4:raise RuntimeError('Warmstart flow parity failed')
            samples.append(dict(sample=j,seed=key,true_center_rmsd=weighted_rmsd(centers,target,weights),
                native_latent_mse=float((generator.normalize(c)-generator.normalize(item['coarse'])).square().mean())))
            np.savez_compressed(directory/f'{name}_sample{j}.npz',coarse=c.numpy(),centers=centers)
        rows.append(dict(target=name,subset=item['row']['subset'],samples=samples,
            native_encoder_center_rmsd=weighted_rmsd(native,target,weights),
            two_draw_center_spread=weighted_rmsd(positions[0],positions[1],weights)))
    groups={}
    for subset in ['train','validation']:
        cohort=[r for r in rows if r['subset']==subset]
        groups[subset]=dict(proteins=len(cohort),true_center_rmsd=float(np.mean([np.mean([s['true_center_rmsd'] for s in r['samples']]) for r in cohort])),
            latent_mse=float(np.mean([np.mean([s['native_latent_mse'] for s in r['samples']]) for r in cohort])),
            two_draw_center_spread=float(np.mean([r['two_draw_center_spread'] for r in cohort])),
            native_encoder_center_rmsd=float(np.mean([r['native_encoder_center_rmsd'] for r in cohort])))
    result=dict(status='complete',groups=groups,targets=rows,original_cache_parity_max_abs=parity,
        inference_calls_per_protein=p['samples']*p['sampling_steps'] if arm=='flow' else 1,
        scope='All899training used for updates;only8trainingprobes/16reuseddev evaluated,separate. Weighted trueblock-center properKabsch,notfullCA. Regression duplicatesare notindependentdraws.')
    atomic_json(directory/'evaluation.json',result);return result


def smoke(p):
    seed=p['seeds'][0];records=[]
    for arm in ARMS:
        torch.manual_seed(seed);generator,decoder,items=load(p,seed);item=next(x for x in items if x['row']['subset']=='train')
        fixed=(state_hash(decoder),state_hash(generator.fine));opt=torch.optim.AdamW(generator.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        for _ in range(3):update(generator,decoder,item,arm,opt,p)
        path=RUN/f'smoke_{arm}.pt';rng=np.random.default_rng(seed);save_checkpoint(path,generator,opt,rng)
        expected=update(generator,decoder,item,arm,opt,p);weights=copy.deepcopy(generator.state_dict());adam=copy.deepcopy(opt.state_dict());trng=torch.get_rng_state().clone()
        restored,_,_=load(p,seed);other=torch.optim.AdamW(restored.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        restore_checkpoint(path,restored,other,np.random.default_rng(999),torch.device('cpu'));actual=update(restored,decoder,item,arm,other,p)
        if expected!=actual or not torch.equal(trng,torch.get_rng_state()):raise RuntimeError('Smallset resume loss/RNG differs')
        for name,v in weights.items():
            if not torch.equal(v,restored.state_dict()[name]):raise RuntimeError('Smallset resume weight differs '+name)
        for key,vs in adam['state'].items():
            for name,v in vs.items():
                if not torch.equal(v,other.state_dict()['state'][key][name]):raise RuntimeError('Smallset resume Adam differs')
        if fixed!=(state_hash(decoder),state_hash(generator.fine)):raise RuntimeError('Frozen components changed')
        restored.eval()
        with torch.no_grad():c=predict(restored,item['features'],arm,seed,16);centers=decoder.coarse_centers(c,len(item['features']))
        if not torch.isfinite(centers).all():raise RuntimeError('Smallset nonfinite sampling')
        records.append(dict(arm=arm,next_update_model_adam_rng_exact=True,structural_and_fine_weights_unchanged=True,
                            trainable_parameters=sum(v.numel() for v in generator.coarse_flow.parameters())))
    atomic_json(RUN/'smoke.json',dict(status='passed',arms=records));print('COARSE_SMALLSET_SMOKE_PASSED',records,flush=True)


def train(p,seed,arm):
    if json.loads((RUN/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke required')
    torch.manual_seed(seed);rng=np.random.default_rng(seed);generator,decoder,items=load(p,seed)
    training=[x for x in items if x['row']['subset']=='train'];evaluation_items=[x for x in items if x['row']['id'] in p['train_probe_ids']+p['development_ids']];fixed=(state_hash(decoder),state_hash(generator.fine))
    out=RUN/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Probe already complete')
    opt=torch.optim.AdamW(generator.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    first=0;order=[];trace=[];psha=sha(RUN/'protocol.json');started=time.perf_counter()
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',generator,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=psha:raise RuntimeError('Resume protocol differs')
        first=state['updates'];order=state['order'];trace=[r for r in json.loads((out/'trace.json').read_text()) if r['update']<=first]
        if [r['update'] for r in trace]!=list(range(1,first+1)):raise RuntimeError('Resume trace prefix incomplete')
    elif not (out/'step0/evaluation.json').exists():evaluate(generator,decoder,evaluation_items,p,seed,arm,out/'step0',initial=True)
    if first in p['checkpoints'] and not (out/f'step{first}'/'evaluation.json').exists():
        evaluate(generator,decoder,evaluation_items,p,seed,arm,out/f'step{first}')
    for step in range(first,p['updates']):
        if step%len(training)==0:order=rng.permutation(len(training)).tolist()
        item=training[order[step%len(training)]]
        torch.manual_seed(sample_seed(seed,item['row']['id'],30000+step))
        lr=p['min_lr']+.5*(p['lr']-p['min_lr'])*(1+math.cos(math.pi*step/max(1,p['updates']-1)))
        for group in opt.param_groups:group['lr']=lr
        r=update(generator,decoder,item,arm,opt,p);trace.append(dict(update=step+1,target=item['row']['id'],lr=lr,**r))
        if (step+1)%64==0:atomic_json(out/'progress.json',dict(updates=step+1,total=p['updates'],loss=r['loss'],elapsed_seconds=time.perf_counter()-started))
        if step+1 in p['checkpoints']:
            atomic_json(out/'trace.json',trace)
            save_checkpoint(out/'last.pt',generator,opt,rng,updates=step+1,order=order,seed=seed,arm=arm,protocol_sha256=psha)
            result=evaluate(generator,decoder,evaluation_items,p,seed,arm,out/f'step{step+1}')
            print('COARSE_SMALLSET_CHECKPOINT',seed,arm,step+1,json.dumps(result['groups']),flush=True)
    if fixed!=(state_hash(decoder),state_hash(generator.fine)) or any(v.grad is not None for v in decoder.parameters()):raise RuntimeError('Frozen component changed')
    save_checkpoint(out/'final.pt',generator,opt,rng,updates=p['updates'],order=order,seed=seed,arm=arm,protocol_sha256=psha)
    final=json.loads((out/f"step{p['updates']}"/'evaluation.json').read_text())
    initial=json.loads((out/'step0/evaluation.json').read_text());train_final=final['groups']['train']['true_center_rmsd'];train_initial=initial['groups']['train']['true_center_rmsd']
    full=full_generation(generator,decoder,evaluation_items,p,seed,arm,out/'full_generation')
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arm=arm,updates=p['updates'],training_proteins=len(training),initial=initial['groups'],final=final['groups'],
        full_generation_metrics=full['metrics'],full_spread=full['spread'],coarse_spread=full['coarse_spread'],seconds=time.perf_counter()-started,frozen_components_unchanged=True))
    print('COARSE_SMALLSET_COMPLETE',seed,arm,flush=True)


@torch.no_grad()
def full_generation(generator,decoder,items,p,seed,arm,out):
    generator.eval();out.mkdir(exist_ok=True);rows=[]
    for item in items:
        name=item['row']['id']
        if name not in p['development_ids']:continue
        n=len(item['features']);weights=np.minimum(4,n-4*np.arange((n+3)//4));samples=[];ca=[];centers=[];point=None
        for j in range(p['full_samples']):
            key=sample_seed(seed,name,j)
            if arm=='regression' and point is not None:c=point
            else:c=predict(generator,item['features'],arm,key,p['sampling_steps'])
            if arm=='regression':point=c
            f=generator.fine.sample(item['features'],generator.normalize(c),(key+104729)%(2**32),16,1.)
            bb=decoder.decode(c,f).numpy();ca.append(bb[:,1]);centers.append(decoder.coarse_centers(c,n).numpy())
            samples.append(dict(sample=j,seed=key,metrics=metrics(bb,item['bb'].numpy())))
            np.savez_compressed(out/f'{name}_sample{j}.npz',backbone=bb,coarse=c.numpy(),fine=f.numpy())
        pairs=[weighted_rmsd(centers[i],centers[j],weights) for i in range(len(centers)) for j in range(i+1,len(centers))]
        rows.append(dict(target=name,samples=samples,spread=ensemble_summary(ca)['mean_pairwise_ca_rmsd'],coarse_spread=float(np.mean(pairs))))
    if len(rows)!=len(p['development_ids']):raise RuntimeError('Full generation cohort incomplete')
    result=dict(status='complete',proteins=len(rows),samples_per_protein=p['full_samples'],
        metrics={k:float(np.mean([np.mean([s['metrics'][k] for s in r['samples']]) for r in rows])) for k in rows[0]['samples'][0]['metrics']},
        spread=float(np.mean([r['spread'] for r in rows])),coarse_spread=float(np.mean([r['coarse_spread'] for r in rows])),targets=rows,
        scope='Unfilteredsequence-onlycoarse+unchangedoldfine generation. CoarseFMstochastic;regressiondeterministiccoarse,sofine-onlyspreadmustnotbeinterpretedasglobalensemblecoverage. No reference or geometryselection.')
    atomic_json(out/'evaluation.json',result);return result


def report(p):
    rows=[];baselines={}
    for seed in p['seeds']:
        pair=[]
        for arm in ARMS:
            root=RUN/f'seed{seed}'/arm;rows.append(json.loads((root/'complete.json').read_text()));trace=json.loads((root/'trace.json').read_text())
            pair.append([(r['target'],r['draw_sha256']) for r in trace])
        if pair[0]!=pair[1]:raise RuntimeError('Unpaired fullcorpus training')
        baselines[str(seed)]=json.loads(Path(p['baseline_generation'][str(seed)]).read_text())
    atomic_json(RUN/'comparison.json',dict(status='complete',rows=rows,baseline=baselines,paired_training_draws=True,scope=p['scope']))
    lines=['完整899训练：coarse flow延续与点回归','====================================','','两个目标从同一原60epochcoarse权重重新设Adam、各固定12epoch/10788更新；不使用小样本微调后的权重。训练全899；结构与原60epochfine全部冻结。8trainprobe/16复用开发分开监控，所有固定epoch1/3/6/12点保留，最终点主报告。','','seed / arm / train-probe center RMSD / dev center RMSD / fullCA RMSD / CN% / fullspread / coarsespread']
    for r in rows:
        m=r['full_generation_metrics'];lines.append(f"{r['seed']} / {r['arm']} / {r['final']['train']['true_center_rmsd']:.5f} / {r['final']['validation']['true_center_rmsd']:.5f} / {m['ca_rmsd']:.5f} / {100*m['cn_rate']:.3f} / {r['full_spread']:.5f} / {r['coarse_spread']:.5f}")
    lines+=['','回归是确定性coarse点预测，coarse ensemble spread为零是构造结果；即使参考误差改善，也不能单独称系综生成升级。原fine随机采样带来的spread不能替代全局coarse覆盖。','flow延续是匹配新训练预算的对照，不是仅追加原训练并事后选点；两个目标都固定相同10788更新/同样本顺序/noise-time抽样，但不同推理调用数，不称等FLOPs。','完整生成同16开发×8，不过滤、不根据参考接受；比较相同原fine权重、原1024码本、decoder。原发布ESM3路径不是本轮匹配对照。开发反复使用，不独立确认，也不自动替换权重。']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    atomic_json(RUN/'decision.json',dict(status='complete_requires_accuracy_geometry_diversity_review',adopt_new_weights=False,next='Reviewallarms/seedsandcoarse/fullspread;deterministicbaselineisnotanensembleupgrade.Noautomaticextraepochs.'))
    print('COARSE_GENERALIZATION_REPORT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train','report'],required=True);parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    p=protocol();torch.set_num_threads(p['threads'])
    if args.stage=='smoke':smoke(p)
    elif args.stage=='report':report(p)
    else:train(p,p['seeds'][args.index//len(ARMS)],ARMS[args.index%len(ARMS)])

if __name__=='__main__':main()
