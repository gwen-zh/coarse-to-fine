"""Fixed-budget eight-protein coarse fitting, with separate reused-dev reporting."""
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

RUN=Path('artifacts/coarse_smallset_probe_v1')


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
    generator.load_state_dict(saved['model']);generator.eval();generator.fine.requires_grad_(False);generator.coarse_flow.requires_grad_(True)
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
        scope='8training sequences only;development16reused,separate. Weighted true CA-block-center properKabsch,not fullbackbone. Regression deterministic duplicate outputs are not independent draws;zero spread is built in.')
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
    training=[x for x in items if x['row']['subset']=='train'];fixed=(state_hash(decoder),state_hash(generator.fine))
    out=RUN/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Probe already complete')
    opt=torch.optim.AdamW(generator.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    first=0;order=[];trace=[];psha=sha(RUN/'protocol.json');started=time.perf_counter()
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',generator,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=psha:raise RuntimeError('Resume protocol differs')
        first=state['updates'];order=state['order'];trace=[r for r in json.loads((out/'trace.json').read_text()) if r['update']<=first]
        if [r['update'] for r in trace]!=list(range(1,first+1)):raise RuntimeError('Resume trace prefix incomplete')
    elif not (out/'step0/evaluation.json').exists():evaluate(generator,decoder,items,p,seed,arm,out/'step0',initial=True)
    if first in p['checkpoints'] and not (out/f'step{first}'/'evaluation.json').exists():
        evaluate(generator,decoder,items,p,seed,arm,out/f'step{first}')
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
            result=evaluate(generator,decoder,items,p,seed,arm,out/f'step{step+1}')
            print('COARSE_SMALLSET_CHECKPOINT',seed,arm,step+1,json.dumps(result['groups']),flush=True)
    if fixed!=(state_hash(decoder),state_hash(generator.fine)) or any(v.grad is not None for v in decoder.parameters()):raise RuntimeError('Frozen component changed')
    save_checkpoint(out/'final.pt',generator,opt,rng,updates=p['updates'],order=order,seed=seed,arm=arm,protocol_sha256=psha)
    final=json.loads((out/f"step{p['updates']}"/'evaluation.json').read_text())
    initial=json.loads((out/'step0/evaluation.json').read_text());train_final=final['groups']['train']['true_center_rmsd'];train_initial=initial['groups']['train']['true_center_rmsd']
    # Engineering screen only; it does not authorize further training or deployment.
    gate=dict(train_center_rmsd_below2A=train_final<=2.,train_mean_error_reduced75pct=train_final<=.25*train_initial)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arm=arm,updates=p['updates'],initial=initial['groups'],final=final['groups'],engineering_fit_gate=gate,
        learned_eight_protein_point_targets=all(gate.values()),seconds=time.perf_counter()-started,frozen_components_unchanged=True))
    print('COARSE_SMALLSET_COMPLETE',seed,arm,flush=True)


def report(p):
    rows=[]
    for seed in p['seeds']:
        pair=[]
        for arm in ARMS:
            root=RUN/f'seed{seed}'/arm;rows.append(json.loads((root/'complete.json').read_text()));trace=json.loads((root/'trace.json').read_text())
            pair.append([(r['target'],r['draw_sha256']) for r in trace])
        if pair[0]!=pair[1]:raise RuntimeError('Unpaired training examples/noise/time')
    atomic_json(RUN/'comparison.json',dict(status='complete',rows=rows,paired_training_draws=True,scope=p['scope']))
    lines=['Coarse小样本拟合：flow与确定性回归','===================================','','只在8训练蛋白微调原60epochcoarse权重；每组固定1024步。16复用开发仅评估；不训练encoder/decoder/fine。','','seed / arm / initial_train / final_train / initial_dev / final_dev / engineering_fit_gate']
    for r in rows:lines.append(f"{r['seed']} / {r['arm']} / {r['initial']['train']['true_center_rmsd']:.5f} / {r['final']['train']['true_center_rmsd']:.5f} / {r['initial']['validation']['true_center_rmsd']:.5f} / {r['final']['validation']['true_center_rmsd']:.5f} / {r['learned_eight_protein_point_targets']}")
    lines+=['','主要指标为真实CA块中心按有效残基数加权properKabsch RMSD；不是完整CA/骨架生成。所有固定0/128/512/1024点保留，最终点主报告，不选最优epoch/seed。','回归复用同一网络零状态零时间输出，每蛋白只调用一次，重复的两输出完全确定；spread零是构造结果，不是良好系综。与flow对照比较的是目标/推理过程，更新数相同不等于等FLOPs，不能按点预测准确性就宣布系综生成优越。','几何辅助项两臂均用同一随机t>=0.5门控和权重0.1。两臂每步都消耗相同noise/time抽样，训练对象顺序和抽样hash逐步核对；相同warmstart，新Adam，原fine及结构模型冻结。','两个工程拟合门槛同时满足才算小样本可学：训练平均块中心RMSD<=2Å，且相对该臂初始化降低>=75%。门槛不是泛化/独立确认，也不触发自动加轮数或部署。','8个对象此前参与原899训练，warmstart本来见过它们；本实验是对现有模型在小集合适配的可学性检查，不是从头学习或干净heldout泛化。16开发反复使用，不能用其结果确认方法优越。']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    atomic_json(RUN/'decision.json',dict(status='complete_requires_separate_fit_and_generalization_review',adopt_new_weights=False,
        next='Review training-fit gates and reused-development behavior separately;deterministic regression is diagnostic,not an ensemble replacement.No automatic extra updates or independent-test use.'))
    print('COARSE_SMALLSET_REPORT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','train','report'],required=True);parser.add_argument('--index',type=int,default=0);args=parser.parse_args()
    p=protocol();torch.set_num_threads(p['threads'])
    if args.stage=='smoke':smoke(p)
    elif args.stage=='report':report(p)
    else:train(p,p['seeds'][args.index//len(ARMS)],ARMS[args.index%len(ARMS)])

if __name__=='__main__':main()
