"""Staged geometry-supervised coarse study; fine and structural networks fixed."""
import argparse,copy,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from mixed_hierarchical_io import sample_seed
from run_coarse_generalization import load as load_original,evaluate as evaluate_original,full_generation as full_original
from coarse_smallset import state_hash
from coarse_geometry_supervision import ARMS,geometry_flow_loss

RUN=Path('artifacts/coarse_geometry_supervision_v1')


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen geometry input changed '+path)
    return p


def load(p,seed,phase):
    q=dict(p);q['train_ids']=p['small_train_ids'] if phase=='small' else p['train_ids']
    return load_original(q,seed)


def update(generator,decoder,item,arm,opt):
    generator.eval();generator.coarse_flow.train();opt.zero_grad(set_to_none=True)
    loss,terms=geometry_flow_loss(generator,decoder,item['features'],item['coarse'],arm)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite geometry loss')
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(generator.coarse_flow.parameters(),1.)
    if not torch.isfinite(norm):raise RuntimeError('Nonfinite geometry gradient')
    opt.step()
    return dict(loss=float(loss.detach()),**{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in terms.items()})


def evaluate(generator,decoder,items,p,seed,arm,phase,directory,initial=False):
    # All arms retain the same stochastic flow sampler. Reuse the audited metric
    # implementation, explicitly replacing its previous experiment's scope.
    r=evaluate_original(generator,decoder,items,p,seed,'flow',directory,initial)
    r.update(objective_arm=arm,phase=phase,scope='WeightedtrueCAblockcenterproperKabsch;8trainingprobes/16reuseddev,separate. Gaussian16stepflow,2draws,structuralmodelandfinefixed. Notfullbackboneorindependentconfirmation.')
    atomic_json(directory/'evaluation.json',r);return r


def gradient_audit(p):
    records=[]
    for seed in p['seeds']:
        generator,decoder,items=load(p,seed,'small');generator.eval()
        frozen=(state_hash(generator),state_hash(decoder));params=list(generator.coarse_flow.parameters())
        train=[x for x in items if x['row']['subset']=='train'][:4]
        for item in train:
            for tau in [.25,.75]:
                torch.manual_seed(sample_seed(seed,item['row']['id'],int(tau*10000)))
                noise=torch.randn_like(generator.normalize(item['coarse'])[None]);t=torch.tensor([tau])
                for arm in ARMS:
                    _,terms=geometry_flow_loss(generator,decoder,item['features'],item['coarse'],arm,noise,t)
                    a=torch.autograd.grad(terms['flow'],params,retain_graph=True,allow_unused=True)
                    b=torch.autograd.grad(terms['weighted_geometry'],params,allow_unused=True)
                    if any(not torch.isfinite(g).all() for g in list(a)+list(b) if g is not None):raise RuntimeError('Nonfinite audited geometry gradient')
                    na=float(sum(g.double().square().sum() for g in a if g is not None).sqrt())
                    nb=float(sum(g.double().square().sum() for g in b if g is not None).sqrt())
                    dot=float(sum((x.double()*y.double()).sum() for x,y in zip(a,b) if x is not None and y is not None))
                    if arm=='shape_geometry' and nb<=0:raise RuntimeError('Shape objective has no gradient to generator')
                    records.append(dict(seed=seed,target=item['row']['id'],t=tau,arm=arm,flow_grad_norm=na,weighted_geometry_grad_norm=nb,
                        geometry_to_flow_norm_ratio=nb/max(na,1e-30),cosine=dot/(na*nb) if na*nb>0 else None))
        if frozen!=(state_hash(generator),state_hash(decoder)):raise RuntimeError('Gradient audit changed weights')
    atomic_json(RUN/'gradient_audit.json',dict(status='passed',targets=records,
        scope='Train-only4probes/seed,t=.25/.75,fullcoarse-parametergradient. No updates,weightscanorqualityclaim. SVDenvelopegradient separately finite-difference tested.'))


def smoke(p):
    seed=p['seeds'][0];rows=[]
    for arm in ARMS:
        torch.manual_seed(seed);generator,decoder,items=load(p,seed,'small');item=next(x for x in items if x['row']['subset']=='train')
        fixed=(state_hash(decoder),state_hash(generator.fine));opt=torch.optim.AdamW(generator.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        for _ in range(3):update(generator,decoder,item,arm,opt)
        path=RUN/f'smoke_{arm}.pt';rng=np.random.default_rng(seed);save_checkpoint(path,generator,opt,rng)
        expected=update(generator,decoder,item,arm,opt);weights=copy.deepcopy(generator.state_dict());adam=copy.deepcopy(opt.state_dict());trng=torch.get_rng_state().clone()
        restored,_,_=load(p,seed,'small');other=torch.optim.AdamW(restored.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        restore_checkpoint(path,restored,other,np.random.default_rng(8),torch.device('cpu'));actual=update(restored,decoder,item,arm,other)
        if actual!=expected or not torch.equal(trng,torch.get_rng_state()):raise RuntimeError('Geometry resume loss/RNG differs')
        for name,v in weights.items():
            if not torch.equal(v,restored.state_dict()[name]):raise RuntimeError('Geometry resume model differs '+name)
        for key,vs in adam['state'].items():
            for name,v in vs.items():
                if not torch.equal(v,other.state_dict()['state'][key][name]):raise RuntimeError('Geometry resume Adam differs')
        if fixed!=(state_hash(decoder),state_hash(generator.fine)):raise RuntimeError('Frozen structure/fine changed')
        rows.append(dict(arm=arm,next_update_model_adam_rng_exact=True,frozen_components_unchanged=True))
    gradient_audit(p)
    atomic_json(RUN/'smoke.json',dict(status='passed',arms=rows,gradient_audit='passed'))
    print('COARSE_GEOMETRY_SMOKE_PASSED',flush=True)


def train(p,seed,arm,phase):
    out=RUN/phase/f'seed{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Geometry arm already complete')
    if json.loads((RUN/'smoke.json').read_text())['status']!='passed':raise RuntimeError('Geometry smoke required')
    if phase=='full' and not json.loads((RUN/'small_report.json').read_text())['ready_for_full']:
        atomic_json(out/'blocked.json',dict(status='blocked_by_predeclared_small_fit_gate',no_training_updates=True));return
    torch.manual_seed(seed);rng=np.random.default_rng(seed);generator,decoder,items=load(p,seed,phase)
    training=[x for x in items if x['row']['subset']=='train'];probes=[x for x in items if x['row']['id'] in p['train_probe_ids']+p['development_ids']]
    fixed=(state_hash(decoder),state_hash(generator.fine));opt=torch.optim.AdamW(generator.coarse_flow.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    schedule=p[phase];first=0;trace=[];order=[];psha=sha(RUN/'protocol.json');started=time.perf_counter()
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',generator,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=psha:raise RuntimeError('Geometry resume protocol changed')
        first=state['updates'];order=state['order'];trace=[x for x in json.loads((out/'trace.json').read_text()) if x['update']<=first]
        if [x['update'] for x in trace]!=list(range(1,first+1)):raise RuntimeError('Incomplete resume trace prefix')
    elif not (out/'step0/evaluation.json').exists():evaluate(generator,decoder,probes,p,seed,arm,phase,out/'step0',True)
    if first in schedule['checkpoints'] and not (out/f'step{first}/evaluation.json').exists():evaluate(generator,decoder,probes,p,seed,arm,phase,out/f'step{first}')
    for step in range(first,schedule['updates']):
        if step%len(training)==0:order=rng.permutation(len(training)).tolist()
        item=training[order[step%len(training)]];torch.manual_seed(sample_seed(seed,item['row']['id'],(50000 if phase=='small' else 70000)+step))
        lr=p['min_lr']+.5*(p['lr']-p['min_lr'])*(1+math.cos(math.pi*step/max(1,schedule['updates']-1)))
        for group in opt.param_groups:group['lr']=lr
        record=update(generator,decoder,item,arm,opt);trace.append(dict(update=step+1,target=item['row']['id'],lr=lr,**record))
        if (step+1)%64==0:atomic_json(out/'progress.json',dict(updates=step+1,total=schedule['updates'],loss=record['loss'],elapsed_seconds=time.perf_counter()-started))
        if step+1 in schedule['checkpoints']:
            atomic_json(out/'trace.json',trace)
            save_checkpoint(out/'last.pt',generator,opt,rng,updates=step+1,order=order,protocol_sha256=psha,seed=seed,arm=arm,phase=phase)
            ev=evaluate(generator,decoder,probes,p,seed,arm,phase,out/f'step{step+1}')
            print('COARSE_GEOMETRY_CHECKPOINT',phase,seed,arm,step+1,json.dumps(ev['groups']),flush=True)
    if fixed!=(state_hash(decoder),state_hash(generator.fine)) or any(x.grad is not None for x in decoder.parameters()):raise RuntimeError('Frozen components changed')
    save_checkpoint(out/'final.pt',generator,opt,rng,updates=schedule['updates'],order=order,protocol_sha256=psha,seed=seed,arm=arm,phase=phase)
    initial=json.loads((out/'step0/evaluation.json').read_text());final=json.loads((out/f"step{schedule['updates']}"/'evaluation.json').read_text())
    result=dict(status='complete',phase=phase,seed=seed,arm=arm,updates=schedule['updates'],training_proteins=len(training),initial=initial['groups'],final=final['groups'],frozen_components_unchanged=True,seconds=time.perf_counter()-started)
    if phase=='small':
        result['train_fit_gate']=final['groups']['train']['true_center_rmsd']<=(1-p['small']['required_training_improvement'])*initial['groups']['train']['true_center_rmsd']
    else:
        # Use the unchanged 16-step fine sampler only for evaluation. No fine
        # training or reference-based acceptance is performed in this phase.
        full=full_original(generator,decoder,probes,p,seed,'flow',out/'full_generation')
        full['scope']='Stochasticgeometrytrainedcoarse+frozenoriginalfine,16reuseddevx8unfilteredoutputs,no referencegeometryselection. Notindependentconfirmationorthermodynamicensemblecoverage.'
        atomic_json(out/'full_generation/evaluation.json',full)
        result.update(full_generation_metrics=full['metrics'],full_spread=full['spread'],coarse_spread=full['coarse_spread'],seconds=time.perf_counter()-started)
    atomic_json(out/'complete.json',result);print('COARSE_GEOMETRY_COMPLETE',phase,seed,arm,flush=True)


def report(p,phase):
    rows=[];blocked=[]
    for seed in p['seeds']:
        signatures=[]
        for arm in ARMS:
            root=RUN/phase/f'seed{seed}'/arm
            if (root/'blocked.json').exists():blocked.append(dict(seed=seed,arm=arm));continue
            rows.append(json.loads((root/'complete.json').read_text()));trace=json.loads((root/'trace.json').read_text())
            signatures.append([(x['target'],x['draw_sha256']) for x in trace])
        if signatures and not all(s==signatures[0] for s in signatures):raise RuntimeError('Unpaired geometry training streams')
    if blocked:
        if len(blocked)!=len(p['seeds'])*len(ARMS):raise RuntimeError('Partial unexpected block')
        atomic_json(RUN/'full_report.json',dict(status='blocked_by_small_shape_fit_gate',arms=blocked,training_updates=0));return
    result=dict(status='complete',rows=rows,paired_training_draws=True,scope=p['scope'])
    if phase=='small':result['ready_for_full']=all(r['train_fit_gate'] for r in rows if r['arm']=='shape_geometry')
    else:
        readiness=[]
        for seed in p['seeds']:
            s=next(r for r in rows if r['seed']==seed and r['arm']=='shape_geometry');baseline=json.loads(Path(p['baseline_generation'][str(seed)]).read_text())
            gate=dict(dev_center_rmsd_below5A=s['final']['validation']['true_center_rmsd']<=5.,
                dev_center_error_reduced10pct=s['final']['validation']['true_center_rmsd']<=.9*s['initial']['validation']['true_center_rmsd'],
                sample_spread_at_least_half_baseline=s['full_spread']>=.5*baseline['spread'])
            readiness.append(dict(seed=seed,gate=gate,pass_gate=all(gate.values())))
        result.update(coarse_readiness=readiness,ready_for_fine=all(r['pass_gate'] for r in readiness),adopt_new_weights=False)
    atomic_json(RUN/f'{phase}_report.json',result)
    lines=[f'Coarse几何监督：{phase}阶段','================================','','seed / arm / train center RMSD / dev center RMSD']
    if phase=='full':lines[3]+=' / fullCA RMSD / CN% / spread'
    for r in rows:
        line=f"{r['seed']} / {r['arm']} / {r['final']['train']['true_center_rmsd']:.5f} / {r['final']['validation']['true_center_rmsd']:.5f}"
        if phase=='full':line+=f" / {r['full_generation_metrics']['ca_rmsd']:.5f} / {100*r['full_generation_metrics']['cn_rate']:.3f} / {r['full_spread']:.5f}"
        lines.append(line)
    lines+=['',p['scope'],'','Smallfit只根据训练8蛋白的25%误差降低决定是否完整训练；不使用开发选点。Fullcoarse是否可进入fine研究另用预设工程条件，非独立确认、非物理系综保证。不按开发最好epoch选权重。']
    (RUN/f'{phase}_results_zh.rst').write_text('\n'.join(lines)+'\n')
    print('COARSE_GEOMETRY_REPORT_COMPLETE',phase,flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['smoke','small','small_report','full','full_report'],required=True);a.add_argument('--index',type=int,default=0);args=a.parse_args();p=protocol();torch.set_num_threads(4)
    if args.stage=='smoke':smoke(p)
    elif args.stage.endswith('_report'):report(p,args.stage.split('_')[0])
    else:train(p,p['seeds'][args.index//len(ARMS)],ARMS[args.index%len(ARMS)],args.stage)

if __name__=='__main__':main()
