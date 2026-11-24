"""Fresh, capacity-matched representation comparison, not generation validation."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from block_frame_tokenizer import BlockFrameFineTokenizer
from block_local_frames import FrameGeometryError
from atlas_decoder_repair import repair_loss
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json

RUN=Path('artifacts/block_frame_tokenizer_v1')


def freeze():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Use a new block-frame experiment namespace')
    parent=Path('artifacts/atlas_shared_frame_flow_v1/protocol.json');q=json.loads(parent.read_text())
    for path,digest in q['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed parent source '+path)
    manifest=Path('artifacts/pdb_training_v2/manifest.json');targets=json.loads(manifest.read_text())['targets']
    training=[r for r in targets if r['subset']=='train'];development=[r for r in targets if r['subset']=='validation']
    development.sort(key=lambda r:hashlib.sha256(('block-frame-development-v1:'+r['id']).encode()).hexdigest());development=development[:32]
    references={k:[dict(id=r['id'],sequence=r['sequence'],path=str(manifest.parent/r['backbone'])) for r in rows] for k,rows in [('train',training),('development',development)]}
    files=[Path(__file__),Path('upstream/experiments/block_frame_tokenizer.py'),Path('upstream/experiments/block_local_frames.py'),Path('upstream/tests/test_block_local_frames.py'),Path('upstream/scripts/block_frame_probe.sbatch'),parent,Path('artifacts/atlas_shared_frame_flow_v1/labels.pt'),manifest,*map(Path,q['tokenizer'].values()),*[Path(r['path']) for rows in references.values() for r in rows],*[Path('upstream/experiments')/name for name in ['hierarchical_rigid_vq.py','hierarchical_latent_contract.py','two_level_autoencoder.py','mixed_hierarchical_generator.py','atlas_decoder_repair.py','compare_two_level_ae_author.py','structural_metrics.py','backbone_quality_audit.py','hierarchical_pipeline_io.py','diagnostic_io.py']]]
    p=dict(version=1,seeds=[2026,2027],arms=['reference','local'],tokenizer=q['tokenizer'],references=references,model={'fine_dim':8,'fine_size':1024,'block_size':4},warmup_updates=1024,quantized_updates=4096,lr=3e-4,min_lr=3e-5,weight_decay=1e-4,gradient_clip=1.,batch={'MD_R1':3,'PDB_train':1},initialization_MD_frames=32,initialization_PDB_proteins=32,kmeans_iterations=10,
        geometry='Frozen reconstructed coarse centers determine tangent frames. Geometry normal is regularized by an off-tangent anchor chosen from the coarse trace, never a fixed global axis. Exactly collinear or fewer than three centers fail explicitly; no silent filtering.',
        matching='Both arms reset fine encoder/decoder and EMA book from identical RNG states. Identical 193604 trainable parameters, fine8D, 1024 book, batch draws, update counts, objectives and initialization cohort. Both decoder inputs include 18 neighbor-frame features; frame convention is the intervention.',
        objective='canonical/shared-reference world-coordinate MSE + CN MSE + boundary CN MSE + .25 inter-residue angle-cosine MSE + .1 CA clash hinge + EMA commitment',
        eligibility='Every frozen train/development case must yield valid frames. A failure stops the pipeline; no geometry/quality-based exclusion.',
        readiness={'MD_CA_max_A':1.,'MD_CN_max':.05,'MD_boundary_CN_max':.15,'PDB_CA_max_A':1.,'PDB_CN_max':.10,'PDB_boundary_CN_max':.25,'train_probe_used_min':64,'train_probe_perplexity_min':32},
        stop_rule='Fixed final quantized endpoint only. No best epoch, budget extension or R3 use. Stop before generator training unless local readiness passes and reconstruction/geometry tradeoffs are reviewed. This is a lower-budget capacity comparison between two fresh fine models, not a fair claim of superiority to the mature old tokenizer.',
        tests_passed=8,source_sha256={str(f):sha(f) for f in files})
    atomic_json(RUN/'protocol.json',p);print('BLOCK_FRAME_PROTOCOL_READY',p['warmup_updates'],p['quantized_updates'],flush=True)


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed block-frame source '+path)
    return p


def make_models(p,seed):
    models={}
    for arm in p['arms']:
        torch.manual_seed(seed);models[arm]=BlockFrameFineTokenizer(arm,**p['model'])
    a,b=[models[x] for x in p['arms']]
    for key,value in a.state_dict().items():
        if not torch.equal(value,b.state_dict()[key]):raise RuntimeError('Unequal initial fine weights')
    return models


@torch.no_grad()
def prepare(p):
    if (RUN/'cache_manifest.json').exists():raise RuntimeError('Block-frame cache already frozen')
    base=HybridTokenizer.from_checkpoints(**p['tokenizer']);source=torch.load('artifacts/atlas_shared_frame_flow_v1/labels.pt',map_location='cpu',weights_only=True)
    models=make_models(p,2026);groups={};failures=[]
    def case(target,c,name):
        centers=base.coarse_centers(c,len(target))
        try:contexts={arm:model.context(centers,len(target)) for arm,model in models.items()}
        except FrameGeometryError as error:failures.append(dict(id=name,reason=str(error)));return None
        return dict(id=name,target=target,coarse=c,contexts=contexts)
    for src,dst in [('R1_train','MD_train'),('R2_dense','MD_development')]:
        groups[dst]=[case(r['target'],r['coarse'],r['id']) for r in source[src]]
    for subset,dst in [('train','PDB_train'),('development','PDB_development')]:
        groups[dst]=[]
        for row in p['references'][subset]:
            with np.load(row['path'],allow_pickle=False) as z:x=torch.from_numpy(z['backbone'].copy()).float()
            c,_,target=base.model.encode_latents(x);groups[dst].append(case(target,c,row['id']))
    atomic_json(RUN/'frame_preflight.json',dict(status='passed' if not failures else 'failed',groups={k:len(v) for k,v in groups.items()},failures=failures,rows_removed=0,R3_loaded=False))
    if failures:raise RuntimeError('Frame preflight failed; no rows may be dropped')
    md_ids=np.linspace(0,len(groups['MD_train'])-1,p['initialization_MD_frames'],dtype=int)
    pdb_ids=sorted(range(len(groups['PDB_train'])),key=lambda i:hashlib.sha256(('block-frame-kmeans-v1:'+groups['PDB_train'][i]['id']).encode()).hexdigest())[:p['initialization_PDB_proteins']]
    groups['train_probe']=[groups['MD_train'][int(i)] for i in md_ids]+[groups['PDB_train'][int(i)] for i in pdb_ids]
    torch.save(groups,RUN/'labels.pt');atomic_json(RUN/'cache_manifest.json',dict(status='complete',cache_sha256=sha(RUN/'labels.pt'),protocol_sha256=sha(RUN/'protocol.json'),groups={k:len(v) for k,v in groups.items()},initialization_MD_indices=md_ids.tolist(),initialization_PDB_indices=pdb_ids,R3_loaded=False))
    print('BLOCK_FRAME_DATA_READY',json.dumps({k:len(v) for k,v in groups.items()}),flush=True)


def load_cache():
    m=json.loads((RUN/'cache_manifest.json').read_text())
    if m['cache_sha256']!=sha(RUN/'labels.pt') or m['protocol_sha256']!=sha(RUN/'protocol.json'):raise RuntimeError('Changed block-frame cache')
    return torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True)


def code_usage(counts):
    counts=np.asarray(counts,float);probs=counts[counts>0]/counts.sum()
    return dict(used=int((counts>0).sum()),vocabulary=len(counts),perplexity=float(np.exp(-(probs*np.log(probs)).sum())))


@torch.no_grad()
def evaluate(model,groups,quantized):
    model.eval();rows=[];summaries={};drift=0.
    for group in ['MD_development','PDB_development','train_probe']:
        records=[];hist=np.zeros(model.book.size,dtype=np.int64);continuous=[]
        for case in groups[group]:
            context=case['contexts'][model.mode];out=model(case['target'],context,quantized=quantized);bb=out['backbone'];record=dict(group=group,id=case['id'],metrics=metrics(bb.numpy(),case['target'].numpy()));records.append(record);rows.append(record)
            from hierarchical_latent_contract import pool_ca
            drift=max(drift,float((pool_ca(bb)-context['centers']).abs().max()))
            if quantized:
                hist+=np.bincount(out['fine_ids'].numpy(),minlength=model.book.size)
                pred=model.decode_vectors(out['fine_continuous'],context,len(case['target']));continuous.append(metrics(pred.numpy(),case['target'].numpy()))
        summary=dict(cases=len(records),metrics={key:float(np.mean([r['metrics'][key] for r in records])) for key in records[0]['metrics']})
        if quantized:summary.update(codebook=code_usage(hist),continuous_fine_metrics={key:float(np.mean([r[key] for r in continuous])) for key in continuous[0]})
        summaries[group]=summary
    return dict(groups=summaries,rows=rows,max_centroid_error_A=drift,book_restarts=int(model.book.restarts))


def train(p,arm,seed):
    groups=load_cache();model=make_models(p,seed)[arm];opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=np.random.default_rng(seed)
    # The model constructor resets the Torch RNG identically in both modes.
    out=RUN/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('This fixed-budget arm is already complete')
    start=0;trace=[];total=p['warmup_updates']+p['quantized_updates'];digest=sha(RUN/'protocol.json');began=time.perf_counter()
    if (out/'last.pt').exists():
        saved=restore_checkpoint(out/'last.pt',model,opt,rng,torch.device('cpu'))
        if saved['protocol_sha256']!=digest:raise RuntimeError('Resume protocol differs')
        start=saved['updates'];trace=json.loads((out/'trace.json').read_text())[:start]
    for step in range(start,total):
        quantized=step>=p['warmup_updates']
        if quantized and not model.book.initialized:
            model.eval();vectors=[]
            with torch.no_grad():
                for case in groups['train_probe']:vectors.append(model.encode(case['target'],case['contexts'][arm]))
            model.book.initialize(torch.cat(vectors),p['kmeans_iterations']);print('BLOCK_FRAME_BOOK_INITIALIZED',arm,seed,flush=True)
        model.train();phase_step=step-p['warmup_updates'] if quantized else step;phase_total=p['quantized_updates'] if quantized else p['warmup_updates'];lr=p['min_lr']+.5*(p['lr']-p['min_lr'])*(1+math.cos(math.pi*phase_step/max(1,phase_total-1)))
        for group in opt.param_groups:group['lr']=lr
        selected=[groups['MD_train'][int(i)] for i in rng.integers(len(groups['MD_train']),size=p['batch']['MD_R1'])]
        selected.extend(groups['PDB_train'][int(i)] for i in rng.integers(len(groups['PDB_train']),size=p['batch']['PDB_train']))
        opt.zero_grad(set_to_none=True);total_loss=0.;terms_mean={}
        for case in selected:
            output=model(case['target'],case['contexts'][arm],quantized=quantized);loss,terms=repair_loss(output['backbone'],case['target'],geometry=True);loss=loss+output['commitment']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fine-tokenizer loss')
            (loss/len(selected)).backward();total_loss+=float(loss.detach())/len(selected)
            for key,value in terms.items():terms_mean[key]=terms_mean.get(key,0.)+float(value)/len(selected)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'])
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite fine-tokenizer gradient')
        opt.step();trace.append(dict(update=step+1,quantized=quantized,batch=[r['id'] for r in selected],loss=total_loss,**terms_mean))
        if (step+1)%64==0 or step+1==total:
            atomic_json(out/'progress.json',dict(updates=step+1,total=total,quantized=quantized,seconds=time.perf_counter()-began,loss=total_loss));atomic_json(out/'trace.json',trace)
        if (step+1)%512==0 or step+1==total:save_checkpoint(out/'last.pt',model,opt,rng,updates=step+1,model_config=model.config,arm=arm,seed=seed,protocol_sha256=digest)
        if step+1==p['warmup_updates']:atomic_json(out/'warmup_evaluation.json',evaluate(model,groups,False))
    evaluation=evaluate(model,groups,True);atomic_json(out/'evaluation.json',evaluation)
    atomic_json(out/'complete.json',dict(status='complete',arm=arm,seed=seed,updates=total,seconds=time.perf_counter()-began,trainable_parameters=sum(x.numel() for x in model.parameters()),coarse_weights_updated=False,R3_loaded=False))
    print('BLOCK_FRAME_TRAIN_COMPLETE',arm,seed,json.dumps({k:v['metrics']['ca_rmsd'] for k,v in evaluation['groups'].items()}),flush=True)


def readiness(p,e):
    g=p['readiness'];checks={}
    for group,prefix in [('MD_development','MD'),('PDB_development','PDB')]:
        m=e['groups'][group]['metrics'];checks[group]=dict(CA=m['ca_rmsd']<=g[prefix+'_CA_max_A'],CN=m['cn_rate']<=g[prefix+'_CN_max'],boundary_CN=m['boundary_cn_rate']<=g[prefix+'_boundary_CN_max'])
    book=e['groups']['train_probe']['codebook'];checks['book']=dict(used=book['used']>=g['train_probe_used_min'],perplexity=book['perplexity']>=g['train_probe_perplexity_min']);checks['contract']=dict(centroid=e['max_centroid_error_A']<=3e-5)
    return checks,all(all(x.values()) for x in checks.values())


def report(p):
    rows=[]
    for arm in p['arms']:
        for seed in p['seeds']:
            out=RUN/arm/f'seed{seed}';complete=json.loads((out/'complete.json').read_text());e=json.loads((out/'evaluation.json').read_text());checks,passed=readiness(p,e)
            rows.append(dict(arm=arm,seed=seed,pass_readiness=passed,checks=checks,evaluation=e,complete=complete))
    for seed in p['seeds']:
        traces=[json.loads((RUN/arm/f'seed{seed}/trace.json').read_text()) for arm in p['arms']]
        if [r['batch'] for r in traces[0]]!=[r['batch'] for r in traces[1]]:raise RuntimeError('Data minibatches were not matched')
    comparison=[]
    for seed in p['seeds']:
        a=next(r for r in rows if r['arm']=='reference' and r['seed']==seed);b=next(r for r in rows if r['arm']=='local' and r['seed']==seed)
        comparison.append(dict(seed=seed,reference=a['evaluation']['groups'],local=b['evaluation']['groups']))
    result=dict(status='complete',arms=rows,comparison=comparison,minibatches_matched=True,identical_parameter_counts=len({r['complete']['trainable_parameters'] for r in rows})==1,local_ready_both_seeds=all(r['pass_readiness'] for r in rows if r['arm']=='local'),R3_used=False,generator_training_started=False,scope=p['stop_rule'])
    atomic_json(RUN/'report.json',result);print('BLOCK_FRAME_REPORT',json.dumps(dict(local_ready_both_seeds=result['local_ready_both_seeds'],results=[dict(arm=r['arm'],seed=r['seed'],ready=r['pass_readiness'],MD_CA=r['evaluation']['groups']['MD_development']['metrics']['ca_rmsd'],PDB_CA=r['evaluation']['groups']['PDB_development']['metrics']['ca_rmsd']) for r in rows])),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['freeze','prepare','train','report']);a.add_argument('--arm',choices=['reference','local']);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='freeze':freeze()
    else:
        p=protocol()
        if args.stage=='prepare':prepare(p)
        elif args.stage=='train':
            if args.arm not in p['arms'] or args.seed not in p['seeds']:raise RuntimeError('Unregistered arm')
            train(p,args.arm,args.seed)
        else:report(p)
