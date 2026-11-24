"""Fixed-budget decoder repair on ATLAS R1, developed on R2; R3 stays sealed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import mdtraj as md
import numpy as np
import torch
from atlas_md_io import backbone_indices, backbone_angstrom, diagnostic_trajectories, uniform_frame_indices
from atlas_decoder_repair import enable_fine_decoder_training, frozen_structure_state, assert_frozen_structure, repair_loss
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_rigid_vq import canonical_backbone
from compare_two_level_ae_author import metrics
from structural_metrics import aligned_rmsd
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_decoder_repair_v1')
MD=Path('artifacts/atlas_md_pilot_v1')


def freeze():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Use a new protocol namespace')
    parent=json.loads((MD/'reconstruction_protocol.json').read_text())
    for path,digest in parent['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed parent source '+path)
    manifest=Path('artifacts/pdb_training_v2/manifest.json')
    targets=json.loads(manifest.read_text())['targets']
    replay={}
    for subset,count in [('train',32),('validation',16)]:
        eligible=[r for r in targets if r['subset']==subset and 64<=r['length']<=192]
        eligible.sort(key=lambda r:hashlib.sha256(('atlas-decoder-replay-v1:'+r['id']).encode()).hexdigest())
        if len(eligible)<count:raise RuntimeError('Insufficient predeclared replay cohort')
        replay[subset]=[dict(id=r['id'],sequence=r['sequence'],path=str(manifest.parent/r['backbone'])) for r in eligible[:count]]
    sources=[Path(__file__),Path('upstream/experiments/atlas_decoder_repair.py'),Path('upstream/experiments/atlas_md_io.py'),Path('upstream/tests/test_atlas_decoder_repair.py'),Path('upstream/scripts/atlas_decoder_repair.sbatch'),MD/'selection.json',MD/'download_manifest.json',MD/'reconstruction_protocol.json',manifest,*map(Path,parent['tokenizer'].values()),*[Path(r['path']) for rows in replay.values() for r in rows],*[MD/f'reconstruction/{r}_diagnostic.npz' for r in ['R1','R2']],*[Path('upstream/experiments')/name for name in ['mixed_hierarchical_generator.py','hierarchical_rigid_vq.py','hierarchical_latent_contract.py','two_level_autoencoder.py','structural_metrics.py','compare_two_level_ae_author.py','backbone_quality_audit.py','diagnostic_io.py','hierarchical_pipeline_io.py']]]
    p=dict(version=1,seeds=[2026,2027],arms=['coordinate','geometry'],candidate_arm='geometry',export_seed=2026,tokenizer=parent['tokenizer'],replay=replay,
        updates=2048,lr=1e-4,min_lr=1e-5,weight_decay=1e-4,gradient_clip=1.,train_batch={'R1_frames':3,'pdb_replay':1},
        data={'R1_stride':10,'R2_dense_frames':64,'old_MD_diagnostic_frames':16,'R3':'sealed_no_coordinate_loading'},
        objectives={'coordinate':'mean squared canonical N/CA/C coordinate error in Angstrom^2','geometry':'coordinate + mean CN squared length error + mean boundary CN squared length error + 0.25 inter-residue angle-cosine MSE + 0.1 CA clash hinge squared'},
        trainable='Only the existing 104484-parameter fine_decoder MLP. Encoders, both books and coarse decoder unchanged; rigid residue and block centroid contracts unchanged.',
        selection_rule='Fixed final update only; the geometry arm is the only candidate. Both seeds must pass; seed2026 chosen in advance for export. Coordinate arm is an attribution control, not a candidate tournament.',
        gates={'MD_mean_ca_max_A':1.,'MD_mean_cn_max':.05,'MD_mean_boundary_cn_max':.15,'MD_ca_increase_vs_baseline_max_A':.05,'PDB_validation_ca_increase_max_A':.05,'PDB_validation_cn_increase_max':.01,'R2_dense_spread_error_increase_max_A':.05},
        scope='Single-sequence engineering/development repair; reused R2 and old PDB development examples are not independent tests. No MD flow training unless both geometry-arm seeds pass. No checkpoint sweep or automatic budget extension.',
        source_sha256={str(x):sha(x) for x in sources})
    atomic_json(RUN/'protocol.json',p)
    print('DECODER_REPAIR_PROTOCOL_FROZEN',json.dumps({k:p[k] for k in ['updates','arms','seeds','trainable','gates']}),flush=True)


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed frozen repair source '+path)
    return p


def case_from_backbone(decoder,bb,name,sequence):
    bb=torch.as_tensor(np.array(bb,copy=True),dtype=torch.float32)
    with torch.no_grad():c,f,target=decoder.model.encode_latents(bb);ids=decoder.model.fine_book.nearest(f)
    if len(sequence)!=len(bb):raise RuntimeError('AA/backbone length mismatch')
    return dict(id=name,sequence=sequence,coarse=c.cpu(),fine_continuous=f.cpu(),fine_ids=ids.cpu(),target=target.cpu())


def load_cache():
    m=json.loads((RUN/'cache_manifest.json').read_text())
    if m['protocol_sha256']!=sha(RUN/'protocol.json') or m['cache_sha256']!=sha(RUN/'labels.pt'):raise RuntimeError('Changed decoder-repair cache')
    return torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True)


@torch.no_grad()
def evaluate(decoder,groups):
    decoder.eval();rows=[];summaries={}
    for group in ['R1_original','R2_original','R2_dense','pdb_validation']:
        predictions=[];references=[];records=[]
        for case in groups[group]:
            pred=decoder.decode(case['coarse'],case['fine_ids']).cpu().numpy();target=case['target'].numpy()
            row=dict(group=group,id=case['id'],metrics=metrics(pred,target));rows.append(row);records.append(row)
            predictions.append(pred[:,1]);references.append(target[:,1])
        summary=dict(cases=len(records),metrics={k:float(np.mean([r['metrics'][k] for r in records])) for k in records[0]['metrics']})
        if group=='R2_dense':
            pairs=list(zip(*np.triu_indices(len(predictions),1)))
            # Every pair, no geometric or frame selection.
            summary['mean_pairwise_ca_rmsd_A']=float(np.mean([aligned_rmsd(predictions[i],predictions[j]) for i,j in pairs]))
            summary['reference_mean_pairwise_ca_rmsd_A']=float(np.mean([aligned_rmsd(references[i],references[j]) for i,j in pairs]))
        summaries[group]=summary
    return dict(groups=summaries,rows=rows)


def cache(p):
    if (RUN/'cache_manifest.json').exists():raise RuntimeError('Cache already frozen')
    decoder=HybridTokenizer.from_checkpoints(**p['tokenizer'])
    selection=json.loads((MD/'selection.json').read_text());sequence=selection['selected']['sequence']
    download=json.loads((MD/'download_manifest.json').read_text())
    for row in download['extracted']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('Changed MD source '+row['path'])
    pdbs=list((MD/'raw').rglob('*.pdb'))
    if len(pdbs)!=1:raise RuntimeError('One explicit topology required')
    top=md.load_topology(str(pdbs[0]));indices=backbone_indices(top,sequence)
    trajectories=diagnostic_trajectories(MD/'raw')
    groups={};frame_indices={};started=time.perf_counter()
    for replicate in ['R1','R2']:
        stride=p['data']['R1_stride'] if replicate=='R1' else 1
        traj=md.load(str(trajectories[replicate]),top=top,stride=stride)
        selected=np.arange(len(traj)) if replicate=='R1' else uniform_frame_indices(len(traj),p['data']['R2_dense_frames'])
        values=backbone_angstrom(traj.xyz[selected],indices)
        cn=np.linalg.norm(values[:,:-1,2]-values[:,1:,0],axis=-1)
        if ((cn<1.)|(cn>1.8)).mean(1).max()>.01:raise RuntimeError('Wrapped or nonphysical native MD backbone')
        key='R1_train' if replicate=='R1' else 'R2_dense'
        groups[key]=[case_from_backbone(decoder,bb,f'{replicate}:frame{int(i*stride)}',sequence) for bb,i in zip(values,selected)]
        frame_indices[key]=(selected*stride).tolist()
        print('MD_LABELS_READY',key,len(groups[key]),flush=True)
        del traj,values
    for replicate in ['R1','R2']:
        with np.load(MD/f'reconstruction/{replicate}_diagnostic.npz',allow_pickle=False) as z:
            groups[replicate+'_original']=[dict(id=f'{replicate}:frame{int(index)}',sequence=sequence,coarse=torch.from_numpy(c.copy()),fine_continuous=torch.from_numpy(f.copy()),fine_ids=torch.from_numpy(ids.copy()),target=canonical_backbone(torch.from_numpy(bb.copy()))) for bb,c,f,ids,index in zip(z['backbone'],z['coarse'],z['fine_continuous'],z['fine_ids'],z['frame_indices'])]
    for subset,key in [('train','pdb_replay'),('validation','pdb_validation')]:
        groups[key]=[]
        for row in p['replay'][subset]:
            with np.load(row['path'],allow_pickle=False) as z:bb=z['backbone'].copy()
            groups[key].append(case_from_backbone(decoder,bb,row['id'],row['sequence']))
    torch.save(groups,RUN/'labels.pt')
    atomic_json(RUN/'baseline.json',evaluate(decoder,groups))
    atomic_json(RUN/'cache_manifest.json',dict(status='complete',protocol_sha256=sha(RUN/'protocol.json'),cache_sha256=sha(RUN/'labels.pt'),baseline_sha256=sha(RUN/'baseline.json'),groups={k:len(v) for k,v in groups.items()},frame_indices=frame_indices,seconds=time.perf_counter()-started,sealed_test_coordinates_loaded=False))
    print('DECODER_REPAIR_CACHE_COMPLETE',json.dumps({k:len(v) for k,v in groups.items()}),flush=True)


def train(p,arm,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);groups=load_cache()
    decoder=HybridTokenizer.from_checkpoints(**p['tokenizer']);frozen=frozen_structure_state(decoder)
    first=groups['R1_train'][0]
    with torch.no_grad():epoch0=decoder.decode(first['coarse'],first['fine_ids']).clone()
    params=enable_fine_decoder_training(decoder)
    with torch.no_grad():torch.testing.assert_close(epoch0,decoder.decode(first['coarse'],first['fine_ids']),atol=0,rtol=0)
    opt=torch.optim.AdamW(params,lr=p['lr'],weight_decay=p['weight_decay'])
    out=RUN/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Fixed-budget repair already completed')
    start=0;trace=[];digest=sha(RUN/'protocol.json');began=time.perf_counter()
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=digest:raise RuntimeError('Resume repair protocol changed')
        decoder.model.fine_decoder.load_state_dict(state['fine_decoder']);opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['numpy_rng'];start=state['updates'];trace=json.loads((out/'trace.json').read_text())[:start]
    for step in range(start,p['updates']):
        selected=[groups['R1_train'][int(i)] for i in rng.integers(len(groups['R1_train']),size=3)]
        selected.append(groups['pdb_replay'][int(rng.integers(len(groups['pdb_replay'])))])
        lr=p['min_lr']+.5*(p['lr']-p['min_lr'])*(1+math.cos(math.pi*step/max(1,p['updates']-1)))
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);loss_sum=0.;terms_sum={}
        for case in selected:
            pred=decoder.decode(case['coarse'],case['fine_ids']);loss,terms=repair_loss(pred,case['target'],geometry=arm=='geometry')
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite repair loss')
            (loss/len(selected)).backward();loss_sum+=float(loss.detach())/len(selected)
            for k,v in terms.items():terms_sum[k]=terms_sum.get(k,0.)+float(v)/len(selected)
        grad=torch.nn.utils.clip_grad_norm_(params,p['gradient_clip'])
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite repair gradient')
        opt.step();trace.append(dict(update=step+1,batch=[c['id'] for c in selected],loss=loss_sum,**terms_sum))
        if (step+1)%64==0 or step+1==p['updates']:
            atomic_json(out/'progress.json',dict(updates=step+1,total=p['updates'],seconds=time.perf_counter()-began,loss=loss_sum));atomic_json(out/'trace.json',trace)
        if (step+1)%512==0 or step+1==p['updates']:
            assert_frozen_structure(decoder,frozen)
            tmp=out/'last.tmp.pt';torch.save(dict(fine_decoder=decoder.model.fine_decoder.state_dict(),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,updates=step+1,arm=arm,seed=seed,protocol_sha256=digest,parent_checkpoint_sha256={path:sha(path) for path in p['tokenizer'].values()}),tmp);tmp.replace(out/'last.pt')
    assert_frozen_structure(decoder,frozen);result=evaluate(decoder,groups)
    atomic_json(out/'evaluation.json',result)
    atomic_json(out/'complete.json',dict(status='complete',arm=arm,seed=seed,updates=p['updates'],seconds=time.perf_counter()-began,trainable_parameters=sum(v.numel() for v in params),frozen_structure_unchanged=True,epoch0_exact=True))
    print('DECODER_REPAIR_COMPLETE',arm,seed,json.dumps({k:v['metrics']['cn_rate'] for k,v in result['groups'].items()}),flush=True)


def gates(p,result,baseline):
    checks={};g=p['gates']
    for name in ['R1_original','R2_original','R2_dense']:
        m=result['groups'][name]['metrics'];b=baseline['groups'][name]['metrics']
        checks[name]=dict(ca=m['ca_rmsd']<=g['MD_mean_ca_max_A'],cn=m['cn_rate']<=g['MD_mean_cn_max'],boundary_cn=m['boundary_cn_rate']<=g['MD_mean_boundary_cn_max'],ca_not_degraded=m['ca_rmsd']<=b['ca_rmsd']+g['MD_ca_increase_vs_baseline_max_A'])
    m=result['groups']['pdb_validation']['metrics'];b=baseline['groups']['pdb_validation']['metrics']
    checks['pdb_validation']=dict(ca=m['ca_rmsd']<=b['ca_rmsd']+g['PDB_validation_ca_increase_max_A'],cn=m['cn_rate']<=b['cn_rate']+g['PDB_validation_cn_increase_max'])
    a=result['groups']['R2_dense'];b=baseline['groups']['R2_dense'];native=b['reference_mean_pairwise_ca_rmsd_A']
    checks['R2_dense_spread']=dict(error_not_degraded=abs(a['mean_pairwise_ca_rmsd_A']-native)<=abs(b['mean_pairwise_ca_rmsd_A']-native)+g['R2_dense_spread_error_increase_max_A'])
    return checks,all(all(v.values()) for v in checks.values())


def report(p):
    cache=load_cache();baseline=json.loads((RUN/'baseline.json').read_text());rows=[]
    for arm in p['arms']:
        for seed in p['seeds']:
            out=RUN/arm/f'seed{seed}'
            complete=json.loads((out/'complete.json').read_text());evaluation=json.loads((out/'evaluation.json').read_text());checks,passed=gates(p,evaluation,baseline)
            rows.append(dict(arm=arm,seed=seed,pass_gate=passed,checks=checks,groups=evaluation['groups'],complete=complete))
    # Confirm the data minibatches were paired across objective arms.
    for seed in p['seeds']:
        traces=[json.loads((RUN/arm/f'seed{seed}'/'trace.json').read_text()) for arm in p['arms']]
        if len(traces[0])!=p['updates'] or [x['batch'] for x in traces[0]]!=[x['batch'] for x in traces[1]]:raise RuntimeError('Decoder repair controls were not paired')
    candidate=all(r['pass_gate'] for r in rows if r['arm']==p['candidate_arm'])
    export=str(RUN/p['candidate_arm']/f'seed{p["export_seed"]}'/'last.pt') if candidate else None
    result=dict(status='complete',baseline=baseline['groups'],arms=rows,paired_minibatches=True,candidate_accepted_for_MD_pilot=candidate,repair_checkpoint=export,export_seed_fixed_in_advance=p['export_seed'],sealed_test_evaluated=False,scope='Engineering/development gate only, not independent protein validation. Coordinate control is not an alternative selected candidate.',next='Prepare a fixed-budget native-MD hierarchical flow pilot using the accepted repair.' if candidate else 'Stop before MD generator fitting; diagnose the failed repair gates without relaxing them.')
    atomic_json(RUN/'report.json',result)
    print('DECODER_REPAIR_REPORT',json.dumps(dict(accepted=candidate,checkpoint=export,arms=[dict(arm=r['arm'],seed=r['seed'],pass_gate=r['pass_gate'],R2=r['groups']['R2_dense']) for r in rows])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','cache','train','report'],required=True);parser.add_argument('--arm',choices=['coordinate','geometry']);parser.add_argument('--seed',type=int,default=2026);args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='freeze':freeze()
    else:
        p=protocol()
        if args.stage=='cache':cache(p)
        elif args.stage=='train':
            if args.arm not in p['arms'] or args.seed not in p['seeds']:raise RuntimeError('Invalid frozen repair arm')
            train(p,args.arm,args.seed)
        else:report(p)
