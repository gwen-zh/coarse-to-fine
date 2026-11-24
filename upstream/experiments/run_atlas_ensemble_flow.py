"""First native-MD hierarchical density-fit pilot; one sequence, R1 -> R2.

R3 is sealed. Single-sequence success is not learned sequence generalization,
thermodynamic calibration, or a comparison to the author's different benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from atlas_decoder_repair import load_repaired_tokenizer, frozen_structure_state, assert_frozen_structure
from run_atlas_decoder_repair import load_cache, protocol as repair_protocol
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow, snap_fine, component_seed
from joint_sequence_predictor import encode_aa
from mixed_hierarchical_generator import HybridTokenizer
from native_ensemble_metrics import distance_features, train_histogram_edges, compare_ensembles
from compare_two_level_ae_author import metrics
from structural_metrics import aligned_rmsd
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_ensemble_flow_v1')
REPAIR=Path('artifacts/atlas_decoder_repair_v1')


def freeze():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Use a new flow protocol namespace')
    parent=repair_protocol();decision=json.loads((REPAIR/'report.json').read_text())
    if not decision['candidate_accepted_for_MD_pilot']:raise RuntimeError('Geometry repair gate failed; no native-MD flow training')
    repair=Path(decision['repair_checkpoint']);groups=load_cache();train=groups['R1_train']
    if len({c['sequence'] for c in train})!=1:raise RuntimeError('This pilot expects exactly one protein sequence')
    c=torch.cat([r['coarse'] for r in train]);f=torch.cat([r['fine_continuous'] for r in train])
    normalization={name:value.tolist() for name,value in [('coarse_mean',c.mean(0)),('coarse_std',c.std(0,unbiased=False).clamp_min(.05)),('fine_mean',f.mean(0)),('fine_std',f.std(0,unbiased=False).clamp_min(.05))]}
    train_ca=np.stack([r['target'][:,1].numpy() for r in train]);d,r=distance_features(train_ca)
    bins=RUN/'train_histogram_edges.npz';np.savez_compressed(bins,distance=train_histogram_edges(d),rg=train_histogram_edges(r))
    names=['hierarchical_ensemble_flow.py','joint_sequence_predictor.py','atlas_decoder_repair.py','run_atlas_decoder_repair.py','atlas_md_io.py','native_ensemble_metrics.py','mixed_hierarchical_generator.py','hierarchical_rigid_vq.py','hierarchical_latent_contract.py','two_level_autoencoder.py','structural_metrics.py','compare_two_level_ae_author.py','backbone_quality_audit.py','hierarchical_pipeline_io.py','diagnostic_io.py']
    sources=[Path(__file__),Path('upstream/scripts/atlas_ensemble_flow.sbatch'),Path('upstream/tests/test_native_ensemble_metrics.py'),Path('upstream/tests/test_hierarchical_ensemble_flow.py'),REPAIR/'protocol.json',REPAIR/'cache_manifest.json',REPAIR/'labels.pt',REPAIR/'report.json',repair,bins,*map(Path,parent['tokenizer'].values()),*[Path('upstream/experiments')/n for n in names]]
    p=dict(version=1,seeds=[2026,2027],model={'width':128,'layers':4},updates=2048,batch_size=4,lr=3e-4,min_lr=3e-5,weight_decay=1e-4,gradient_clip=1.,coarse_jitter_std=.05,samples=64,coarse_steps=16,fine_steps=16,tokenizer=parent['tokenizer'],repair_checkpoint=str(repair),repair_seed=2026,normalization=normalization,sequence=train[0]['sequence'],train_frames=len(train),development_frames=len(groups['R2_dense']),
        training='R1 paired continuous coarse/fine targets only. Both flow velocity fields share one AA-only Transformer. Fine condition is same-frame native coarse plus small normalized jitter. No R2 training or R3 coordinate loading.',
        evaluation_arms=['full_flow_original_decoder','full_flow_repaired_decoder','oracle_coarse_conditional_fine_repaired_decoder','empirical_R1_repaired_decoder','latent_mean_repaired_decoder'],
        control_scope='Oracle coarse reads R2 reference latent for diagnostics only. Empirical control retrieves paired R1 latent frames and is not a deployable unseen-protein method. Latent mean is a deterministic point baseline.',
        metrics='All raw samples retained. Closest-reference CA RMSD and coverage are posthoc. JS divergence in nats over sequence pairs separated by >=4 residues and Rg, using R1-only bins with overflow. Pairwise spread and aligned RMSF. MD frames are correlated; no iid frame bootstrap.',
        gates={'mean_sample_nearest_reference_vs_empirical_max_extra_A':1.,'CN_vs_native_reconstruction_max_extra':.05,'boundary_CN_vs_native_reconstruction_max_extra':.15,'spread_vs_R2_range':[.5,1.5],'JS_distance_must_not_exceed_latent_mean':True},
        stop_rule='Fixed final 2048 updates per seed; no best-checkpoint selection, temperature sweep, budget extension, test opening, or automatic larger training. This is a one-sequence density-fit/trajectory-transfer pilot, not evidence of AA generalization or physical equilibrium weights.',
        source_sha256={str(x):sha(x) for x in sources})
    atomic_json(RUN/'protocol.json',p);print('NATIVE_MD_FLOW_PROTOCOL_FROZEN',json.dumps({k:p[k] for k in ['seeds','model','updates','batch_size','train_frames','development_frames','samples','repair_checkpoint']}),flush=True)


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed native-MD flow source '+path)
    return p


def decoders(p):
    original=HybridTokenizer.from_checkpoints(**p['tokenizer'])
    repaired=load_repaired_tokenizer(**p['tokenizer'],repair_checkpoint=p['repair_checkpoint'])
    return original,repaired


def make_model(p):return HierarchicalEnsembleFlow(**p['normalization'],**p['model'])


def sample_seed(seed,index):return int(hashlib.sha256(f'atlas-native-flow-v1:{seed}:{index}'.encode()).hexdigest()[:8],16)


@torch.no_grad()
def evaluate(p,model,groups,out):
    model.eval();original,repaired=decoders(p);frozen=frozen_structure_state(repaired)
    aa=encode_aa(p['sequence']);reference=np.stack([r['target'].numpy() for r in groups['R2_dense']]);reference_ca=reference[:,:,1]
    with np.load(RUN/'train_histogram_edges.npz',allow_pickle=False) as z:distance_edges=z['distance'].copy();rg_edges=z['rg'].copy()
    alignment_reference=groups['R1_train'][0]['target'][:,1].numpy();codes=repaired.model.fine_book.codes
    arms={name:[] for name in p['evaluation_arms']};sample_rows=[];oracle_paired=[];oracle_ids=[]
    replay_rng=np.random.default_rng(31337);replay_indices=replay_rng.integers(len(groups['R1_train']),size=p['samples'])
    mean_c=torch.stack([r['coarse'] for r in groups['R1_train']]).mean(0);mean_f=torch.stack([r['fine_continuous'] for r in groups['R1_train']]).mean(0);mean_ids,_=snap_fine(mean_f,codes);mean_bb=repaired.decode(mean_c,mean_ids).numpy()
    native=[]
    for case in groups['R2_dense']:native.append(repaired.decode(case['coarse'],case['fine_ids']).numpy())
    geometry_native=[metrics(bb,ref) for bb,ref in zip(native,reference)]
    started=time.perf_counter()
    for index in range(p['samples']):
        key=sample_seed(int(out.name.replace('seed','')),index)
        c,f=model.sample_latents(aa,key,p['coarse_steps'],p['fine_steps']);ids,error=snap_fine(f,codes)
        before=original.decode(c,ids).numpy();after=repaired.decode(c,ids).numpy()
        arms['full_flow_original_decoder'].append(before);arms['full_flow_repaired_decoder'].append(after)
        condition=groups['R2_dense'][index]
        oracle_f=model.sample_fine(aa,condition['coarse'],component_seed(key,'fine'),p['fine_steps']);oid,oerror=snap_fine(oracle_f,codes);oracle=repaired.decode(condition['coarse'],oid).numpy()
        arms['oracle_coarse_conditional_fine_repaired_decoder'].append(oracle)
        oracle_paired.append(aligned_rmsd(oracle[:,1],reference[index,:,1]));oracle_ids.append(float((oid==condition['fine_ids']).float().mean()))
        case=groups['R1_train'][int(replay_indices[index])]
        arms['empirical_R1_repaired_decoder'].append(repaired.decode(case['coarse'],case['fine_ids']).numpy());arms['latent_mean_repaired_decoder'].append(mean_bb.copy())
        np.savez_compressed(out/f'sample{index}.npz',coarse=c.numpy(),fine_continuous=f.numpy(),fine_ids=ids.numpy(),original_backbone=before,repaired_backbone=after,oracle_coarse_backbone=oracle)
        sample_rows.append(dict(sample=index,seed=key,fine_snap_rmse=float(error.mean()),coarse_outside_native_tanh_range_fraction=float((c.abs()>1.).float().mean()),empirical_R1_index=int(replay_indices[index])))
        if (index+1)%16==0:atomic_json(out/'evaluation_progress.json',dict(samples=index+1,total=p['samples'],seconds=time.perf_counter()-started))
    rows=[]
    for name,backbones in arms.items():
        backbones=np.stack(backbones);stats=compare_ensembles(backbones[:,:,1],reference_ca,distance_edges,rg_edges,alignment_reference)
        geometry=[]
        for bb in backbones:
            closest=int(np.argmin([aligned_rmsd(bb[:,1],ref[:,1]) for ref in reference]));geometry.append(metrics(bb,reference[closest]))
        stats['geometry']={k:float(np.mean([r[k] for r in geometry])) for k in geometry[0]}
        if name.startswith('oracle_'):stats.update(paired_reference_ca_rmsd_A=float(np.mean(oracle_paired)),fine_id_match_to_paired_native=float(np.mean(oracle_ids)))
        rows.append(dict(arm=name,**stats));np.savez_compressed(out/(name+'.npz'),backbone=backbones)
    native_summary={k:float(np.mean([r[k] for r in geometry_native])) for k in geometry_native[0]}
    lookup={r['arm']:r for r in rows};full=lookup['full_flow_repaired_decoder'];empirical=lookup['empirical_R1_repaired_decoder'];point=lookup['latent_mean_repaired_decoder'];native_spread=full['reference_mean_pairwise_ca_rmsd_A'];g=p['gates']
    checks=dict(nearest_reference_fit=full['mean_sample_nearest_reference_ca_rmsd_A']<=empirical['mean_sample_nearest_reference_ca_rmsd_A']+g['mean_sample_nearest_reference_vs_empirical_max_extra_A'],cn=full['geometry']['cn_rate']<=native_summary['cn_rate']+g['CN_vs_native_reconstruction_max_extra'],boundary_cn=full['geometry']['boundary_cn_rate']<=native_summary['boundary_cn_rate']+g['boundary_CN_vs_native_reconstruction_max_extra'],spread=g['spread_vs_R2_range'][0]*native_spread<=full['mean_pairwise_ca_rmsd_A']<=g['spread_vs_R2_range'][1]*native_spread,JS_no_worse_than_point=full['js_pairwise_distance_divergence_nats']<=point['js_pairwise_distance_divergence_nats'])
    assert_frozen_structure(repaired,frozen)
    result=dict(status='complete',arms=rows,native_reconstruction_geometry=native_summary,samples=sample_rows,checks=checks,pass_capacity_gate=all(checks.values()),all_samples_retained=True,sealed_test_coordinates_loaded=False,scope=p['stop_rule'])
    atomic_json(out/'evaluation.json',result);return result


def train(p,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);groups=load_cache();model=make_model(p)
    opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);flow_rng=torch.Generator().manual_seed(seed+22017);out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Fixed native-MD flow pilot already complete')
    start=0;trace=[];digest=sha(RUN/'protocol.json');began=time.perf_counter()
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',model,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=digest:raise RuntimeError('Resume protocol mismatch')
        start=state['updates'];flow_rng.set_state(state['flow_rng_state']);trace=json.loads((out/'trace.json').read_text())[:start]
    for step in range(start,p['updates']):
        model.train();selected=[groups['R1_train'][int(i)] for i in rng.integers(len(groups['R1_train']),size=p['batch_size'])]
        rate=p['min_lr']+.5*(p['lr']-p['min_lr'])*(1+math.cos(math.pi*step/max(1,p['updates']-1)))
        for group in opt.param_groups:group['lr']=rate
        opt.zero_grad(set_to_none=True);total=0.;coarse=0.;fine=0.
        for case in selected:
            loss,terms=model.flow_loss(encode_aa(case['sequence']),case['coarse'],case['fine_continuous'],p['coarse_jitter_std'],flow_rng)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite native-MD flow loss')
            (loss/p['batch_size']).backward();total+=float(loss.detach())/p['batch_size'];coarse+=float(terms['coarse_flow'])/p['batch_size'];fine+=float(terms['fine_flow'])/p['batch_size']
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'])
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite native-MD flow gradient')
        opt.step();trace.append(dict(update=step+1,batch=[c['id'] for c in selected],loss=total,coarse_flow=coarse,fine_flow=fine))
        if (step+1)%64==0 or step+1==p['updates']:
            atomic_json(out/'progress.json',dict(updates=step+1,total=p['updates'],seconds=time.perf_counter()-began,loss=total));atomic_json(out/'trace.json',trace)
        if (step+1)%512==0 or step+1==p['updates']:save_checkpoint(out/'last.pt',model,opt,rng,updates=step+1,flow_rng_state=flow_rng.get_state(),model_config=model.config,protocol_sha256=digest)
    result=evaluate(p,model,groups,out)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,updates=p['updates'],seconds=time.perf_counter()-began,parameters=sum(v.numel() for v in model.parameters()),pass_capacity_gate=result['pass_capacity_gate']))
    print('NATIVE_MD_FLOW_COMPLETE',seed,json.dumps(result['checks']),flush=True)


def report(p):
    rows=[]
    for seed in p['seeds']:
        result=json.loads((RUN/f'seed{seed}'/'evaluation.json').read_text());rows.append(dict(seed=seed,arms=result['arms'],checks=result['checks'],pass_capacity_gate=result['pass_capacity_gate']))
    result=dict(status='complete',seeds=rows,both_seeds_pass_capacity_gate=all(r['pass_capacity_gate'] for r in rows),sealed_test_evaluated=False,scope='One-sequence R1-to-R2 density-fit capacity pilot. No new-sequence generalization, population calibration, or author-model superiority claim. No samples selected or filtered and no best-checkpoint selection.')
    atomic_json(RUN/'report.json',result);print('NATIVE_MD_FLOW_REPORT',json.dumps(dict(both_seeds_pass=result['both_seeds_pass_capacity_gate'],checks=[r['checks'] for r in rows])),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['freeze','train','report']);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='freeze':freeze()
    else:
        p=protocol()
        if args.stage=='train':
            if args.seed not in p['seeds']:raise RuntimeError('Undeclared seed')
            train(p,args.seed)
        else:report(p)
