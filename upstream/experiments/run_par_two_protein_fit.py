"""Fixed-budget, training-only two-protein learnability control."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from coarse_generation_diagnostic import weighted_rmsd_matrix
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from par_coarse_backbone import ar_condition, sample_coarse_batch, flow_batch
from par_coarse_transfer import conditioning_features, frozen_versions, assert_frozen
from par_coarse_training_fit import fixed_probe
from par_two_protein_fit import ARMS, sample_keys, update, identity
from run_par_coarse_transfer import LABELS, FeatureBank, model_for, verify


ROOT=Path('artifacts/par_two_protein_fit_v1')
PRIOR=Path('artifacts/par_coarse_training_fit_v1')
PARENT=Path('artifacts/par_coarse_cpu_transfer_v1')


def read(path): return json.loads(Path(path).read_text())


def cases_and_targets():
    previous=read(PRIOR/'protocol.json')
    groups=torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    cases=[];targets=[]
    for protein in ['1b2s_E','1ef1_D']:
        entry=next(r for r in previous['targets'] if r['protein']==protein)
        case=next(c for c in groups['train'][protein] if c['id']==entry['probe_case_ids'][0])
        if case['split']!='train' or len(case['sequence'])!=90: raise RuntimeError('Training pair contract changed')
        cases.append(case)
        targets.append(dict(protein=protein,case_id=case['id'],sequence=case['sequence'],length=90,
                            previous_probe_exposure=entry['probe_exposure']))
    if cases[0]['sequence']==cases[1]['sequence']:raise RuntimeError('Different sequences required')
    return cases,targets


@torch.no_grad()
def denoising(model,cases,bank,arm,seed):
    mode,_=ARMS[arm];records=[]
    for i,case in enumerate(cases):
        b,z=ar_condition(model,conditioning_features(bank(case['sequence']),mode),90)
        for t in [.05,.5,.9]:
            for replicate in [0,1]:
                state,truth,_,coin,digest=fixed_probe(case,seed,100+i,replicate,t)
                state=state[None];truth=truth[None];sc=None
                if coin:
                    v=model.atom_diffusion.nn(flow_batch(b,z,state,t))['coors_pred']
                    sc=state+(1-t)*v
                prediction=model.atom_diffusion.nn(flow_batch(b,z,state,t,sc))['coors_pred']
                value=float((prediction-truth).square().mean())
                if not np.isfinite(value):raise RuntimeError('Nonfinite denoising control')
                records.append(dict(case_id=case['id'],protein=case['protein'],time=t,replicate=replicate,
                    self_condition=coin,draw_sha256=digest,velocity_MSE_nm2=value))
    return records


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise RuntimeError('Preflight must precede the frozen trial')
    verify(read(PRIOR/'protocol.json'))
    cases,targets=cases_and_targets();parent=read(PARENT/'protocol.json')
    model,factory,view=model_for(2026,'cpu');guard=frozen_versions(model)
    bank=FeatureBank(parent['training_feature_index']);outputs=[]
    with torch.no_grad():
        for case in cases:
            outputs.append(sample_coarse_batch(model,bank(case['sequence']),90,sample_keys(2026,2),16))
    torch.testing.assert_close(*outputs,atol=0,rtol=0)
    optimizer=torch.optim.AdamW(view.parameters(),lr=1e-4,weight_decay=1e-4)
    settings=dict(lr=1e-4,min_lr=1e-5,updates=1024,gradient_clip=1.)
    start=time.perf_counter();row=update(model,optimizer,cases,bank,'sequence_uniform',2026,0,settings)
    seconds=time.perf_counter()-start
    assert row['gradient_norm']>0
    assert_frozen(model,guard)
    with torch.no_grad():
        _,a=ar_condition(model,bank(cases[0]['sequence']),90)
        _,b=ar_condition(model,bank(cases[1]['sequence']),90)
        delta=float((a-b).abs().max())
    if delta<=0:raise RuntimeError('No sequence sensitivity after the discarded update')
    refs=[c['coarse'][None].numpy() for c in cases]
    separation=float(weighted_rmsd_matrix(refs[0],refs[1],90)[0,0])
    atomic_json(ROOT/'preflight.json',dict(status='passed',discarded_optimizer_updates=1,actual_trial_updates=0,
        cases=targets,native_coarse_separation_A=separation,length_only_macro_RMSD_lower_bound_A=separation/2,
        initial_same_noise_outputs_exact=True,finite_gradient_norm=row['gradient_norm'],one_update_seconds=seconds,
        sequence_condition_delta_after_discarded_update=delta,official_parameters_unchanged=True,
        trainable_parameters=sum(p.numel() for p in view.parameters()),host=platform.node(),
        development_or_test_coordinates_loaded=False))
    print('TWO_PROTEIN_PREFLIGHT',seconds,separation,flush=True)


def prepare():
    if (ROOT/'protocol.json').exists():raise RuntimeError('Fresh namespace required')
    pre=read(ROOT/'preflight.json')
    if pre['status']!='passed' or not pre['initial_same_noise_outputs_exact']:raise RuntimeError('Preflight required')
    previous=read(PRIOR/'protocol.json');verify(previous)
    cases,targets=cases_and_targets();parent=read(PARENT/'protocol.json')
    p=dict(seed=2026,arms=ARMS,targets=targets,updates=1024,lr=1e-4,min_lr=1e-5,
        weight_decay=1e-4,gradient_clip=1.,progress_every=32,checkpoint_every=128,
        endpoints=[0,256,1024],primary=dict(steps=16,samples=32),diagnostic=dict(steps=64,samples=8,endpoints=[0,1024]),
        training_feature_index=parent['training_feature_index'],
        initialization='Reload exact official117107160-parameter PAR AR+flow backbone for every arm. Fresh464512-parameter AA adapter with same seed and zero final projection; identical initial function for all arms and both same-length proteins. The previous512-update adapter and the discarded preflight update are not reused.',
        dataset='One fixed previously presented MD training frame from each of1b2s_E and1ef1_D,each90residues,23physical block centers. Case IDs are inherited deterministically from the preceding fit diagnostic. Balanced2 examples per protein per update,independent proper rotation and OT/Gaussian draws. A memorization sanity check,not ensemble training.',
        factorial='2x2:AA features available versus zeros;original2%uniform+98%Beta(1.9,1) versus uniform time. Same underlying noise,rotation,time-uniform draw,mixture coin,self-conditioning coin,data order,and LR in all4arms. Changing time distribution is a separate controlled factor,not a silent objective change.',
        training='Unweighted native coarse velocity MSE,AdamW1024updates,only the464512AA adapter parameters trainable. Frozen own fine and codec are not loaded or modified. All4arms use the same CPU allocation,2Torch threads per process.',
        generation='Coarse-only AA/noise inference. Same finite Gaussian keys for BOTH proteins,allarms,endpoints,and solver-prefixes.16stepsx32samples at0/256/1024.64stepsx8fixed-prefix samples at0/1024; compare solver budgets only on the same8samples. No geometric scoring,reference gating,rejection,resampling or checkpoint selection.',
        operational_fit_target='Predeclared sanity target:each training protein has mean proper-aligned block-center RMSD<=2A at final1024,primary16steps/32samples. Report continuous values even when not met. The target is not a statistical significance or ensemble-quality criterion.',
        negative_control='With equal length,zero AA features,and same noise,position-only predictions must be bitwise identical between proteins. Triangle inequality then lower-bounds their macro mean reference RMSD by half the native target separation. Beating this bound with AA features is training-pair conditional control,not generalization.',
        native_coarse_separation_A=pre['native_coarse_separation_A'],
        length_only_macro_RMSD_lower_bound_A=pre['length_only_macro_RMSD_lower_bound_A'],
        failure_policy='Retain all requested outputs. Any nonfinite sample invalidates that whole metric group;no filtering,fallback or selected best step.',
        scope='Exploratory one-seed,two training structures of equal length. No new development samples,independent test/R3,thermodynamic ensemble or overall model-superiority claim. Failure to fit within1024updates does not establish architectural impossibility.',
        stop_rule='Fixed1024updates and declared sampling endpoints for all4arms. No adaptive extension or default promotion. If condition control works,return to multi-frame MD;otherwise inspect conditioning capacity and time/objective limits before scaling.')
    paths=[Path(__file__),Path('upstream/experiments/par_two_protein_fit.py'),Path('upstream/experiments/report_par_two_protein_fit.py'),
        Path('upstream/tests/test_par_two_protein_fit.py'),Path('upstream/scripts/par_two_protein_fit.sbatch'),
        ROOT/'preflight.json',PRIOR/'protocol.json',PRIOR/'review.json',PARENT/'protocol.json']
    p['source_sha256']=dict(previous['source_sha256'])
    p['source_sha256'].update({str(q):sha(q) for q in paths})
    atomic_json(ROOT/'protocol.json',p);print('TWO_PROTEIN_READY',list(ARMS),flush=True)


@torch.no_grad()
def sample_endpoint(p,model,bank,arm,out,endpoint):
    mode,_=ARMS[arm]
    budgets=[p['primary']]+([p['diagnostic']] if endpoint in p['diagnostic']['endpoints'] else [])
    for target in p['targets']:
        h=conditioning_features(bank(target['sequence']),mode)
        for budget in budgets:
            start=time.perf_counter();steps=budget['steps'];count=budget['samples']
            values=sample_coarse_batch(model,h,90,sample_keys(p['seed'],count),steps).numpy()
            folder=out/f'step{endpoint}'/target['protein'];folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'coarse_nfe{steps}.npz',coarse=values,
                valid=np.isfinite(values).all(axis=(1,2)),seeds=np.asarray(sample_keys(p['seed'],count),dtype=np.uint32))
            atomic_json(folder/f'coarse_nfe{steps}.json',dict(seconds=time.perf_counter()-start,steps=steps,samples=count,
                reference_used_for_acceptance=False))
            atomic_json(out/'progress.json',dict(phase='sampling',updates=endpoint,protein=target['protein'],steps=steps))
    model.nn.init_repr_factory.features=None


def train(p,arm,resume):
    cases,targets=cases_and_targets()
    if [r['case_id'] for r in targets]!=[r['case_id'] for r in p['targets']]:raise RuntimeError('Training targets changed')
    model,factory,view=model_for(p['seed'],'cpu');guard=frozen_versions(model)
    optimizer=torch.optim.AdamW(view.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    rng=np.random.default_rng(p['seed']);bank=FeatureBank(p['training_feature_index'])
    out=ROOT/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Arm already complete')
    trace=[];start=0;elapsed=0.;digest=hashlib.sha256();probes={}
    if resume:
        saved=restore_checkpoint(out/'last.pt',view,optimizer,rng,'cpu')
        if saved['protocol_sha256']!=sha(ROOT/'protocol.json') or saved['arm']!=arm:raise RuntimeError('Wrong recovery checkpoint')
        start=saved['updates'];elapsed=saved['training_seconds'];restore_trace_prefix(out,start)
        trace=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
        for row in trace:digest.update(json.dumps(identity(row),sort_keys=True).encode())
        if digest.hexdigest()!=saved['paired_identity_sha256']:raise RuntimeError('Resume trace differs')
        probes=read(out/'native_probes.json')
    else:
        if (out/'trace.jsonl').exists() or (out/'last.pt').exists():raise RuntimeError('Explicit resume required')
        sample_endpoint(p,model,bank,arm,out,0);probes['0']=denoising(model,cases,bank,arm,p['seed'])
        atomic_json(out/'native_probes.json',probes)
    writer=CompactTrainingTrace(p['updates'],start);clock=time.perf_counter()
    for step in range(start,p['updates']):
        row=update(model,optimizer,cases,bank,arm,p['seed'],step,p);trace.append(row)
        digest.update(json.dumps(identity(row),sort_keys=True).encode())
        if (step+1)%p['progress_every']==0:
            writer(out/'trace.json',trace)
            atomic_json(out/'progress.json',dict(phase='training',updates=step+1,total=p['updates'],loss=row['loss']))
        if step+1 in p['endpoints']:
            elapsed+=time.perf_counter()-clock
            sample_endpoint(p,model,bank,arm,out,step+1)
            probes[str(step+1)]=denoising(model,cases,bank,arm,p['seed'])
            atomic_json(out/'native_probes.json',probes);clock=time.perf_counter()
        if (step+1)%p['checkpoint_every']==0:
            assert_frozen(model,guard)
            save_checkpoint(out/'last.pt',view,optimizer,rng,updates=step+1,arm=arm,
                protocol_sha256=sha(ROOT/'protocol.json'),training_seconds=elapsed+time.perf_counter()-clock,
                paired_identity_sha256=digest.hexdigest())
    assert_frozen(model,guard)
    atomic_json(out/'complete.json',dict(status='complete',arm=arm,seed=p['seed'],updates=1024,
        trainable_parameters=sum(v.numel() for v in view.parameters()),official_parameters_unchanged=True,
        paired_identity_sha256=digest.hexdigest(),training_seconds=elapsed+time.perf_counter()-clock,host=platform.node(),
        no_development_or_test_coordinates_loaded=True))
    atomic_json(out/'progress.json',dict(phase='complete',updates=1024))
    print('TWO_PROTEIN_ARM_COMPLETE',arm,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','prepare','train'],required=True)
    parser.add_argument('--arm',choices=list(ARMS));parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='preflight':preflight()
    elif args.stage=='prepare':prepare()
    else:
        if args.arm is None:raise ValueError('Arm required')
        p=read(ROOT/'protocol.json');verify(p);train(p,args.arm,args.resume)


if __name__=='__main__':main()
