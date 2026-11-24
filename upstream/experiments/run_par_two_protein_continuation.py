"""Separate bounded continuation of the completed1024-update memorization trial."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace,restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha,restore_checkpoint,save_checkpoint
from par_coarse_backbone import sample_coarse_batch
from par_coarse_transfer import frozen_versions,assert_frozen
from par_two_protein_fit import sample_keys
from par_two_protein_continuation import ARMS,INITIAL_UPDATES,ADDITIONAL_UPDATES,TOTAL_UPDATES,learning_rate,update,common_identity
from run_par_coarse_transfer import FeatureBank,model_for,verify
from run_par_two_protein_fit import ROOT as PARENT,read,cases_and_targets,denoising,sample_endpoint


ROOT=Path('artifacts/par_two_protein_continuation_v1')


def optimizer_digest(state):
    digest=hashlib.sha256(json.dumps(state['param_groups'],sort_keys=True).encode())
    for index,values in sorted(state['state'].items()):
        digest.update(str(index).encode())
        for key,value in sorted(values.items()):
            digest.update(key.encode())
            if torch.is_tensor(value):
                digest.update(str(value.dtype).encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:digest.update(repr(value).encode())
    return digest.hexdigest()


def warm_model(parent):
    previous=read(PARENT/'protocol.json');model,factory,view=model_for(previous['seed'],'cpu')
    optimizer=torch.optim.AdamW(view.parameters(),lr=previous['lr'],weight_decay=previous['weight_decay'])
    rng=np.random.default_rng(previous['seed'])
    saved=restore_checkpoint(PARENT/parent/'last.pt',view,optimizer,rng,'cpu')
    if saved['updates']!=1024 or saved['arm']!=parent or saved['protocol_sha256']!=sha(PARENT/'protocol.json'):
        raise RuntimeError('Wrong parent continuation checkpoint')
    if any(abs(float(v['step'])-1024)>1e-7 for v in optimizer.state.values()) or not optimizer.state:
        raise RuntimeError('Parent Adam moments are not at1024')
    if any(group['lr']!=1e-5 for group in optimizer.param_groups):raise RuntimeError('Unexpected final parent LR')
    restored=optimizer_digest(optimizer.state_dict())
    if restored!=optimizer_digest(saved['optimizer']):raise RuntimeError('Optimizer state was not preserved')
    return model,view,optimizer,rng,saved,restored


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise RuntimeError('Preflight precedes the frozen continuation')
    previous=read(PARENT/'protocol.json');verify(previous)
    cases,targets=cases_and_targets();bank=FeatureBank(previous['training_feature_index']);records=[]
    for parent,trial in [('sequence_beta','beta_floor'),('sequence_uniform','uniform_restart')]:
        model,view,optimizer,rng,saved,moment_hash=warm_model(parent);guard=frozen_versions(model)
        maxima=[]
        with torch.no_grad():
            for target in targets:
                generated=sample_coarse_batch(model,bank(target['sequence']),90,sample_keys(previous['seed'],4),16).numpy()
                with np.load(PARENT/parent/f"step1024/{target['protein']}/coarse_nfe16.npz") as z:expected=z['coarse'][:4].copy()
                np.testing.assert_allclose(generated,expected,atol=1e-3,rtol=1e-5)
                maxima.append(float(np.max(np.abs(generated-expected))))
        old_probe=read(PARENT/parent/'native_probes.json')['1024']
        replay=denoising(model,cases,bank,parent,previous['seed'])
        np.testing.assert_allclose([r['velocity_MSE_nm2'] for r in replay],[r['velocity_MSE_nm2'] for r in old_probe],atol=1e-6,rtol=1e-5)
        start=time.perf_counter();row=update(model,optimizer,cases,bank,trial,previous['seed'],0,dict(gradient_clip=1.))
        seconds=time.perf_counter()-start
        if row['global_update']!=1025 or row['update']!=1 or row['lr']!=learning_rate(0,ARMS[trial][1]):
            raise RuntimeError('Wrong continuation step or LR')
        if row['gradient_norm']<=0:raise RuntimeError('Missing warm-start gradient')
        assert_frozen(model,guard)
        records.append(dict(parent=parent,trial=trial,replay_max_abs_A=max(maxima),optimizer_state_sha256=moment_hash,
            optimizer_moments_preserved=True,discarded_update_seconds=seconds,discarded_update_lr=row['lr'],
            finite_gradient_norm=row['gradient_norm'],first_global_update=1025,official_weights_unchanged=True))
        del model,view,optimizer,saved
    atomic_json(ROOT/'preflight.json',dict(status='passed',parents=records,discarded_optimizer_updates=2,
        actual_continuation_updates=0,coarse_replay_atol_A=1e-3,coarse_replay_rtol=1e-5,
        native_probe_replayed=True,host=platform.node(),new_development_or_test_coordinates_loaded=False))
    print('CONTINUATION_PREFLIGHT',json.dumps(records),flush=True)


def prepare():
    if (ROOT/'protocol.json').exists():raise RuntimeError('Fresh continuation namespace required')
    pre=read(ROOT/'preflight.json')
    if pre['status']!='passed':raise RuntimeError('Parent replay preflight required')
    previous=read(PARENT/'protocol.json');verify(previous)
    p=dict(seed=previous['seed'],arms=ARMS,targets=previous['targets'],parent_updates=INITIAL_UPDATES,
        additional_updates=ADDITIONAL_UPDATES,total_updates=TOTAL_UPDATES,gradient_clip=1.,weight_decay=1e-4,
        progress_every=32,checkpoint_every=128,endpoints=[1024,2048,4096],primary=previous['primary'],
        diagnostic=dict(steps=64,samples=8,endpoints=[1024,4096]),training_feature_index=previous['training_feature_index'],
        initialization='Restore each AA-conditioned parent1024-update adapter,AdamW moments,parameter groups and RNG. Floor/restart variants of the same time rule start from the SAME parent. Reload the same frozen117107160-parameter official PAR backbone;464512 trainable adapter parameters. Discard the two replay-preflight updates.',
        lr_factor='Floor policy:constant1e-5 for3072additional updates. Restart policy:1e-4->1e-5 cosine over exactly3072additional updates. Adam moments are retained in BOTH policies;the restart refers only to LR. No undocumented optimizer reset.',
        time_factor='Retain the parent beta-mixture or uniform-time training rule. Four arms share data order,proper rotations,OT/Gaussian draws,time-uniform draws,mixture coins and self-conditioning coins. Native update keys advance from globalstep1024 through4095;the first1024draws are not replayed.',
        dataset=previous['dataset'],generation=previous['generation'],
        evaluation='Reuse the exact saved parent1024coarse/probe outputs as baseline after a declared-tolerance replay. Sample at2048and4096with16steps/32noises per protein. Final4096also has64steps/8prefix noises;solver comparisons use only the same8samples. All declared endpoints and failures are reported.',
        operational_fit_target=previous['operational_fit_target'].replace('final1024','final4096'),
        negative_control='The completed position-only study is historical context,not a compute-matched continued baseline. Its analytic same-length lower bound remains valid regardless of training duration.',
        native_coarse_separation_A=previous['native_coarse_separation_A'],
        length_only_macro_RMSD_lower_bound_A=previous['length_only_macro_RMSD_lower_bound_A'],
        failure_policy=previous['failure_policy'],
        scope='Separate bounded training-only continuation of a one-seed,two-fixed-structure learnability control. No new development sampling,independent test/R3,multi-frame ensemble-quality or published-model superiority claim. No default promotion.',
        stop_rule='Exactly3072additional updates per arm,total4096. Score1024/2048/4096without selecting a best endpoint or extending after results. Keep the original1024-update experiment immutable;report LR-policy differences explicitly.')
    paths=[Path(__file__),Path('upstream/experiments/par_two_protein_continuation.py'),
        Path('upstream/experiments/report_par_two_protein_continuation.py'),Path('upstream/tests/test_par_two_protein_continuation.py'),
        Path('upstream/scripts/par_two_protein_continuation.sbatch'),ROOT/'preflight.json',PARENT/'protocol.json',PARENT/'report.json',PARENT/'review.json']
    for parent in ['sequence_beta','sequence_uniform']:
        folder=PARENT/parent;paths.extend([folder/'last.pt',folder/'trace.json',folder/'complete.json',folder/'native_probes.json'])
        for target in p['targets']:
            for steps in [16,64]:paths.append(folder/f"step1024/{target['protein']}/coarse_nfe{steps}.npz")
    p['source_sha256']=dict(previous['source_sha256']);p['source_sha256'].update({str(q):sha(q) for q in paths})
    atomic_json(ROOT/'protocol.json',p);print('CONTINUATION_READY',list(ARMS),flush=True)


def copy_baseline(p,arm,out,moment_hash):
    parent,_=ARMS[arm]
    for target in p['targets']:
        folder=out/f"step1024/{target['protein']}";folder.mkdir(parents=True,exist_ok=True)
        for steps in [16,64]:
            source=PARENT/parent/f"step1024/{target['protein']}/coarse_nfe{steps}.npz"
            shutil.copyfile(source,folder/source.name)
    probes={'1024':read(PARENT/parent/'native_probes.json')['1024']}
    atomic_json(out/'native_probes.json',probes)
    atomic_json(out/'baseline.json',dict(parent=parent,parent_checkpoint_sha256=sha(PARENT/parent/'last.pt'),
        optimizer_state_sha256=moment_hash,optimizer_moments_preserved=True,outputs_copied_exact=True))
    return probes


def train(p,arm,resume):
    parent,policy=ARMS[arm];model,view,optimizer,rng,saved,moment_hash=warm_model(parent)
    guard=frozen_versions(model);cases,targets=cases_and_targets();bank=FeatureBank(p['training_feature_index'])
    if [r['case_id'] for r in targets]!=[r['case_id'] for r in p['targets']]:raise RuntimeError('Training pair changed')
    out=ROOT/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Continuation already complete')
    trace=[];start=0;elapsed=0.;digest=hashlib.sha256()
    if resume:
        recovered=restore_checkpoint(out/'last.pt',view,optimizer,rng,'cpu')
        if recovered['protocol_sha256']!=sha(ROOT/'protocol.json') or recovered['arm']!=arm:raise RuntimeError('Wrong continuation resume')
        start=recovered['updates'];elapsed=recovered['training_seconds'];restore_trace_prefix(out,start)
        trace=[json.loads(line) for line in (out/'trace.jsonl').read_text().splitlines()]
        for row in trace:digest.update(json.dumps(common_identity(row),sort_keys=True).encode())
        if digest.hexdigest()!=recovered['common_event_sha256']:raise RuntimeError('Continuation trace differs')
        probes=read(out/'native_probes.json')
    else:
        if (out/'last.pt').exists() or (out/'trace.jsonl').exists():raise RuntimeError('Explicit resume required')
        probes=copy_baseline(p,arm,out,moment_hash)
    writer=CompactTrainingTrace(ADDITIONAL_UPDATES,start);clock=time.perf_counter()
    for step in range(start,ADDITIONAL_UPDATES):
        row=update(model,optimizer,cases,bank,arm,p['seed'],step,p);trace.append(row)
        digest.update(json.dumps(common_identity(row),sort_keys=True).encode())
        global_updates=INITIAL_UPDATES+step+1
        if (step+1)%p['progress_every']==0:
            writer(out/'trace.json',trace)
            atomic_json(out/'progress.json',dict(phase='training',global_updates=global_updates,
                additional_updates=step+1,total_global_updates=4096,loss=row['loss'],lr=row['lr']))
        if global_updates in p['endpoints']:
            elapsed+=time.perf_counter()-clock
            sample_endpoint(p,model,bank,parent,out,global_updates)
            probes[str(global_updates)]=denoising(model,cases,bank,parent,p['seed'])
            atomic_json(out/'native_probes.json',probes);clock=time.perf_counter()
        if (step+1)%p['checkpoint_every']==0:
            assert_frozen(model,guard)
            metadata=dict(updates=step+1,global_updates=global_updates,arm=arm,parent=parent,
                protocol_sha256=sha(ROOT/'protocol.json'),training_seconds=elapsed+time.perf_counter()-clock,
                common_event_sha256=digest.hexdigest())
            save_checkpoint(out/'last.pt',view,optimizer,rng,**metadata)
            if global_updates in p['endpoints']:save_checkpoint(out/f'step{global_updates}.pt',view,optimizer,rng,**metadata)
    assert_frozen(model,guard)
    if any(float(v['step'])!=4096 for v in optimizer.state.values()):raise RuntimeError('Adam step counter did not continue')
    atomic_json(out/'complete.json',dict(status='complete',arm=arm,parent=parent,lr_policy=policy,global_updates=4096,
        additional_updates=3072,parent_optimizer_preserved=True,optimizer_steps=4096,official_parameters_unchanged=True,
        common_event_sha256=digest.hexdigest(),training_seconds=elapsed+time.perf_counter()-clock,
        host=platform.node(),new_development_or_test_coordinates_loaded=False))
    atomic_json(out/'progress.json',dict(phase='complete',global_updates=4096))
    print('CONTINUATION_COMPLETE',arm,flush=True)


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
