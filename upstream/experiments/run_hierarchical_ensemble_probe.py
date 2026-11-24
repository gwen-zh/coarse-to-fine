"""Fixed-budget two-mode capacity probes, not independent ensemble validation."""
import argparse,json,math,time,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow,snap_fine
from joint_sequence_predictor import encode_aa
from mixed_hierarchical_generator import HybridTokenizer
from compare_two_level_ae_author import metrics
from structural_metrics import aligned_rmsd

RUN=Path('artifacts/hierarchical_ensemble_flow_v1')


def get_protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Frozen ensemble-probe source changed '+name)
    return p


def load(stage,p):
    if stage=='toy':
        cases=[]
        for i,sequence in enumerate(['ACDEFGHIKLMN','NPQRSTVWYACD']):
            aa=encode_aa(sequence);m=(len(aa)+3)//4;offset=.2 if i==0 else -.2
            c=torch.stack([torch.full((m,16),offset-.6),torch.full((m,16),offset+.6)])
            f=torch.stack([torch.full((len(aa),8),-.6),torch.full((len(aa),8),.6)])
            if i==1:f=f.flip(0)
            cases.append(dict(id=f'toy_condition{i}',aa=aa,coarse=c,fine=f,offset=offset,flipped=i==1))
        decoder=None
    else:
        with np.load(RUN/'paired_native_labels.npz') as z:
            aa=encode_aa(str(z['sequence'].item()))
            cases=[dict(id='1JFJ_1JFK_fit_only',aa=aa,coarse=torch.from_numpy(z['coarse'].copy()).float(),fine=torch.from_numpy(z['fine_continuous'].copy()).float(),backbone=z['backbone'].copy())]
        decoder=HybridTokenizer.from_checkpoints(p['tokenizer']['warmup'],p['tokenizer']['final'])
    c=torch.cat([row['coarse'].reshape(-1,16) for row in cases]);f=torch.cat([row['fine'].reshape(-1,8) for row in cases])
    normalization=[c.mean(0),c.std(0,unbiased=False).clamp_min(.05),f.mean(0),f.std(0,unbiased=False).clamp_min(.05)]
    return cases,decoder,normalization


def make_model(p,stage,normalization):
    return HierarchicalEnsembleFlow(*normalization,**p[stage]['model'])


@torch.no_grad()
def evaluate(model,cases,decoder,p,stage,seed,out):
    model.eval();rows=[];sample_records=[]
    codes=torch.stack([torch.full((8,),-.6),torch.full((8,),.6)]) if stage=='toy' else decoder.model.fine_book.codes
    for case in cases:
        mode_counts=[0,0];errors=[];fine_match=[];snap=[];near_counts=[0,0];geometry=[];valid_count=0
        for index in range(p[stage]['samples']):
            key=int(hashlib.sha256(f'ensemble-probe-v1:{seed}:{case["id"]}:{index}'.encode()).hexdigest()[:8],16)
            c,f=model.sample_latents(case['aa'],key,p['sampling_steps'],p['sampling_steps']);ids,distance=snap_fine(f,codes)
            latent_errors=((case['coarse']-c[None])**2).mean((1,2)).sqrt();mode=int(latent_errors.argmin());mode_counts[mode]+=1;errors.append(float(latent_errors[mode]));snap.append(float(distance.mean()))
            expected=((case['fine'][mode][:,None]-codes[None])**2).sum(-1).argmin(-1)
            accuracy=float((ids==expected).float().mean());fine_match.append(accuracy)
            record=dict(case=case['id'],sample=index,seed=key,assigned_coarse_mode=mode,nearest_coarse_latent_rmse=float(latent_errors[mode]),fine_token_match_to_assigned_mode=accuracy,fine_snap_rmse=float(distance.mean()))
            if decoder is not None:
                bb=decoder.decode(c,ids).numpy();rms=[aligned_rmsd(bb[:,1],x[:,1]) for x in case['backbone']];best=int(np.argmin(rms));radius=p['pair']['native_pair_separation_A']/3.
                if rms[best]<radius:near_counts[best]+=1;valid_count+=1
                q=metrics(bb,case['backbone'][best]);geometry.append(q);record.update(nearest_reference_rmsd_A=min(rms),nearest_reference=best,cn_rate=q['cn_rate'],near_mode_radius_A=radius)
                np.savez_compressed(out/f'sample{index}.npz',backbone=bb,coarse=c.numpy(),fine_continuous=f.numpy(),fine_ids=ids.numpy())
            sample_records.append(record)
        record=dict(case=case['id'],samples=p[stage]['samples'],coarse_mode_counts=mode_counts,mean_nearest_coarse_latent_rmse=float(np.mean(errors)),fine_token_match_to_assigned_mode=float(np.mean(fine_match)),mean_fine_snap_rmse=float(np.mean(snap)))
        if decoder is not None:
            record.update(near_native_mode_counts=near_counts,near_native_fraction=valid_count/p[stage]['samples'],mean_nearest_reference_rmsd_A=float(np.mean([r['nearest_reference_rmsd_A'] for r in sample_records if r['case']==case['id']])),geometry={k:float(np.mean([v[k] for v in geometry])) for k in geometry[0]})
        # Paired conditional intervention: keep fine noise fixed, use valid native coarse modes.
        flips=[];native_match=[]
        for j in range(16):
            predictions=[]
            for mode in [0,1]:
                f=model.sample_fine(case['aa'],case['coarse'][mode],seed+900000+j,p['sampling_steps']);ids,_=snap_fine(f,codes);predictions.append(ids)
                expected=((case['fine'][mode][:,None]-codes[None])**2).sum(-1).argmin(-1);native_match.append(float((ids==expected).float().mean()))
            flips.append(float((predictions[0]!=predictions[1]).float().mean()))
        record.update(same_fine_noise_changed_coarse_fraction=float(np.mean(flips)),fine_match_with_native_coarse=float(np.mean(native_match)))
        if stage=='toy':
            record['gate']=dict(both_coarse_modes_at_least10pct=min(mode_counts)>=.1*p[stage]['samples'],coarse_near_mode_rmse_below025=record['mean_nearest_coarse_latent_rmse']<=.25,fine_matches_generated_coarse_at_least85pct=record['fine_token_match_to_assigned_mode']>=.85)
            record['analytic_point_mean_distance_to_each_mode']=.6
        else:
            record['gate']=dict(both_native_modes_at_least10pct=min(near_counts)>=.1*p[stage]['samples'],near_native_fraction_at_least_half=record['near_native_fraction']>=.5,cn_not_over_native_ceiling_plus5pp=record['geometry']['cn_rate']<=max(p['pair']['native_reconstruction_cn_rates'])+.05)
        rows.append(record)
    result=dict(status='complete',stage=stage,seed=seed,cases=rows,pass_capacity_gate=all(all(r['gate'].values()) for r in rows),scope='Fittedtrainingdistributioncapacityprobeonly;notheldoutgeneralization,physicaloccupancy,Boltzmanncalibration,orauthorcomparison.',samples=sample_records)
    atomic_json(out/'evaluation.json',result);return result


def train(p,stage,seed):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    cases,decoder,norm=load(stage,p);model=make_model(p,stage,norm);opt=torch.optim.AdamW(model.parameters(),lr=p[stage]['lr'],weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed+22017);out=RUN/stage/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Fixed-budget probe already complete')
    start=0;trace=[];digest=sha(RUN/'protocol.json');began=time.perf_counter();updates=p[stage]['updates']
    if (out/'last.pt').exists():
        state=restore_checkpoint(out/'last.pt',model,opt,rng,torch.device('cpu'))
        if state['protocol_sha256']!=digest:raise RuntimeError('Resume protocol differs')
        start=state['updates'];generator.set_state(state['flow_rng_state']);trace=json.loads((out/'trace.json').read_text())[:start]
    for step in range(start,updates):
        model.train();case=cases[int(rng.integers(len(cases)))];mode=int(rng.integers(2));rate=p[stage]['min_lr']+.5*(p[stage]['lr']-p[stage]['min_lr'])*(1+math.cos(math.pi*step/max(1,updates-1)))
        for group in opt.param_groups:group['lr']=rate
        opt.zero_grad(set_to_none=True);loss,terms=model.flow_loss(case['aa'],case['coarse'][mode],case['fine'][mode],p['coarse_jitter_std'],generator)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite flow loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite ensemble gradient')
        opt.step();trace.append(dict(update=step+1,case=case['id'],mode=mode,loss=float(loss.detach()),coarse_flow=float(terms['coarse_flow']),fine_flow=float(terms['fine_flow']),draw_sha256=terms['draw_sha256']))
        if (step+1)%128==0 or step+1==updates:
            atomic_json(out/'progress.json',dict(updates=step+1,total=updates,seconds=time.perf_counter()-began));atomic_json(out/'trace.json',trace)
        if (step+1)%512==0 or step+1==updates:save_checkpoint(out/'last.pt',model,opt,rng,model_config=model.config,updates=step+1,flow_rng_state=generator.get_state(),protocol_sha256=digest)
    evaluation=evaluate(model,cases,decoder,p,stage,seed,out)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,stage=stage,updates=updates,parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-began,pass_capacity_gate=evaluation['pass_capacity_gate'],scope=evaluation['scope']))
    print('ENSEMBLE_PROBE_COMPLETE',stage,seed,json.dumps(evaluation['cases']),flush=True)


def report(p):
    rows=[]
    for stage in ['toy','pair']:
        for seed in p['seeds']:
            f=RUN/stage/f'seed{seed}'/'evaluation.json'
            if not f.exists():raise RuntimeError('Missing fixed-budget probe '+str(f))
            r=json.loads(f.read_text());rows.append(dict(stage=stage,seed=seed,cases=r['cases'],pass_capacity_gate=r['pass_capacity_gate']))
    r=dict(status='complete',probes=rows,toy_pass_both_seeds=all(x['pass_capacity_gate'] for x in rows if x['stage']=='toy'),pair_pass_both_seeds=all(x['pass_capacity_gate'] for x in rows if x['stage']=='pair'),scope='Capacitytestsontwo-pointconditionaldatasets;notvalidationofrealMDensemblepopulationsorgeneralization. Noautomatic899training,budgetextension,orbestcheckpointselection.')
    atomic_json(RUN/'report.json',r);print(json.dumps(r,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['toy','pair','report'],required=True);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();p=get_protocol()
    report(p) if args.stage=='report' else train(p,args.stage,args.seed)
