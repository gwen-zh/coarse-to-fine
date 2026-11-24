"""Bounded sequence-conditioning transfer on the official frozen PAR coarse field."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from coarse_generation_diagnostic import coarse_metrics,pool_backbones
from compact_training_trace import CompactTrainingTrace,restore_trace_prefix
from diagnostic_io import atomic_json
from fine_geometry_flow import sample_training_batch
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha,save_checkpoint,restore_checkpoint
from par_coarse_backbone import ROOT as BACKBONE,CHECKPOINT,load_official,attach_sequence_conditioner,sample_coarse_batch
from par_coarse_transfer import ARMS,adapter_view,conditioning_features,update,identity,frozen_versions,assert_frozen
from run_aa_structural_coarse_condition import load_fine
from run_author_codec_interface import evaluate_values,load
from coarse_full_cohort_evaluation import draw_fine_backbone
from joint_sequence_predictor import encode_aa
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec


ROOT=Path('artifacts/par_coarse_transfer_v1')
LABELS=Path('artifacts/multiframe_early_fine_v1')
PRIOR=Path('artifacts/coarse_coverage_repair_v1')
FINE=Path('artifacts/multiframe_coarse_context_v1')


class FeatureBank:
    def __init__(self,index):self.index=load(index);self.cache={}
    def __call__(self,sequence):
        if sequence not in self.index:raise ValueError('Sequence outside scoped AA feature bank')
        if sequence not in self.cache:
            with np.load(self.index[sequence]['path']) as z:h=torch.from_numpy(z['h'].copy()).float()
            if h.shape!=(len(sequence),1536) or not torch.isfinite(h).all():raise ValueError('Wrong AA-only features')
            self.cache[sequence]=h
        return self.cache[sequence]


def verify(p):
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Frozen PAR transfer input changed: '+name)


def batch_rng(seed):return np.random.default_rng(seed+99471)


def model_for(seed,device):
    model,cfg=load_official(device)
    torch.manual_seed(seed);factory=attach_sequence_conditioner(model)
    return model,factory,adapter_view(factory)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise RuntimeError('Fresh PAR transfer namespace required')
    preflight=load(BACKBONE/'real_weight_preflight.json')
    if preflight['status']!='passed' or not preflight['zero_initial_16step_sample_exact']:
        raise RuntimeError('Complete the actual pretrained GPU preflight first')
    old=load(PRIOR/'protocol.json');p={k:old[k] for k in ['tokenizer','fine_checkpoint']}
    p.update(seeds=[2026,2027],arms=ARMS,updates=512,lr=1e-4,min_lr=1e-5,weight_decay=1e-4,
        gradient_clip=1.,checkpoint_every=128,progress_every=32,sampling={"16":64,"400":8},
        training_feature_index='artifacts/multiframe_early_fine_v1/training_feature_index.json',
        feature_index=old['all_feature_index'],targets=old['author']['targets'],
        architecture='Official ratio-based nominal60M PAR checkpoint:117107160 frozen AR+flow parameters. Add a zero-output ordered stride4 AA feature path at AR initial features;464512 trainable parameters. Coarse-only transfer outputsceil(N/4)x3 physical centers. Original own geometry-aware fine8192 and original continuous codec remain frozen for end-to-end evaluation.',
        position_only_control='Same nominal adapter shape/count and native training batches, but AA feature tensors are replaced by zeros. Position/length/partial-block mask and all pretrained structure modules remain. Effective gradient capacity can differ; this isolates availability of AA information, not equal effective capacity.',
        training='512 fixed AdamW updates per arm/seed.3 uniform static PDB structures plus1 uniformly selected MD training protein/frame per update. Original899 PDB plus2264 native frames from5 MD proteins. No development optimization. Native center targets are unweighted-centered nanometers, independently proper-rotated; Gaussian noise is Kabsch-coupled to the same target.',
        objective='Native coarse velocity MSE only. Same CPU Gaussian/proper-rotation/time/self-conditioning draws and LR across arms per seed. Official time mixture2%uniform+98%Beta(1.9,1), and50%detached self-conditioning. No geometry loss, VQ, decoder gradient, ensemble reweighting or checkpoint-quality selection.',
        initialization='Every run reloads the exact official checkpoint. Residue projection initializes from the declared seed and final block projection is zero. Both arms start at the same pretrained coarse function. The runtime preflight update is discarded. No optimizer state from official pretraining is reused.',
        interface_change='Official PAR usesfloor(N/4) linearly interpolated CA points; this controlled coarse-only transfer usesceil(N/4) native stride4 CA means so our fixed fine/decoder interface stays valid. Positional metadata uses the same linear residue-index rule. This is not a reproduction of published PAR scores or a fully trained3-scale sequence-conditioned PAR.',
        evaluation='Before/after512 only. Save AA-only coarse outputs at16steps x64samples and400steps x8fixed-prefix samples per each of2 reused development sequences.16steps is the engineering primary,400steps a fixed solver diagnostic, never select the better budget after scoring. Same per-sample initial-noise keys at both endpoints and both arms. CPU fine sampling16steps and own decoder use the original noise convention. Reference coordinates are loaded only after all full generated structures are saved.',
        failure_policy='Retain every requested coarse sample and NaN fine failure. Any failed sample makes the group full metrics unavailable; no filtering/resampling or epoch0 fallback. Unexpected feature/shape errors stop the run.',
        compute='A different117M pretrained coarse network replaces the1.1M prototype and has16/400calls. No matched pretraining, capacity, total inference compute or speedup claim against the original prototype/author. Report hardware and elapsed times.',
        scope='Fixed small-budget pretrained coarse transfer pilot. Two training-draw seeds and two repeatedly reused development proteins. Static PDB is not an equilibrium ensemble. No independent test/R3 coordinate access or default model promotion.',
        stop_rule='Exactly512 updates and final endpoint only, all arms/seeds reported. Intermediate snapshots are for recovery. No loss/LR/step scan, adaptive extension, sample selection or automatic default promotion.')
    train=torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    if len(train['PDB_train'])!=899 or sum(map(len,train['train'].values()))!=2264:raise RuntimeError('Training cohort changed')
    cases=train['PDB_train']+[c for group in train['train'].values() for c in group]
    if any(c['split']!='train' or len(c['sequence'])>256 for c in cases):raise RuntimeError('Unexpected training split or length exceeds pretrained domain')
    paths=[Path(__file__),Path('upstream/experiments/par_coarse_transfer.py'),
        Path('upstream/experiments/report_par_coarse_transfer.py'),Path('upstream/tests/test_par_coarse_transfer.py'),
        Path('upstream/scripts/par_coarse_transfer.sbatch'),BACKBONE/'real_weight_preflight.json',
        LABELS/'train_labels.pt',LABELS/'development_labels.pt',Path(p['training_feature_index']),Path(p['feature_index']),
        Path(p['fine_checkpoint']),*[Path(v) for v in p['tokenizer'].values()],PRIOR/'protocol.json',FINE/'protocol.json']
    for seed in p['seeds']:paths.append(FINE/f'local_geometry/seed{seed}/last.pt')
    index=load(p['training_feature_index']);all_index=load(p['feature_index'])
    train_sequences={c['sequence'] for c in cases}
    if train_sequences!=set(index):raise RuntimeError('Training AA feature scope differs')
    if train_sequences&{t['sequence'] for t in p['targets']}:raise RuntimeError('Development sequence in training')
    paths += [Path(v['path']) for v in index.values()]
    paths += [Path(all_index[t['sequence']]['path']) for t in p['targets']]
    p['source_sha256']=dict(preflight['source_sha256'])
    p['source_sha256'].update({str(q):sha(q) for q in dict.fromkeys(paths)})
    atomic_json(ROOT/'protocol.json',p);print('PAR_COARSE_TRANSFER_READY',p['updates'],len(train_sequences),flush=True)


@torch.no_grad()
def sample_endpoint(p,model,bank,arm,seed,out,phase):
    for item in p['targets']:
        name=item['protein'];h=conditioning_features(bank(item['sequence']).to(next(model.parameters()).device),arm)
        for steps,count in [(int(k),v) for k,v in p['sampling'].items()]:
            keys=[component_seed(sample_seed(seed,name,i),'coarse') for i in range(count)]
            clock=time.perf_counter();c=sample_coarse_batch(model,h,item['length'],keys,steps)
            torch.cuda.synchronize();seconds=time.perf_counter()-clock
            folder=out/f'step{phase}'/name;folder.mkdir(parents=True,exist_ok=True)
            values=c.cpu().numpy();np.savez_compressed(folder/f'coarse_nfe{steps}.npz',coarse=values,
                valid=np.isfinite(values).all(axis=(1,2)),seeds=np.asarray(keys,dtype=np.uint32))
            atomic_json(folder/f'coarse_nfe{steps}.json',dict(samples=count,steps=steps,batch_seconds=seconds,
                GPU=torch.cuda.get_device_name(),native_coordinates_loaded=False,phase=phase))
            atomic_json(out/'progress.json',dict(phase='sampling',endpoint=phase,protein=name,steps=steps,samples=count))
            print('PAR_COARSE_TRANSFER_SAMPLED',seed,arm,phase,name,steps,flush=True)
    model.nn.init_repr_factory.features=None


def train(p,seed,arm,resume):
    if not torch.cuda.is_available():raise RuntimeError('GPU transfer job required')
    device=torch.device('cuda');model,factory,view=model_for(seed,device)
    expected=frozen_versions(model);optimizer=torch.optim.AdamW(view.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
    rng=batch_rng(seed);bank=FeatureBank(p['training_feature_index']);evaluation_bank=FeatureBank(p['feature_index'])
    groups=torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    out=ROOT/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Transfer run already complete')
    start=0;trace=[];elapsed=0.;digest=hashlib.sha256()
    if resume:
        saved=restore_checkpoint(out/'last.pt',view,optimizer,rng,device)
        if saved['protocol_sha256']!=sha(ROOT/'protocol.json') or saved['arm']!=arm or saved['seed']!=seed:raise RuntimeError('Wrong transfer resume')
        start=saved['updates'];elapsed=saved['training_seconds'];restore_trace_prefix(out,start)
        trace=[json.loads(x) for x in (out/'trace.jsonl').read_text().splitlines()]
        for row in trace:digest.update(json.dumps(identity(row),sort_keys=True).encode())
        if digest.hexdigest()!=saved['paired_identity_sha256']:raise RuntimeError('Trace/checkpoint identity mismatch')
    else:
        if (out/'last.pt').exists() or (out/'trace.jsonl').exists():raise RuntimeError('Explicit resume required')
        sample_endpoint(p,model,evaluation_bank,arm,seed,out,0)
    writer=CompactTrainingTrace(p['updates'],start);clock=time.perf_counter()
    for step in range(start,p['updates']):
        row=update(model,optimizer,sample_training_batch(groups,rng),bank,arm,seed,step,p)
        trace.append(row);digest.update(json.dumps(identity(row),sort_keys=True).encode())
        if (step+1)%p['progress_every']==0 or step+1==p['updates']:
            writer(out/'trace.json',trace)
            atomic_json(out/'progress.json',dict(phase='training',updates=step+1,total=p['updates'],
                training_seconds=elapsed+time.perf_counter()-clock,loss=row['loss']))
        if (step+1)%p['checkpoint_every']==0 or step+1==p['updates']:
            assert_frozen(model,expected)
            save_checkpoint(out/'last.pt',view,optimizer,rng,updates=step+1,arm=arm,seed=seed,
                protocol_sha256=sha(ROOT/'protocol.json'),training_seconds=elapsed+time.perf_counter()-clock,
                paired_identity_sha256=digest.hexdigest())
    torch.cuda.synchronize();seconds=elapsed+time.perf_counter()-clock
    sample_endpoint(p,model,evaluation_bank,arm,seed,out,p['updates']);assert_frozen(model,expected)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arm=arm,updates=p['updates'],
        paired_identity_sha256=digest.hexdigest(),training_seconds=seconds,official_weights_unchanged=True,
        trainable_parameters=sum(v.numel() for v in view.parameters()),host=platform.node(),gpu=torch.cuda.get_device_name(),
        native_development_coordinates_loaded=False,scope=p['scope']))
    atomic_json(out/'progress.json',dict(phase='complete',updates=p['updates'],total=p['updates']))


@torch.no_grad()
def evaluate(p,seed,arm):
    torch.set_num_threads(2);out=ROOT/arm/f'seed{seed}'
    if load(out/'complete.json')['status']!='complete':raise RuntimeError('Wait for fixed transfer endpoint')
    fine=load_fine(p,seed);codec=load_codec(p,'centers3');records=[]
    for endpoint in [0,p['updates']]:
        for item in p['targets']:
            aa=encode_aa(item['sequence']);name=item['protein']
            for steps,count in [(int(k),v) for k,v in p['sampling'].items()]:
                folder=out/f'step{endpoint}'/name
                with np.load(folder/f'coarse_nfe{steps}.npz') as z:c=z['coarse'].copy()
                backbones=[];failures=[]
                for i,value in enumerate(c):
                    key=component_seed(sample_seed(seed,name,i),'fine')
                    bb,failure=draw_fine_backbone(fine,codec,aa,torch.from_numpy(value),key,16)
                    backbones.append(bb)
                    if failure is not None:failures.append(dict(sample=i,**failure))
                bb=np.stack(backbones);np.savez_compressed(folder/f'full_nfe{steps}.npz',backbone=bb,
                    coarse=c,valid=np.isfinite(bb).all(axis=(1,2,3)))
                records.append(dict(endpoint=endpoint,protein=name,steps=steps,backbone=bb,coarse=c,
                    audit=dict(samples=count,failure_count=len(failures),failures=failures,all_samples_retained=True)))
                atomic_json(out/'evaluation_progress.json',dict(phase='decoding',completed_groups=len(records),total_groups=8))
    atomic_json(out/'full_prediction_complete.json',dict(status='complete',native_coordinates_loaded=False,groups=8))
    dev=torch.load(LABELS/'development_labels.pt',map_location='cpu',weights_only=True);rows=[]
    for record in records:
        cases=dev['validation'][record['protein']];reference=np.stack([c['target'].numpy() for c in cases])
        measured=evaluate_values(record['backbone'],reference,record['audit'])
        cm=None
        if np.isfinite(record['coarse']).all():cm=coarse_metrics(record['coarse'],pool_backbones(reference),len(cases[0]['sequence']))
        rows.append(dict(seed=seed,arm=arm,endpoint=record['endpoint'],steps=record['steps'],
            protein=record['protein'],metrics=measured,coarse_metrics=cm))
        atomic_json(out/'evaluation_progress.json',dict(phase='evaluation',completed_groups=len(rows),total_groups=8))
    atomic_json(out/'evaluation.json',dict(status='complete',rows=rows,all_samples_retained=True,
        test_coordinates_opened=False,R3_opened=False,scope=p['scope']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','train','evaluate'],required=True)
    parser.add_argument('--seed',type=int,choices=[2026,2027]);parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':prepare()
    else:
        if args.seed is None or args.arm is None:raise ValueError('Declared arm/seed required')
        p=load(ROOT/'protocol.json');verify(p)
        if args.stage=='train':train(p,args.seed,args.arm,args.resume)
        else:evaluate(p,args.seed,args.arm)


if __name__=='__main__':main()
