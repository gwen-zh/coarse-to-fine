"""Matched fresh-start flow trial: shared R1 label frame, identical training budget.

Reuse the frozen original trial engine without editing its source. Only its
process-local output root and cache reader are configured here. Separate Slurm
processes prevent changes to the original experiment or other running jobs.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import run_atlas_ensemble_flow as engine
from run_atlas_decoder_repair import load_cache as original_cache
from atlas_shared_frame import align_backbone_to_training_reference,encode_in_supplied_frame
from atlas_decoder_repair import load_repaired_tokenizer
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_shared_frame_flow_v1')
PARENT=Path('artifacts/atlas_ensemble_flow_v1')


def shared_cache():
    info=json.loads((RUN/'cache_manifest.json').read_text())
    if sha(RUN/'labels.pt')!=info['cache_sha256']:raise RuntimeError('Changed shared-frame labels')
    return torch.load(RUN/'labels.pt',map_location='cpu',weights_only=True)


def configure_engine():
    # Reuse the exact model/loss/optimizer/solver/evaluation code; no source patch.
    engine.RUN=RUN
    engine.load_cache=shared_cache


@torch.no_grad()
def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists() or (RUN/'preparation_protocol.json').exists():raise RuntimeError('Use a new matched-trial namespace')
    parent=engine.protocol();audit=Path('artifacts/atlas_shared_frame_probe_v1/report.json')
    if not json.loads(audit.read_text())['all_reconstruction_gates_pass']:raise RuntimeError('Shared-frame reconstruction check failed')
    source_paths=[Path(__file__),Path('upstream/scripts/atlas_shared_frame_flow.sbatch'),Path('upstream/experiments/atlas_shared_frame.py'),Path('upstream/tests/test_atlas_shared_frame.py'),PARENT/'protocol.json',PARENT/'report.json',audit,Path('artifacts/atlas_decoder_repair_v1/labels.pt')]
    atomic_json(RUN/'preparation_protocol.json',dict(version=1,reference='First R1 training frame',weights_changed=False,training_restart='Fresh from same seeds, never continue parent checkpoints',changes='Shared R1 training-reference coordinate frame and TRAIN-only normalization of the new labels. Model, losses, seeds, minibatch draws, solver and update budget unchanged. R1/R2 indices unchanged.',source_sha256={str(x):sha(x) for x in source_paths}))
    old=original_cache();reference=old['R1_train'][0]['target'];decoder=load_repaired_tokenizer(**parent['tokenizer'],repair_checkpoint=parent['repair_checkpoint']);groups={}
    for group in ['R1_train','R2_dense']:
        groups[group]=[]
        for case in old[group]:
            target=align_backbone_to_training_reference(case['target'],reference);c,f=encode_in_supplied_frame(decoder.model,target);ids=decoder.model.fine_book.nearest(f)
            groups[group].append(dict(id=case['id'],sequence=case['sequence'],target=target.cpu(),coarse=c.cpu(),fine_continuous=f.cpu(),fine_ids=ids.cpu()))
        print('SHARED_FRAME_FLOW_LABELS',group,len(groups[group]),flush=True)
    torch.save(groups,RUN/'labels.pt')
    atomic_json(RUN/'cache_manifest.json',dict(cache_sha256=sha(RUN/'labels.pt'),groups={k:len(v) for k,v in groups.items()},R1_reference=old['R1_train'][0]['id'],sealed_test_coordinates_loaded=False))
    shutil.copyfile(PARENT/'train_histogram_edges.npz',RUN/'train_histogram_edges.npz')
    c=torch.cat([r['coarse'] for r in groups['R1_train']]);f=torch.cat([r['fine_continuous'] for r in groups['R1_train']])
    normalization={name:value.tolist() for name,value in [('coarse_mean',c.mean(0)),('coarse_std',c.std(0,unbiased=False).clamp_min(.05)),('fine_mean',f.mean(0)),('fine_std',f.std(0,unbiased=False).clamp_min(.05))]}
    p=copy.deepcopy(parent);p['version']=1;p['parent_trial']=str(PARENT/'protocol.json');p['target_frame']='Native structures rigid-aligned to the first R1 training reference before existing encoders are applied, bypassing per-frame first-residue recanonicalization. Generation still uses AA+noise only.';p['normalization']=normalization;p['matching']='Exact same sequence/model/seeds/AA inputs/update count/batch IDs/noise-time draws/optimizer/solver/sample seeds and histogram edges. Fresh training, not budget extension of parent.'
    new_sources=[RUN/'preparation_protocol.json',RUN/'labels.pt',RUN/'cache_manifest.json',RUN/'train_histogram_edges.npz',*source_paths]
    p['source_sha256'].update({str(x):sha(x) for x in new_sources});atomic_json(RUN/'protocol.json',p)
    print('SHARED_FRAME_FLOW_PROTOCOL_READY',json.dumps(dict(updates=p['updates'],seeds=p['seeds'],R1_frames=len(groups['R1_train']),R2_frames=len(groups['R2_dense']))),flush=True)


def report(p):
    engine.report(p)
    original=json.loads((PARENT/'report.json').read_text());new=json.loads((RUN/'report.json').read_text());rows=[]
    for seed in p['seeds']:
        old_trace=json.loads((PARENT/f'seed{seed}/trace.json').read_text());new_trace=json.loads((RUN/f'seed{seed}/trace.json').read_text())
        if [x['batch'] for x in old_trace]!=[x['batch'] for x in new_trace]:raise RuntimeError('Matched frame trial changed minibatches')
        a=next(r for r in original['seeds'] if r['seed']==seed);b=next(r for r in new['seeds'] if r['seed']==seed)
        rows.append(dict(seed=seed,original=next(r for r in a['arms'] if r['arm']=='full_flow_repaired_decoder'),shared_frame=next(r for r in b['arms'] if r['arm']=='full_flow_repaired_decoder'),original_checks=a['checks'],shared_frame_checks=b['checks']))
    atomic_json(RUN/'matched_comparison.json',dict(status='complete',rows=rows,minibatches_matched=True,scope='Fresh fixed-budget target-frame comparison on one previously used MD sequence/development trajectory. No test evaluation or independent-protein claim. No automatic larger training.'))
    print('MATCHED_SHARED_FRAME_COMPLETE',json.dumps([dict(seed=r['seed'],before_CA=r['original']['mean_sample_nearest_reference_ca_rmsd_A'],after_CA=r['shared_frame']['mean_sample_nearest_reference_ca_rmsd_A'],before_CN=r['original']['geometry']['cn_rate'],after_CN=r['shared_frame']['geometry']['cn_rate'],checks=r['shared_frame_checks']) for r in rows]),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['prepare','train','report'],required=True);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='prepare':prepare()
    else:
        configure_engine();p=engine.protocol()
        if args.stage=='train':
            if args.seed not in p['seeds']:raise RuntimeError('Undeclared shared-frame seed')
            engine.train(p,args.seed)
        else:report(p)
