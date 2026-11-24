"""One-shot R3 confirmation of frozen single-sequence MD candidate and controls.

The development trials never opened R3. This separately frozen protocol does so
only after both candidate seeds passed development gates. Future iterations
must not call this same R3 an untouched holdout again.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mdtraj as md
import numpy as np
import torch
from atlas_md_io import backbone_indices,backbone_angstrom,uniform_frame_indices
from atlas_shared_frame import align_backbone_to_training_reference,encode_in_supplied_frame
from atlas_decoder_repair import load_repaired_tokenizer
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow,snap_fine
from native_ensemble_metrics import compare_ensembles
from structural_metrics import aligned_rmsd
from compare_two_level_ae_author import metrics
from run_atlas_decoder_repair import load_cache
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_md_confirmation_v1')
OLD=Path('artifacts/atlas_ensemble_flow_v1')
NEW=Path('artifacts/atlas_shared_frame_flow_v1')
MD=Path('artifacts/atlas_md_pilot_v1')


def freeze():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Confirmation protocol already frozen')
    candidate=json.loads((NEW/'report.json').read_text())
    if not candidate['both_seeds_pass_capacity_gate']:raise RuntimeError('Candidate did not pass both development seeds')
    comparison=json.loads((NEW/'matched_comparison.json').read_text())
    if not comparison['minibatches_matched']:raise RuntimeError('Development comparison was not paired')
    settings={name:json.loads((root/'protocol.json').read_text()) for name,root in [('original_frame',OLD),('shared_frame',NEW)]}
    for data in settings.values():
        for path,digest in data['source_sha256'].items():
            if sha(path)!=digest:raise RuntimeError('Changed frozen development input '+path)
    raw=json.loads((MD/'download_manifest.json').read_text());r3=next(r['path'] for r in raw['extracted'] if '_R3_' in r['path']);top=next(r['path'] for r in raw['extracted'] if r['path'].endswith('.pdb'))
    # R3 has only been integrity-hashed before this separately frozen test.
    sources=[Path(__file__),Path('upstream/scripts/atlas_md_confirmation.sbatch'),MD/'download_manifest.json',Path(top),Path(r3),OLD/'protocol.json',NEW/'protocol.json',NEW/'report.json',NEW/'matched_comparison.json',Path('artifacts/atlas_decoder_repair_v1/labels.pt'),OLD/'train_histogram_edges.npz',*[root/f'seed{seed}/last.pt' for root in [OLD,NEW] for seed in [2026,2027]]]
    p=dict(version=1,seeds=[2026,2027],reference_frames=64,samples_per_model_seed=64,trajectory=r3,topology=top,sequence=settings['shared_frame']['sequence'],settings=settings,reference_selection='64 evenly spaced R3 frames including endpoints, chosen by index without inspecting coordinates or metrics',generation_seed_rule='sha256(atlas-confirmation-v1:training_seed:sample_index) first32bits, paired between original/shared-frame models; disjoint namespace from development samples',
        comparisons='Original-frame vs shared-frame generators, same repaired decoder, all samples retained. Native encode/decode checks separate frozen decoder repair from full generation. No oracle inputs to full generation.',
        checks='For each seed: shared-frame improves sample-to-reference and reference-to-sample nearest CA RMSD, pairwise-distance JS divergence, and spread proximity to R3; generated CN <=5% and boundary CN <=15%. Descriptive two-seed confirmation, not an iid frame significance test.',
        stop_rule='Exactly these fixed checkpoints/settings/seeds; no test-based model or sample selection, fitting, threshold changes, or retries with new random samples. R3 becomes used after this confirmation. Same-sequence MD replicate, not unseen-protein or thermodynamic validation.',source_sha256={str(s):sha(s) for s in sources})
    atomic_json(RUN/'protocol.json',p);print('R3_CONFIRMATION_PROTOCOL_FROZEN',r3,flush=True)


def protocol():
    p=json.loads((RUN/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Changed confirmation source '+path)
    return p


@torch.no_grad()
def prepare(p):
    if (RUN/'reference.npz').exists():raise RuntimeError('R3 reference already prepared')
    # Explicitly log the first use before parsing any R3 coordinates.
    atomic_json(RUN/'test_access.json',dict(status='R3_unsealed_for_frozen_confirmation',protocol_sha256=sha(RUN/'protocol.json'),future_reuse_is_not_untouched_test=True))
    top=md.load_topology(p['topology']);indices=backbone_indices(top,p['sequence']);traj=md.load(p['trajectory'],top=top);chosen=uniform_frame_indices(len(traj),p['reference_frames']);bb=backbone_angstrom(traj.xyz[chosen],indices)
    cn=np.linalg.norm(bb[:,:-1,2]-bb[:,1:,0],axis=-1)
    if ((cn<1.)|(cn>1.8)).mean(1).max()>.01:raise RuntimeError('Invalid native R3 bonds')
    train_reference=load_cache()['R1_train'][0]['target'];settings=p['settings']['shared_frame'];old=HybridTokenizer.from_checkpoints(**settings['tokenizer']);repaired=load_repaired_tokenizer(**settings['tokenizer'],repair_checkpoint=settings['repair_checkpoint'])
    rows=[];reference=[]
    for x in bb:
        c,f,target=old.model.encode_latents(torch.from_numpy(x));ids=old.model.fine_book.nearest(f);shared=align_backbone_to_training_reference(target,train_reference);sc,sf=encode_in_supplied_frame(old.model,shared);sid=old.model.fine_book.nearest(sf);reference.append(shared.numpy())
        for frame,cv,fid in [('original_frame',c,ids),('shared_frame',sc,sid)]:
            for name,d in [('before_repair',old),('repaired',repaired)]:rows.append(dict(frame=frame,decoder=name,metrics=metrics(d.decode(cv,fid).numpy(),x)))
    np.savez_compressed(RUN/'reference.npz',backbone=np.stack(reference),frame_indices=chosen,time_ps=traj.time[chosen],train_alignment_reference=train_reference.numpy())
    summary=[]
    for frame in ['original_frame','shared_frame']:
        for decoder in ['before_repair','repaired']:
            selected=[r['metrics'] for r in rows if r['frame']==frame and r['decoder']==decoder];summary.append(dict(frame=frame,decoder=decoder,metrics={k:float(np.mean([r[k] for r in selected])) for k in selected[0]}))
    atomic_json(RUN/'reconstruction.json',dict(status='complete',groups=summary,frames=len(reference),reference_sha256=sha(RUN/'reference.npz')));print('R3_REFERENCE_READY',len(reference),flush=True)


@torch.no_grad()
def evaluate(p,seed):
    out=RUN/f'seed{seed}';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():raise RuntimeError('Confirmation seed already evaluated')
    with np.load(RUN/'reference.npz',allow_pickle=False) as z:reference=z['backbone'].copy();alignment=z['train_alignment_reference'].copy()[:,1]
    with np.load(OLD/'train_histogram_edges.npz',allow_pickle=False) as z:distance_edges=z['distance'].copy();rg_edges=z['rg'].copy()
    from joint_sequence_predictor import encode_aa
    aa=encode_aa(p['sequence']);rows=[]
    for name,root in [('original_frame',OLD),('shared_frame',NEW)]:
        setting=p['settings'][name];model=HierarchicalEnsembleFlow(**setting['normalization'],**setting['model']).eval();state=torch.load(root/f'seed{seed}/last.pt',map_location='cpu',weights_only=False)
        if state['protocol_sha256']!=sha(root/'protocol.json') or state['updates']!=setting['updates']:raise RuntimeError('Confirmation checkpoint does not match frozen training')
        model.load_state_dict(state['model']);decoder=load_repaired_tokenizer(**setting['tokenizer'],repair_checkpoint=setting['repair_checkpoint']);backbones=[];keys=[]
        for index in range(p['samples_per_model_seed']):
            key=int(hashlib.sha256(f'atlas-confirmation-v1:{seed}:{index}'.encode()).hexdigest()[:8],16);keys.append(key)
            c,f=model.sample_latents(aa,key,setting['coarse_steps'],setting['fine_steps']);ids,_=snap_fine(f,decoder.model.fine_book.codes);backbones.append(decoder.decode(c,ids).numpy())
        backbones=np.stack(backbones);stats=compare_ensembles(backbones[:,:,1],reference[:,:,1],distance_edges,rg_edges,alignment);geometry=[]
        for x in backbones:
            nearest=int(np.argmin([aligned_rmsd(x[:,1],r[:,1]) for r in reference]));geometry.append(metrics(x,reference[nearest]))
        stats['geometry']={k:float(np.mean([r[k] for r in geometry])) for k in geometry[0]};rows.append(dict(frame=name,**stats));np.savez_compressed(out/(name+'.npz'),backbone=backbones,sample_seeds=np.array(keys,dtype=np.uint32))
    a,b=rows;native=b['reference_mean_pairwise_ca_rmsd_A'];checks=dict(sample_reference_CA=b['mean_sample_nearest_reference_ca_rmsd_A']<a['mean_sample_nearest_reference_ca_rmsd_A'],reference_sample_CA=b['mean_reference_nearest_sample_ca_rmsd_A']<a['mean_reference_nearest_sample_ca_rmsd_A'],JS_distance=b['js_pairwise_distance_divergence_nats']<a['js_pairwise_distance_divergence_nats'],spread_closer_to_native=abs(b['mean_pairwise_ca_rmsd_A']-native)<abs(a['mean_pairwise_ca_rmsd_A']-native),CN_below5pct=b['geometry']['cn_rate']<=.05,boundary_CN_below15pct=b['geometry']['boundary_cn_rate']<=.15)
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,models=rows,checks=checks,all_checks_passed=all(checks.values()),samples_filtered=False,test_based_changes=False));print('R3_CONFIRMATION_SEED',seed,json.dumps(checks),flush=True)


def report(p):
    results=[json.loads((RUN/f'seed{seed}/complete.json').read_text()) for seed in p['seeds']]
    result=dict(status='complete',seeds=results,both_seeds_all_checks_passed=all(r['all_checks_passed'] for r in results),reconstruction=json.loads((RUN/'reconstruction.json').read_text()),R3_now_used=True,scope='One-shot frozen-model confirmation on a previously unused replicate of the SAME protein, with new paired generation seeds. No new-protein generalization or equilibrium-distribution claim. R3 cannot be recycled as untouched test for future tuning.')
    atomic_json(RUN/'report.json',result);print('R3_CONFIRMATION_REPORT',result['both_seeds_all_checks_passed'],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--stage',choices=['freeze','prepare','evaluate','report'],required=True);a.add_argument('--seed',type=int,default=2026);args=a.parse_args();torch.set_num_threads(2)
    if args.stage=='freeze':freeze()
    else:
        p=protocol()
        if args.stage=='prepare':prepare(p)
        elif args.stage=='evaluate':
            if args.seed not in p['seeds']:raise RuntimeError('Undeclared confirmation seed')
            evaluate(p,args.seed)
        else:report(p)
