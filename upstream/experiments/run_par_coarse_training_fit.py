"""No-training diagnosis of PAR coarse fit on presented training sequences."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from coarse_generation_diagnostic import coarse_metrics, weighted_rmsd_matrix
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from par_coarse_backbone import load_official, coarse_batch, ar_condition, flow_batch, sample_coarse_batch
from par_coarse_training_fit import select_targets, fixed_probe
from par_coarse_transfer import ARMS, conditioning_features, frozen_versions, assert_frozen
from run_multisequence_ensemble_flow import sample_seed
from run_par_coarse_transfer import LABELS, FeatureBank, model_for, verify


ROOT = Path('artifacts/par_coarse_training_fit_v1')
PARENT = Path('artifacts/par_coarse_cpu_transfer_v1')


def read(path): return json.loads(Path(path).read_text())


@torch.no_grad()
def audit():
    """Compare unchanged official VF primitives on a native compatible length.

    Same explicit noise, uniform16 grid and native first-scale point count.
    This is an operator/interface check, not a published-score reproduction.
    """
    parent = read(PARENT/'protocol.json'); verify(parent)
    model, cfg = load_official('cpu')
    from proteinfoundation.protein_autoregressive_modeling.feature_utils import get_int_pos_idx_from_lengths_one_scale
    length = 104
    points = model.protein_decomposer.get_real_num_centroids(length)[0]
    assert points == (length+3)//4
    lengths = torch.full((2,), length, dtype=torch.int32)
    bos = model.protein_decomposer.bos_latent[:, :, :points].clone().permute(0,2,1).expand(2,-1,-1)
    bos = bos-bos.mean(1,keepdim=True)
    upstream = dict(x_input=bos, mask=torch.ones(2,points,dtype=torch.bool), attn_mask=None,
        cur_scale=torch.tensor(0,dtype=torch.int32), cur_nres=points,
        norm_lengths=lengths/model.protein_decomposer.max_length,
        int_pos_idx=get_int_pos_idx_from_lengths_one_scale(lengths,points,target_length=points))
    wrapped = coarse_batch(length,points,'cpu',batch_size=2)
    for key,value in upstream.items():
        if torch.is_tensor(value): torch.testing.assert_close(value,wrapped[key],atol=0,rtol=0)
        else: assert value == wrapped[key]
    guard = frozen_versions(model)
    keys = [989801,989802]
    actual = sample_coarse_batch(model,None,length,keys,16)
    b,z = ar_condition(model,None,length,2)
    state = torch.stack([torch.randn((points,3),generator=torch.Generator().manual_seed(key)) for key in keys])
    state = state-state.mean(1,keepdim=True)
    previous = None
    for index in range(16):
        batch = flow_batch(b,z,state,index/16,previous)
        previous,velocity = model.atom_diffusion.predict_clean_n_v_w_guidance(batch)
        state,_ = model.atom_diffusion.fm.simulation_step(state,velocity,batch['t'],1/16,0.,'vf',0.,0.,mask=b['mask'])
    official = state*10.
    official = official-official.mean(1,keepdim=True)
    maximum = float((actual-official).abs().max())
    average = float((actual-official).abs().mean())
    assert cfg.model.target_pred == 'v' and cfg.model.atom_diff.target_pred == 'v'
    assert model.atom_diffusion.fm.scale_ref == 1.
    torch.testing.assert_close(actual,official,atol=1e-3,rtol=1e-5)
    groups = torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    case = groups['train']['1d4t_A'][0]
    ca = case['target'][:,1]
    interpolated = model.protein_decomposer.downsample(ca.T[None],points)[0].T
    means = torch.stack([ca[j:j+4].mean(0) for j in range(0,len(ca),4)])
    mismatch = float(weighted_rmsd_matrix(means[None].numpy(),interpolated[None].numpy(),length)[0,0])
    assert_frozen(model,guard)
    result = dict(status='passed',length=length,points=points,batch_fields_exact=True,
        official_VF_vs_wrapper_max_abs_A=maximum,official_VF_vs_wrapper_mean_abs_A=average,
        fixed_tolerance_abs_A=1e-3,pretrained_weights_unchanged=True,velocity_parameterization=True,
        gaussian_scale_nm=1.,self_conditioning=bool(cfg.training.self_cond),
        native_interpolation_vs_block_mean_RMSD_A=mismatch,training_case=case['id'],
        point_counts=[dict(length=n,native_floor=model.protein_decomposer.get_real_num_centroids(n)[0],own_ceil=(n+3)//4) for n in [90,94,103,104,109,117]],
        scope='Official native first-scale input fields and VF primitives on the same noise/uniform16 grid; not full three-scale sampling,original SDE settings,designability or an independent quality test.',
        development_or_test_coordinates_loaded=False)
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'official_interface_audit.json',result)
    print('OFFICIAL_INTERFACE_AUDIT',json.dumps(result),flush=True)


def prepare():
    if (ROOT/'protocol.json').exists(): raise RuntimeError('Fresh diagnostic namespace required')
    parent=read(PARENT/'protocol.json'); verify(parent)
    if read(ROOT/'official_interface_audit.json')['status']!='passed': raise RuntimeError('Audit must pass first')
    if read(PARENT/'review.json')['model_default_replaced']: raise RuntimeError('Unexpected default change')
    groups=torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    development=torch.load(LABELS/'development_labels.pt',map_location='cpu',weights_only=True)
    traces={seed:read(PARENT/'sequence'/f'seed{seed}/trace.json') for seed in parent['seeds']}
    targets=select_targets(groups,development,traces,parent['targets'])
    p=dict(parent_protocol=str(PARENT/'protocol.json'),seeds=parent['seeds'],arms=ARMS,targets=targets,
        samples=16,steps=16,endpoints=[0,512],optimizer_updates=0,probe_times=[.05,.5,.9],probe_replicates=[0,1],
        training_feature_index=parent['training_feature_index'],feature_index=parent['feature_index'],
        sampling='AA-only coarse generation,all nine target groups x16 paired noises. Training targets are new draws; the two reused development targets use exact16-sample prefixes from the completed CPU pilot. No fine/decoder calls. Save all coarse predictions before loading native coordinates in each run.',
        cohort='All5 MD training proteins plus2 static PDB training structures,closest length103/109,ties by ID,actually presented in BOTH seeds. Probe MD frames are the first4 lexicographic case IDs presented in BOTH seeds. No outcome-based selection. Two existing development proteins remain reused development.',
        reference_policy='MD and development:128 fixed linspace-selected reference frames. Static PDB:one reference. Report each domain separately; PDB single-reference error and MD multi-reference error are not comparable measures of a generalization gap.',
        denoising='After AA-only predictions are complete,reference-conditioned velocity probes at t=.05/.5/.9,two fixed independent OT/noise/proper-rotation draws per native case. Same self-conditioning coin in all endpoints/arms. Not AA-only generation or additional quality evidence.',
        failures='Retain every requested sample. Any nonfinite prediction makes the entire group unavailable;no filtering,replacement or best-case selection.',
        scope='Read-only post-pilot mechanism diagnosis,2seeds,5training MD proteins,2training PDB proteins,2reused development proteins. No new training,no independent test/R3,no default promotion. Difficulty/distribution differences prevent identifying a unique cause from a train/development gap.',
        stop_rule='Only declared cases,endpoints,and probe grid;no hyperparameter changes or new sampling on development/test. Do not infer architectural impossibility from512 native adapter updates.')
    p['probe_case_order']=sorted({i for row in targets for i in row['probe_case_ids']})
    paths=[Path(__file__),Path('upstream/experiments/par_coarse_training_fit.py'),Path('upstream/experiments/report_par_coarse_training_fit.py'),
        Path('upstream/tests/test_par_coarse_training_fit.py'),Path('upstream/scripts/par_coarse_training_fit.sbatch'),
        ROOT/'official_interface_audit.json',PARENT/'protocol.json',PARENT/'review.json',LABELS/'train_labels.pt',LABELS/'development_labels.pt']
    for seed in p['seeds']:
        for arm in ARMS:
            d=PARENT/arm/f'seed{seed}'
            paths.extend([d/'last.pt',d/'trace.json',d/'complete.json'])
            for target in parent['targets']:
                for ep in [0,512]: paths.append(d/f"step{ep}/{target['protein']}/coarse_nfe16.npz")
    p['source_sha256']=dict(parent['source_sha256'])
    p['source_sha256'].update({str(path):sha(path) for path in paths})
    atomic_json(ROOT/'protocol.json',p)
    print('TRAINING_FIT_READY',[(r['protein'],r['domain'],r['length']) for r in targets],flush=True)


@torch.no_grad()
def run(p,seed,arm):
    out=ROOT/arm/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): raise RuntimeError('Diagnostic already complete')
    model,factory,view=model_for(seed,'cpu'); model.eval().requires_grad_(False)
    initial=copy.deepcopy(view.state_dict())
    saved=torch.load(PARENT/arm/f'seed{seed}/last.pt',map_location='cpu',weights_only=False)
    if saved['updates']!=512 or saved['seed']!=seed or saved['arm']!=arm or saved['protocol_sha256']!=sha(PARENT/'protocol.json'):
        raise RuntimeError('Wrong frozen adapter checkpoint')
    bank=FeatureBank(p['feature_index']);train_bank=FeatureBank(p['training_feature_index'])
    for ep in p['endpoints']:
        view.load_state_dict(initial if ep==0 else saved['model'],strict=True)
        guard=frozen_versions(model)
        for i,target in enumerate(p['targets']):
            name=target['protein'];keys=[component_seed(sample_seed(seed,name,j),'coarse') for j in range(16)]
            start=time.perf_counter()
            if target['domain']=='development':
                source=PARENT/arm/f'seed{seed}/step{ep}/{name}/coarse_nfe16.npz'
                with np.load(source) as z:
                    np.testing.assert_array_equal(z['seeds'][:16],np.asarray(keys,dtype=np.uint32))
                    values=z['coarse'][:16].copy()
            else:
                h=conditioning_features(train_bank(target['sequence']),arm)
                values=sample_coarse_batch(model,h,target['length'],keys,16).numpy()
            folder=out/f'step{ep}';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{name}.npz',coarse=values,valid=np.isfinite(values).all(axis=(1,2)),seeds=np.asarray(keys,dtype=np.uint32))
            atomic_json(folder/f'{name}.json',dict(seconds=time.perf_counter()-start,new_draw=target['domain']!='development',native_coordinates_loaded=False))
            atomic_json(out/'progress.json',dict(phase='sampling',endpoint=ep,completed_groups=i+1,total_groups=len(p['targets']),protein=name))
        assert_frozen(model,guard)
    atomic_json(out/'prediction_complete.json',dict(status='complete',groups=18,requested_coarse_samples=288,native_coordinates_loaded=False))
    # Native labels enter this process only after all AA-only predictions exist.
    groups=torch.load(LABELS/'train_labels.pt',map_location='cpu',weights_only=True)
    development=torch.load(LABELS/'development_labels.pt',map_location='cpu',weights_only=True)
    case_map={c['id']:c for c in groups['PDB_train']}
    for cases in list(groups['train'].values())+list(development['validation'].values()):
        for c in cases:
            if c['id'] in case_map: raise RuntimeError('Ambiguous native case identifier')
            case_map[c['id']]=c
    pdb={c['protein']:[c] for c in groups['PDB_train']};rows=[];probes=[]
    for ep in p['endpoints']:
        view.load_state_dict(initial if ep==0 else saved['model'],strict=True);guard=frozen_versions(model)
        for target in p['targets']:
            name=target['protein']
            cases=(groups['train'] if target['domain']=='MD_train' else pdb if target['domain']=='PDB_train' else development['validation'])[name]
            refs=np.stack([cases[i]['coarse'].numpy() for i in target['reference_indices']])
            with np.load(out/f'step{ep}/{name}.npz') as z:generated=z['coarse'].copy()
            failures=int((~np.isfinite(generated).all(axis=(1,2))).sum())
            measured=coarse_metrics(generated,refs,target['length']) if not failures else None
            rows.append(dict(seed=seed,arm=arm,endpoint=ep,protein=name,domain=target['domain'],metrics=measured,failure_count=failures))
            features=conditioning_features(bank(target['sequence']),arm)
            batch,condition=ar_condition(model,features,target['length'])
            for case_id in target['probe_case_ids']:
                for t in p['probe_times']:
                    for replicate in p['probe_replicates']:
                        state,truth,_,self_condition,digest=fixed_probe(case_map[case_id],seed,p['probe_case_order'].index(case_id),replicate,t)
                        state=state[None];truth=truth[None];sc=None
                        if self_condition:
                            prelim=model.atom_diffusion.nn(flow_batch(batch,condition,state,t))['coors_pred']
                            sc=state+(1-t)*prelim
                        predicted=model.atom_diffusion.nn(flow_batch(batch,condition,state,t,sc))['coors_pred']
                        loss=float((predicted-truth).square().mean());zero_loss=float(truth.square().mean())
                        if not np.isfinite(loss): raise RuntimeError('Nonfinite native denoising probe')
                        probes.append(dict(seed=seed,arm=arm,endpoint=ep,protein=name,domain=target['domain'],
                            case_id=case_id,time=t,replicate=replicate,draw_sha256=digest,self_condition=self_condition,
                            velocity_MSE_nm2=loss,zero_velocity_MSE_nm2=zero_loss,relative_MSE=loss/max(zero_loss,1e-12)))
            atomic_json(out/'progress.json',dict(phase='native_probes',endpoint=ep,protein=name,probes=len(probes)))
        assert_frozen(model,guard)
    atomic_json(out/'evaluation.json',dict(status='complete',rows=rows,probes=probes,native_labels_used_only_after_generation=True))
    atomic_json(out/'complete.json',dict(status='complete',seed=seed,arm=arm,optimizer_updates=0,
        groups=len(rows),probe_rows=len(probes),all_requested_samples_retained=True,weights_unchanged=True,
        test_coordinates_opened=False,R3_opened=False,host=platform.node()))
    atomic_json(out/'progress.json',dict(phase='complete'))
    print('TRAINING_FIT_COMPLETE',seed,arm,len(rows),len(probes),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['audit','prepare','run'],required=True)
    parser.add_argument('--seed',type=int,choices=[2026,2027]);parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=='audit': audit()
    elif args.stage=='prepare': prepare()
    else:
        if args.seed is None or args.arm is None: raise ValueError('Declared arm/seed required')
        p=read(ROOT/'protocol.json');verify(p);run(p,args.seed,args.arm)


if __name__=='__main__': main()
