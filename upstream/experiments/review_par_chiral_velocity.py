"""Read-only numerical audit of the completed frozen chiral-velocity study.

Fit coordinates explicitly rather than reuse the reporter's trace-norm formula.
No training, generation, sample selection, new data split or default promotion.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_par_two_protein_fit import cases_and_targets


ROOT = Path('artifacts/par_chiral_velocity_v1')
ARMS = ['native', 'polar', 'axial']
SEEDS = [2029, 2030]


def read(path):return json.loads(Path(path).read_text())


def explicit_alignment(values, reference):
    x=np.asarray(values,dtype=np.float64)
    y=np.asarray(reference,dtype=np.float64)
    weights=np.asarray([4]*22+[2],dtype=np.float64)/90
    x=x-(x*weights[None,:,None]).sum(1,keepdims=True)
    y=y-(y*weights[:,None]).sum(0)
    u,_,vh=np.linalg.svd(np.einsum('smi,m,mj->sij',x,weights,y))
    orthogonal=u@vh
    determinant=np.linalg.det(orthogonal)
    correction=np.broadcast_to(np.eye(3),(len(x),3,3)).copy()
    correction[:,-1,-1]=np.where(determinant<0,-1.,1.)
    proper=u@correction@vh
    def residual(rotation):
        difference=x@rotation-y
        return np.sqrt(np.square(difference).sum(-1)@weights)
    return dict(proper=residual(proper),reflection_permitted=residual(orthogonal),
                negative_fit=determinant<0)


def main():
    torch.set_num_threads(2)
    p=read(ROOT/'protocol.json');report=read(ROOT/'report.json')
    if report['protocol_sha256']!=sha(ROOT/'protocol.json'):
        raise RuntimeError('Protocol/report mismatch')
    for name,digest in p['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Frozen input changed: '+name)
    cases,targets=cases_and_targets()
    if targets!=p['targets']:raise RuntimeError('Training target changed')
    observed={};sample_rows=[];total=0;max_error=0.
    def group(folder,expected):
        nonlocal total,max_error
        result=[]
        for case,target,old in zip(cases,targets,expected['per_protein']):
            with np.load(folder/target['protein']/'coarse_nfe16.npz')as z:
                values=z['coarse'];keys=z['seeds'];valid=z['valid']
            if values.shape!=(64,23,3)or keys.tolist()!=p['evaluation_noise_keys']:
                raise RuntimeError('Shape/noise mismatch')
            if not np.isfinite(values).all()or not valid.all():
                raise RuntimeError('Nonfinite or filtered outputs')
            fit=explicit_alignment(values,case['coarse'].numpy())
            negative=fit['negative_fit']
            mirror=negative&(fit['proper']>2)&(fit['reflection_permitted']<=2)
            row=dict(protein=target['protein'],samples=64,
                     mean_proper_RMSD_A=float(fit['proper'].mean()),
                     mean_reflection_permitted_RMSD_A=float(fit['reflection_permitted'].mean()),
                     negative_orthogonal_fit_count=int(negative.sum()),mirror_like_count=int(mirror.sum()),
                     samples_within_2A=int((fit['proper']<=2).sum()))
            for key in ['mean_proper_RMSD_A','mean_reflection_permitted_RMSD_A']:
                error=abs(row[key]-old[key]);max_error=max(max_error,error)
                if error>1e-6:raise RuntimeError('Independent alignment differs: '+key)
            for key in ['negative_orthogonal_fit_count','mirror_like_count','samples_within_2A']:
                if row[key]!=old[key]:raise RuntimeError('Count mismatch: '+key)
            total+=64;result.append(row)
            observed[(str(folder),target['protein'])]=fit
            sample_rows.append(dict(folder=str(folder),protein=target['protein'],
                proper_RMSD_A=fit['proper'].tolist(),reflection_permitted_RMSD_A=fit['reflection_permitted'].tolist(),
                negative_fit=negative.tolist()))
        mean=float(np.mean([r['mean_proper_RMSD_A']for r in result]))
        if abs(mean-expected['macro_proper_RMSD_A'])>1e-6:raise RuntimeError('Macro mismatch')
        return dict(per_protein=result,macro_proper_RMSD_A=mean,
                    both_targets_mean_within_2A=all(r['mean_proper_RMSD_A']<=2 for r in result))
    initial=group(ROOT/'baseline',report['baseline'])
    final={};identities={};training_records=[];progress=[]
    for seed in SEEDS:
        scores={}
        for arm in ARMS:
            out=ROOT/arm/f'seed{seed}'
            complete=read(out/'complete.json');trace=read(out/'trace.json')
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            if saved['updates']!=1024 or saved['arm']!=arm or saved['seed']!=seed or saved['protocol_sha256']!=sha(ROOT/'protocol.json'):
                raise RuntimeError('Wrong checkpoint identity')
            if complete['updates']!=1024 or not complete['official_base_parameters_unchanged']or len(trace)!=1024:
                raise RuntimeError('Incomplete training')
            if sum(v.numel()for v in saved['model'].values())!=p['trainable_parameter_counts'][arm]:
                raise RuntimeError('View parameter count mismatch')
            if not all(torch.isfinite(v).all()for v in saved['model'].values()):
                raise RuntimeError('Nonfinite checkpoint parameter')
            opt=saved['optimizer']
            for index,g in enumerate(opt['param_groups']):
                expected=5120 if index==0 else 1024
                if any(float(opt['state'][v]['step'])!=expected for v in g['params']):
                    raise RuntimeError('Adam step count mismatch')
            identities_this=[]
            for index,row in enumerate(trace):
                lr=1e-5+.5*(1e-4-1e-5)*(1+np.cos(np.pi*index/1023))
                if row['update']!=index+1 or row['draw_step']!=6144+index or row['total_adapter_updates']!=4097+index:
                    raise RuntimeError('Training step identity mismatch')
                if abs(row['lr']-lr)>1e-15 or not np.isfinite(row['loss'])or not np.isfinite(row['gradient_norm']):
                    raise RuntimeError('Invalid training loss/gradient/LR')
                if len(row['batch'])!=4:raise RuntimeError('Wrong batch size')
                for slot,event in enumerate(row['batch']):
                    case=cases[(6144+index+slot)%2]
                    if event['case_id']!=case['id']or event['protein']!=case['protein']or event['time']!=event['uniform_time_draw']:
                        raise RuntimeError('Nonpaired data or time rule')
                identities_this.append(dict(lr=row['lr'],batch=row['batch'],draw_step=row['draw_step']))
            digest=hashlib.sha256(json.dumps(identities_this,sort_keys=True).encode()).hexdigest()
            if digest!=complete['common_event_sha256']:raise RuntimeError('Event digest mismatch')
            if seed in identities and identities[seed]!=identities_this:raise RuntimeError('Arms are not paired')
            identities[seed]=identities_this
            scores[arm]=group(out/'step1024',report['effects'][str(seed)]['scores'][arm])
            training_records.append(dict(seed=seed,arm=arm,training_seconds=complete['training_seconds'],
                                         trainable_parameters=p['trainable_parameter_counts'][arm],
                                         loss_last128=float(np.mean([r['loss']for r in trace[-128:]]))))
            progress.append(dict(seed=seed,arm=arm,phase='complete',updates=1024,total=1024))
        final[str(seed)]=dict(scores=scores,
                axial_minus_native_A=scores['axial']['macro_proper_RMSD_A']-scores['native']['macro_proper_RMSD_A'],
                axial_minus_polar_A=scores['axial']['macro_proper_RMSD_A']-scores['polar']['macro_proper_RMSD_A'],
                axial_relative_improvement_vs_native_percent=100*(1-scores['axial']['macro_proper_RMSD_A']/scores['native']['macro_proper_RMSD_A']),
                axial_relative_improvement_vs_polar_percent=100*(1-scores['axial']['macro_proper_RMSD_A']/scores['polar']['macro_proper_RMSD_A']))
    if identities[2029]==identities[2030]or total!=896:raise RuntimeError('Seeds not distinct or missing outputs')
    decomposition=[]
    for seed in SEEDS:
        for target in targets:
            name=target['protein'];new=observed[(str(ROOT/'axial'/f'seed{seed}'/'step1024'),name)]
            for control in ['native','polar']:
                old=observed[(str(ROOT/control/f'seed{seed}'/'step1024'),name)]
                flips=new['negative_fit']!=old['negative_fit'];change=new['proper']-old['proper']
                decomposition.append(dict(seed=seed,protein=name,comparison='axial_minus_'+control,
                    mean_proper_change_A=float(change.mean()),
                    mean_reflection_permitted_change_A=float((new['reflection_permitted']-old['reflection_permitted']).mean()),
                    negative_to_positive=int((old['negative_fit']&~new['negative_fit']).sum()),
                    positive_to_negative=int((~old['negative_fit']&new['negative_fit']).sum()),
                    proper_change_from_parity_switches_A=float(change[flips].sum()/64),
                    proper_change_from_others_A=float(change[~flips].sum()/64)))
    passed=all(e['axial_minus_native_A']<0 and e['axial_minus_polar_A']<0 for e in final.values())
    if passed!=report['directional_consistency_passed']:raise RuntimeError('Primary criterion differs')
    status='complete_reviewed_training_pair_directional_gain_2A_target_not_met_no_promotion'
    if not passed:raise RuntimeError('This review status requires a passed directional criterion')
    atomic_json(ROOT/'independent_alignment_audit.json',dict(status='passed',outputs=total,records=sample_rows,
                max_report_mean_alignment_difference_A=max_error,paired_effect_decomposition=decomposition))
    review=dict(status=status,reviewed_at_UTC=datetime.now(timezone.utc).isoformat(),baseline=initial,effects=final,
        directional_consistency_passed=passed,training_records=training_records,
        interpretation=[
          'Both prespecified training-draw seeds satisfy axial<native AND axial<polar proper coarse macro RMSD. Relative improvement over native is4.77%/9.99%,and over parameter-matched polar4.22%/7.95%. This is directional repeatability on the training pair,not a statistical significance or generalization result.',
          'Benefits concentrate on1ef1_D. The1b2s_E seed2029 gain is only0.00263A versus native. Do not describe the benefit as equally strong on all proteins.',
          'Negative unconstrained orthogonal-fit counts change native41->axial38 of128 for2029 and47->40 for2030. Mirror-like outputs remain common,and both training proteins still miss the mean<=2A fitting target. Negative determinant alone is not an exact-mirror label.',
          'All896 requested outputs are finite and retained. All6144 updates,noise/time/data events and Adam counters were checked. Same parameter/Adam warm start;only training random draws vary across seeds. No best-seed/step choice or adaptive extension.',
          'The pure velocity objective and equal residual parameter counts support this bounded architecture comparison. Different vector bases and magnitudes do not isolate a unique causal chirality mechanism. Whole-model equivariance or thermal-distribution preservation is not established.',
          'Close this fixed study without promoting defaults. Keep the failed signed-volume family closed. A next experiment must be separately specified;remaining mirror modes and training-pair fitting remain unresolved before any large MD expansion or independent-test claim.'
        ],audits=dict(all_frozen_input_hashes_verified=len(p['source_sha256']),
                     all_requested_outputs_independently_aligned=total,all_outputs_finite=True,
                     all_training_events_paired=True,all_checkpoints_and_Adam_counters_verified=True,
                     local_update_draw_offsets_and_uniform_time_verified=True,all_fixed_arms_and_seeds_retained=True,
                     max_report_mean_alignment_difference_A=max_error,figure_visually_inspected=False),
        protocol_sha256=sha(ROOT/'protocol.json'),report_sha256=sha(ROOT/'report.json'),
        review_code_sha256=sha(Path(__file__)),model_default_replaced=False,independent_test_claim=False,
        new_development_predictions=0,test_coordinates_opened=False,R3_opened=False,
        next_experiment_submitted=False,figure='paper/figures/par_chiral_velocity_v1/PAR_Chiral_Velocity.pdf')
    atomic_json(ROOT/'review.json',review)
    path=ROOT/'jobs.json';jobs=read(path)
    jobs.update(status=status,review=str(ROOT/'review.json'),completed_runs=6,all_models_running=False,
                actual_trial_updates=6144,verified_progress=progress,last_verified_UTC=review['reviewed_at_UTC'],
                directional_consistency_passed=True,next_experiment_submitted=False)
    atomic_json(path,jobs)
    path=Path('artifacts/run_state.json');state=read(path);state['par_chiral_velocity_v1']=jobs;atomic_json(path,state)
    print(json.dumps(dict(status=status,effects=final,audits=review['audits']),indent=2),flush=True)


if __name__=='__main__':main()
