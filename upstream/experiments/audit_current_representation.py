"""Check current tokenizer rotation invariance, usage and block-boundary errors.

No training, production-source changes, or R3 coordinate access. The historical
41/512 result belongs to another codebook; report this own 1024-book separately.
"""
import json
from pathlib import Path
import numpy as np
import torch
from atlas_decoder_repair import load_repaired_tokenizer
from atlas_shared_frame import align_backbone_to_training_reference,encode_in_supplied_frame
from run_atlas_decoder_repair import load_cache
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/current_representation_audit_v1')


def usage(ids,size=1024):
    counts=np.bincount(np.asarray(ids,dtype=np.int64).reshape(-1),minlength=size)
    if len(counts)!=size:raise ValueError('Invalid fine token IDs')
    probabilities=counts[counts>0]/counts.sum()
    return dict(tokens=int(counts.sum()),used=int((counts>0).sum()),vocabulary=size,perplexity=float(np.exp(-(probabilities*np.log(probabilities)).sum())))


def aligned_prediction(pred,target):
    x=pred[:,1]-pred[:,1].mean(0);y=target[:,1]-target[:,1].mean(0);u,_,vt=np.linalg.svd(x.T@y);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    return (pred-pred[:,1].mean(0))@(u@correction@vt)+target[:,1].mean(0)


def candidate_frame_vectors(centers):
    previous=centers[1:-1]-centers[:-2];following=centers[2:]-centers[1:-1];tangent=previous+following
    tangent_norm=np.linalg.norm(tangent,axis=-1);e1=tangent/np.maximum(tangent_norm[:,None],1e-12)
    orthogonal=previous-(previous*e1).sum(-1,keepdims=True)*e1
    orthogonal_norm=np.linalg.norm(orthogonal,axis=-1)
    e2=orthogonal/np.maximum(orthogonal_norm[:,None],1e-12)
    relative=orthogonal_norm/np.maximum(np.linalg.norm(previous,axis=-1),1e-12)
    return e1,e2,relative,(tangent_norm>1e-6)&(orthogonal_norm>1e-6)


@torch.no_grad()
def main():
    torch.set_num_threads(2);RUN.mkdir(exist_ok=True)
    if (RUN/'report.json').exists():raise RuntimeError('Representation audit already complete')
    settings=json.loads(Path('artifacts/atlas_shared_frame_flow_v1/protocol.json').read_text());decoder=load_repaired_tokenizer(**settings['tokenizer'],repair_checkpoint=settings['repair_checkpoint']);model=decoder.model
    groups=load_cache();reference=groups['R1_train'][0]['target'];cohort=groups['R2_dense']+groups['pdb_validation']
    sources=[Path(__file__),Path('upstream/experiments/hierarchical_rigid_vq.py'),Path('upstream/experiments/atlas_shared_frame.py'),Path('upstream/experiments/atlas_decoder_repair.py'),Path('artifacts/atlas_decoder_repair_v1/labels.pt'),Path('artifacts/atlas_shared_frame_flow_v1/labels.pt'),Path('artifacts/mixed_hierarchical_generator_v1/labels/seed2026/manifest.json'),Path(settings['tokenizer']['final'])]
    atomic_json(RUN/'protocol.json',dict(scope='Diagnostic only on reused R2 and PDB development data. No model training or R3 access.',rotation_tests=['90deg_x','90deg_y','90deg_z','translation'],paths=['original full tokenizer including canonicalization','shared-reference alignment then encoder','centering-only ablation without either orientation step'],frame_test='Interior block frame from centered tangent and Gram-Schmidt previous segment. End blocks have no such three-center construction; report them separately. Compare frames from exact and reconstructed centers.',source_sha256={str(p):sha(p) for p in sources}))
    rotations=[('90deg_x',torch.tensor([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]),torch.zeros(3)),('90deg_y',torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),torch.zeros(3)),('90deg_z',torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]),torch.zeros(3)),('translation',torch.eye(3),torch.tensor([13.,-7.,4.]))]
    rotation_stats={path:{name:dict(total_tokens=0,changed_tokens=0,max_fine_latent_difference=0.) for name,_,_ in rotations} for path in ['original_full_path','shared_reference_full_path','centering_only_ablation']}
    slots={group:{i:[] for i in range(4)} for group in ['R2_dense','pdb_validation']};frame_rows=[]
    for group in ['R2_dense','pdb_validation']:
        for case in groups[group]:
            target=case['target'];c,f,_=model.encode_latents(target);ids=model.fine_book.nearest(f)
            pred=decoder.decode(case['coarse'],case['fine_ids']).numpy();aligned=aligned_prediction(pred,target.numpy());errors=np.sqrt(((aligned-target.numpy())**2).sum(-1).mean(-1))
            for slot in range(4):slots[group][slot].extend(errors[slot::4].tolist())
            exact=pool_ca(target).numpy();reconstructed=decoder.coarse_centers(case['coarse'],len(target)).numpy();a,b,conditioning,valid=candidate_frame_vectors(exact);a2,b2,_,valid2=candidate_frame_vectors(reconstructed);keep=valid&valid2
            normal_angle=np.degrees(np.arccos(np.clip((b[keep]*b2[keep]).sum(-1),-1,1)));tangent_angle=np.degrees(np.arccos(np.clip((a[keep]*a2[keep]).sum(-1),-1,1)))
            frame_rows.append(dict(group=group,id=case['id'],interior_frames=len(conditioning),end_blocks_without_center_triplet=2,invalid_exact_frames=int((~valid).sum()),near_collinear_ratio_below005=int((conditioning<.05).sum()),normal_angle_error_degrees=normal_angle.tolist(),tangent_angle_error_degrees=tangent_angle.tolist()))
            if group=='R2_dense':shared=align_backbone_to_training_reference(target,reference)
            else:shared=align_backbone_to_training_reference(target,groups[group][0]['target']) if target.shape==groups[group][0]['target'].shape else None
            # Shared alignment needs the same sequence/residue mapping; never align
            # unrelated PDB proteins to the MD reference. Test it on R2 only.
            shared_f=None
            if group=='R2_dense':_,shared_f=encode_in_supplied_frame(model,shared)
            _,raw_f=encode_in_supplied_frame(model,target)
            for name,rotation,translation in rotations:
                moved=target@rotation+translation
                _,new_f,_=model.encode_latents(moved)
                comparisons=[('original_full_path',f,new_f)]
                if group=='R2_dense':
                    _,sf=encode_in_supplied_frame(model,align_backbone_to_training_reference(moved,reference));comparisons.append(('shared_reference_full_path',shared_f,sf))
                _,rf=encode_in_supplied_frame(model,moved);comparisons.append(('centering_only_ablation',raw_f,rf))
                for path,before,after in comparisons:
                    stat=rotation_stats[path][name];stat['total_tokens']+=len(target);stat['changed_tokens']+=int((model.fine_book.nearest(before)!=model.fine_book.nearest(after)).sum());stat['max_fine_latent_difference']=max(stat['max_fine_latent_difference'],float((before-after).abs().max()))
    for values in rotation_stats.values():
        for r in values.values():r['changed_fraction']=r['changed_tokens']/r['total_tokens']
    cache_root=Path('artifacts/mixed_hierarchical_generator_v1/labels/seed2026');manifest=json.loads((cache_root/'manifest.json').read_text());tokens={'train':[],'validation':[]};counts={'train':0,'validation':0}
    for row in manifest['targets']:
        path=cache_root/(row['id']+'.npz')
        if sha(path)!=row['sha256']:raise RuntimeError('Changed fine-token cache '+str(path))
        with np.load(path,allow_pickle=False) as z:tokens[row['subset']].append(z['fine'].copy())
        counts[row['subset']]+=1
    vocabulary={subset:dict(proteins=counts[subset],**usage(np.concatenate(ids))) for subset,ids in tokens.items()}
    md_usage={}
    shared_groups=torch.load('artifacts/atlas_shared_frame_flow_v1/labels.pt',map_location='cpu',weights_only=True)
    for convention,data in [('original_frame',groups),('shared_frame',shared_groups)]:
        for group in ['R1_train','R2_dense']:
            values=np.stack([r['fine_ids'].numpy() for r in data[group]])
            md_usage[convention+'_'+group]=dict(usage(values),mean_per_residue_perplexity=float(np.mean([usage(values[:,i])['perplexity'] for i in range(values.shape[1])])))
    frame_summary={}
    for group in ['R2_dense','pdb_validation']:
        data=[r for r in frame_rows if r['group']==group];angles=np.concatenate([r['normal_angle_error_degrees'] for r in data]);tangents=np.concatenate([r['tangent_angle_error_degrees'] for r in data]);total=sum(r['interior_frames'] for r in data)
        frame_summary[group]=dict(interior_frames=total,near_collinear_fraction=sum(r['near_collinear_ratio_below005'] for r in data)/total,exact_degenerate_frames=sum(r['invalid_exact_frames'] for r in data),e2_normal_angle_median_degrees=float(np.median(angles)),e2_normal_angle_p90_degrees=float(np.percentile(angles,90)),e1_tangent_angle_median_degrees=float(np.median(tangents)),scope='Diagnostic of one proposed center-only frame construction; no local-frame tokenizer trained.')
    slot_summary={group:{str(i+1):dict(residues=len(v),mean_residue_backbone_error_A=float(np.mean(v))) for i,v in data.items()} for group,data in slots.items()}
    result=dict(status='complete',rotation=rotation_stats,full_cohort_code_usage=vocabulary,MD_code_usage=md_usage,trained_codebook_state=dict(initialized=bool(model.fine_book.initialized),EMA_dead_code_restarts=int(model.fine_book.restarts),EMA_counts_min=float(model.fine_book.counts.min())),block_position_reconstruction=slot_summary,center_frame_conditioning=frame_summary,models_changed=False,R3_coordinates_loaded=False,scope='Current model audit. Global rigid invariance and removal of relative block orientation are different properties. Low single-protein code usage is not global collapse.')
    atomic_json(RUN/'report.json',result);atomic_json(RUN/'frame_details.json',frame_rows)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
