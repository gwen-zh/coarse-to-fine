"""Posthoc frame-gauge/valid-point checks; does not change the frozen pilot gates."""
import json
from pathlib import Path
import numpy as np
import torch
from Bio.SeqUtils import seq3
from Bio.PDB import PDBParser
from atlas_decoder_repair import load_repaired_tokenizer
from run_atlas_decoder_repair import load_cache
from native_ensemble_metrics import compare_ensembles
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_ensemble_flow_v1')


def align_frames(values,reference):
    result=[];reference=reference-reference.mean(0)
    for frame in values:
        x=frame-frame.mean(0);u,_,vt=np.linalg.svd(x.T@reference);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt);result.append(x@(u@correction@vt))
    return np.stack(result)


def write_backbone_ensemble(path,backbone,sequence):
    names=[seq3(letter).upper() for letter in sequence];lines=[]
    for model,bb in enumerate(backbone,1):
        lines.append(f'MODEL     {model:4d}')
        serial=0
        for index,(residue,atoms) in enumerate(zip(names,bb),1):
            for atom,xyz in zip(['N','CA','C'],atoms):
                serial+=1;x,y,z=xyz;element='N' if atom=='N' else 'C'
                lines.append(f'ATOM  {serial:5d} {atom:>4s} {residue:>3s} A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}')
        lines.extend(['TER','ENDMDL'])
    lines.append('END');path.write_text('\n'.join(lines)+'\n')
    parsed=PDBParser(QUIET=True).get_structure('generated',str(path));models=list(parsed.get_models())
    if len(models)!=len(backbone) or any(len(list(m.get_residues()))!=len(sequence) for m in models):raise RuntimeError('Backbone PDB export failed its round-trip check')


def main():
    torch.set_num_threads(2)
    p=json.loads((RUN/'protocol.json').read_text());groups=load_cache()
    report=RUN/'report.json'
    if not report.exists():raise RuntimeError('Finish the fixed-budget primary pilot first')
    path=RUN/'posthoc_geometry_protocol.json'
    if path.exists():raise RuntimeError('Posthoc diagnostic already recorded')
    sources=[Path(__file__),report,RUN/'protocol.json',Path('artifacts/atlas_decoder_repair_v1/labels.pt'),*[RUN/f'seed{seed}/full_flow_repaired_decoder.npz' for seed in p['seeds']]]
    atomic_json(path,dict(scope='Posthoc descriptive analysis only; original gates unchanged. All R1/R2 data were already used by the pilot; no R3 loading.',definitions=['Compare variance of canonical CA targets before/after rigid alignment to the first R1 training frame.','Choose one valid R1 native frame nearest the TRAIN average rigid-aligned coordinates as a stronger deterministic control; do not choose using R2.','Fit visualization PCA on rigid-aligned R1 CA coordinates only; show all generated samples.'],source_sha256={str(s):sha(s) for s in sources}))
    train=np.stack([r['target'][:,1].numpy() for r in groups['R1_train']]);reference=np.stack([r['target'][:,1].numpy() for r in groups['R2_dense']]);aligned=align_frames(train,train[0]);dev=align_frames(reference,train[0]);mean=aligned.mean(0)
    before=float(((train-train.mean(0))**2).sum(-1).mean());after=float(((aligned-mean)**2).sum(-1).mean())
    chosen=int(np.argmin(((aligned-mean)**2).sum(-1).mean(-1)));case=groups['R1_train'][chosen]
    decoder=load_repaired_tokenizer(**p['tokenizer'],repair_checkpoint=p['repair_checkpoint'])
    with torch.no_grad():central=decoder.decode(case['coarse'],case['fine_ids']).numpy()
    with np.load(RUN/'train_histogram_edges.npz',allow_pickle=False) as z:distance_edges=z['distance'].copy();rg_edges=z['rg'].copy()
    point=compare_ensembles(np.repeat(central[None,:,1],p['samples'],axis=0),reference,distance_edges,rg_edges,train[0])
    matrix=(aligned-mean).reshape(len(aligned),-1);_,singular,components=np.linalg.svd(matrix,full_matrices=False);components=components[:2]
    plot_data=dict(train_projection=matrix@components.T,development_projection=(dev-mean).reshape(len(dev),-1)@components.T,train_PC_variance_fraction=singular[:2]**2/(singular**2).sum())
    for seed in p['seeds']:
        with np.load(RUN/f'seed{seed}/full_flow_repaired_decoder.npz',allow_pickle=False) as z:bb=z['backbone'].copy()
        projected=align_frames(bb[:,:,1],train[0]);plot_data[f'generated_seed{seed}']=(projected-mean).reshape(len(projected),-1)@components.T
        write_backbone_ensemble(RUN/f'generated_ensemble_seed{seed}.pdb',bb,p['sequence'])
    np.savez_compressed(RUN/'posthoc_CA_projection.npz',**plot_data)
    result=dict(status='complete',scope='Posthoc descriptive only; original capacity checks remain failed. One native sequence, reused R2; no physical-mechanism proof or independent validation.',gauge=dict(canonical_CA_RMS_fluctuation_A=float(np.sqrt(before)),rigid_aligned_CA_RMS_fluctuation_A=float(np.sqrt(after)),fraction_coordinate_variance_removed_by_rigid_alignment=1-after/before,interpretation='The first-residue canonical frame can add apparent whole-protein orientation variation as that residue moves. This is a target-representation diagnostic, not proof it caused this generator failure.'),valid_native_point_control=dict(selected_R1_case=case['id'],selection='Nearest R1 frame to the mean of globally aligned TRAIN CA coordinates; no R2 selection.',metrics=point,interpretation='A valid deterministic structure control. The primary decoded mean-latent control can leave the nonlinear decoder manifold, so beating it does not prove superiority over deterministic predictors.'),visualization_PCA_train_variance_fraction=plot_data['train_PC_variance_fraction'].tolist(),all_64_samples_per_seed_exported_to_PDB=True,sealed_test_evaluated=False)
    atomic_json(RUN/'posthoc_geometry_analysis.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
