"""TRAIN-defined native fine-token motif atlas and position-controlled DEV probes.

No model weights updated. Reused development, descriptive diagnostics only.
"""
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from representation_motif_geometry import fragment_descriptors,fit_templates,predict_templates,circular_error_degrees

RUN=Path('artifacts/representation_motif_atlas_v1')
PARENT=Path('artifacts/mixed_hierarchical_generator_v1/protocol.json')
SEEDS=[2026,2027]


def concat(rows,key):return np.concatenate([r[key] for r in rows],axis=0)


def summarize_points(values,protein,names):
    return np.array([values[protein==i].mean(axis=0) for i in range(len(names))])


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'results.json').exists():raise RuntimeError('Completed diagnostic exists; do not silently overwrite')
    p=json.loads(PARENT.read_text());rows=sorted(p['targets'],key=lambda x:x['id'])
    sources=[str(PARENT),__file__,'upstream/experiments/representation_motif_geometry.py','upstream/tests/test_representation_motif_geometry.py','artifacts/representation_visualization_v1/results.json','artifacts/representation_visualization_v1/example_seed2026.npz']
    for row in rows:
        sources.append(row['reference'])
        for seed in SEEDS:sources.append(f'artifacts/mixed_hierarchical_generator_v1/labels/seed{seed}/{row["id"]}.npz')
    for seed in SEEDS:sources+=list(p['tokenizers'][str(seed)].values())
    protocol=dict(status='fixed_before_analysis',seeds=SEEDS,train_proteins=899,development_proteins=101,
        local_shape='Central residue CA origin and N/CA/C frame; complete three-residue native backbone fragment. Shape error is RMS of the SIX neighbor atoms, excluding the frame-defining central atoms.',
        angles='Central phi(Cprev,N,CA,C) and psi(N,CA,C,Nnext), circular means and circular absolute errors.',
        coverage='All interior residues, excluding exactly first/last residue of every chain. No geometry/outlier filtering. Any undefined descriptor aborts analysis.',
        aggregation='First average per-residue error within protein, then average proteins equally; seeds reported separately.',
        models=['Slot-only TRAIN mean','Code+slot TRAIN mean','Within-slot shuffled-code TRAIN mean'],
        unseen_pairs='Fall back to the TRAIN slot-only mean; no dev fitting.',
        shuffle='For each seed, permute TRAIN code labels independently within each slot using NumPy generator 57500+seed. Preserve exact TRAIN code/slot counts; no DEV shuffling.',
        atlas_selection='For each slot, select top2 codes by TRAIN interior-residue frequency among codes with >=20 TRAIN occurrences across >=5 TRAIN proteins. Tie-break ascending code ID. No DEV criterion.',
        atlas_display='At most12 DEV examples per code, first occurrence per distinct protein, ordered by protein ID. No RMSD/angle selection.',
        uncertainty='95percent paired bootstrap intervals by reused-DEV protein,5000resamples, descriptive only; not independent confirmation.',
        limitations=['Native-structure encoding,not sequence-only generation.','Coarse/fine intervention magnitudes differ and cannot compare representation importance.','Fragment means are statistical descriptors,not decoded physical structures.','A single fine token need not determine a fragment; context and neighboring tokens remain relevant.','Position control does not establish statistical independence or learned topology.'],
        source_sha256={name:sha(name) for name in sources})
    atomic_json(RUN/'protocol.json',protocol)
    print('PROTOCOL_FIXED',flush=True)
    geometry={'train':[],'validation':[]};names={'train':[],'validation':[]};coverage=[]
    for row in rows:
        with np.load(row['reference']) as z:bb=z['backbone'].astype(np.float64)
        if bb.shape!=(row['length'],3,3):raise RuntimeError('Reference shape mismatch')
        d=fragment_descriptors(bb);subset=row['subset'];protein_index=len(names[subset]);names[subset].append(row['id'])
        d.update(protein=np.full(len(d['residue']),protein_index,dtype=np.int64),target=row['id'])
        geometry[subset].append(d)
        coverage.append(dict(target=row['id'],subset=subset,length=len(bb),fragments=len(d['residue']),terminal_residues_excluded=2))
    arrays={subset:{key:concat(geometry[subset],key) for key in ['local','angles','slot','protein','residue']} for subset in geometry}
    print('GEOMETRY_READY',[(k,len(v['slot'])) for k,v in arrays.items()],flush=True)
    results=[];atlas=[]
    for seed in SEEDS:
        labels={'train':[],'validation':[]}
        for row in rows:
            with np.load(f'artifacts/mixed_hierarchical_generator_v1/labels/seed{seed}/{row["id"]}.npz') as z:f=z['fine'].astype(np.int64)
            if f.shape!=(row['length'],):raise RuntimeError('Fine label shape mismatch')
            labels[row['subset']].append(f[1:-1])
        code={subset:np.concatenate(v) for subset,v in labels.items()}
        train=arrays['train'];dev=arrays['validation']
        if len(code['train'])!=len(train['local']) or len(code['validation'])!=len(dev['local']):raise RuntimeError('Descriptor-label mismatch')
        fit=fit_templates(train['local'],train['angles'],train['slot'],code['train'])
        shuffled=code['train'].copy();rng=np.random.default_rng(57500+seed)
        for slot in range(4):
            index=np.flatnonzero(train['slot']==slot);shuffled[index]=shuffled[rng.permutation(index)]
        null=fit_templates(train['local'],train['angles'],train['slot'],shuffled)
        np.testing.assert_equal(fit['pair']['count'],null['pair']['count'])
        unique_proteins=np.zeros(4096,dtype=np.int64)
        for i in range(len(names['train'])):
            mask=train['protein']==i;unique_proteins[np.unique(code['train'][mask]*4+train['slot'][mask])]+=1
        methods={};predictions={}
        for method,templates,use_code in [('slot_only',fit,False),('code_and_slot',fit,True),('shuffled_code_within_slot',null,True)]:
            shape,angle=predict_templates(templates,dev['slot'],code['validation'],use_code)
            shape_error=np.sqrt(np.mean(np.sum((shape[:,[0,2]]-dev['local'][:,[0,2]])**2,axis=-1),axis=(1,2)))
            angular_error=circular_error_degrees(angle,dev['angles'])
            point_values=np.column_stack([shape_error,angular_error])
            by_protein=summarize_points(point_values,dev['protein'],names['validation'])
            by_slot=[]
            for slot in range(4):
                mask=dev['slot']==slot
                by_slot.append(summarize_points(point_values[mask],dev['protein'][mask],names['validation']).mean(0).tolist())
            methods[method]=dict(local_shape_rmsd_A=float(by_protein[:,0].mean()),phi_circular_mae_deg=float(by_protein[:,1].mean()),psi_circular_mae_deg=float(by_protein[:,2].mean()),per_slot=by_slot,
                proteins=[dict(target=name,local_shape_rmsd_A=float(v[0]),phi_circular_mae_deg=float(v[1]),psi_circular_mae_deg=float(v[2])) for name,v in zip(names['validation'],by_protein)])
            predictions[method]=by_protein
        boot_rng=np.random.default_rng(3800+seed);boot_index=boot_rng.integers(0,len(names['validation']),(5000,len(names['validation'])))
        delta=predictions['code_and_slot']-predictions['slot_only'];boot=delta[boot_index].mean(1)
        keys=['local_shape_rmsd_A','phi_circular_mae_deg','psi_circular_mae_deg']
        effects={key:dict(code_minus_slot=float(delta[:,j].mean()),descriptive_paired_bootstrap95=np.quantile(boot[:,j],[.025,.975]).tolist(),proteins_lower=int((delta[:,j]<0).sum()),proteins_higher=int((delta[:,j]>0).sum())) for j,key in enumerate(keys)}
        count=fit['pair']['count'].reshape(1024,4);majority=count.argmax(1)
        position_accuracy=summarize_points((majority[code['validation']]==dev['slot']).astype(float),dev['protein'],names['validation'])
        table=count/count.sum();pr=table.sum(1);ps=table.sum(0);denom=pr[:,None]*ps[None,:];valid=table>0
        mi=float((table[valid]*np.log2(table[valid]/denom[valid])).sum())
        selected=[]
        for slot in range(4):
            pairs=4*np.arange(1024)+slot;eligible=(count[:,slot]>=20)&(unique_proteins[pairs]>=5)
            order=np.lexsort((np.arange(1024),-count[:,slot]));chosen=[int(c) for c in order if eligible[c]][:2]
            if len(chosen)!=2:raise RuntimeError('Insufficient TRAIN support for predeclared atlas')
            for rank,c in enumerate(chosen):
                indices=np.flatnonzero((code['validation']==c)&(dev['slot']==slot));example_indices=[];seen=set()
                for index in indices:
                    prot=int(dev['protein'][index])
                    if prot not in seen:example_indices.append(int(index));seen.add(prot)
                    if len(example_indices)==12:break
                record=dict(seed=seed,slot_zero_based=slot,rank=rank,code=c,train_occurrences=int(count[c,slot]),train_proteins=int(unique_proteins[4*c+slot]),dev_occurrences=int(len(indices)),dev_proteins=int(len(np.unique(dev['protein'][indices]))),examples=[dict(target=names['validation'][dev['protein'][idx]],residue_one_based=int(dev['residue'][idx]+1),descriptor_index=idx) for idx in example_indices])
                selected.append(record);atlas.append(record)
        np.savez_compressed(RUN/f'descriptors_seed{seed}.npz',dev_local=dev['local'],dev_angles=dev['angles'],dev_slot=dev['slot'],dev_code=code['validation'],dev_protein=dev['protein'],train_pair_count=fit['pair']['count'],train_pair_local=fit['pair']['local'],train_pair_angles=fit['pair']['angles'],train_slot_local=fit['slot']['local'],train_slot_angles=fit['slot']['angles'])
        result=dict(seed=seed,train_fragments=len(train['slot']),dev_fragments=len(dev['slot']),methods=methods,code_vs_slot_effect=effects,dev_code_slot_pairs_absent_in_train=int((fit['pair']['count'][4*code['validation']+dev['slot']]==0).sum()),train_code_slot_mi_bits=mi,dev_slot_prediction_accuracy=float(position_accuracy.mean()),atlas_selection=selected)
        results.append(result);print('SEED_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['methods','atlas_selection']}),flush=True)
    for name,digest in protocol['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Input changed '+name)
    report=dict(status='complete',train_proteins=len(names['train']),development_proteins=len(names['validation']),seeds=results,coverage=coverage,scope='TRAIN-defined descriptive representation probe on reusedDEV;not a new-model generation improvement or independent test.',protocol_sha256=sha(RUN/'protocol.json'))
    atomic_json(RUN/'results.json',report);atomic_json(RUN/'atlas_selection.json',atlas)
    print('ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
