"""Freeze the eight-protein coarse fitting protocol before any updates."""
import ast
import json
from pathlib import Path
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN=Path('artifacts/coarse_smallset_probe_v1')

def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Smallset protocol already frozen')
    parent_path=Path('artifacts/mixed_hierarchical_followup_v1/protocol.json');parent=json.loads(parent_path.read_text())
    cond_path=Path('artifacts/coarse_conditioning_audit_v1/protocol.json');cond=json.loads(cond_path.read_text())
    sources=dict(parent['source_sha256']);files=[parent_path,cond_path,Path('artifacts/coarse_conditioning_audit_v1/review.json'),Path('artifacts/codebook_capacity_v1/review.json')]
    for name in ['coarse_smallset.py','run_coarse_smallset.py','prepare_coarse_smallset.py','coarse_solver_audit.py']:
        path=Path('upstream/experiments')/name;ast.parse(path.read_text());files.append(path)
    files += [Path('upstream/tests/test_coarse_smallset.py'),Path('upstream/scripts/coarse_smallset.sbatch'),Path('paper/protocols/coarse_smallset_probe_zh.rst')]
    warmstart={}
    for seed in [2026,2027]:
        root=Path('artifacts/mixed_hierarchical_followup_v1/coarse')/f'seed{seed}'
        if json.loads((root/'complete.json').read_text())['epochs']!=60:raise RuntimeError('Coarse warmstart not complete')
        warmstart[str(seed)]=str(root/'final.pt');files.extend([root/'final.pt',root/'complete.json'])
        labels=Path('artifacts/mixed_hierarchical_generator_v1/labels')/f'seed{seed}'
        files.extend([labels/'manifest.json',labels/'stats.json'])
    sources.update({str(path):sha(path) for path in files})
    p=dict(parent_protocol=str(parent_path),coarse_parent_run=cond['parent_run'],seeds=[2026,2027],arms=['flow','regression'],threads=4,
        warmstart=warmstart,train_ids=cond['train_probe_ids'],development_ids=cond['development_probe_ids'],cache_sha256=cond['cache_sha256'],
        updates=1024,checkpoints=[128,512,1024],samples=2,sampling_steps=16,lr=3e-4,min_lr=3e-5,weight_decay=1e-4,geometry_weight=.1,
        fit_screen=dict(train_mean_center_rmsd_at_most=2.,relative_reduction_at_least=.75),unit_tests_passed=5,
        paired='Same original60epochcheckpoint per seed,Adam reset,only8trainingIDs,normalization retained from899train;fixed1024updates,common sampleorder/noise/time per step. Geometryweight.1and t>=.5gate matched. Flow existing velocity objective;regression same network atzero state/time predicts native latent. Extra nonlinearobjectiveandinference differences disclosed;equalupdates not equalFLOPs.',
        scope='Small-set fitting diagnosis,not a final generation upgrade. Structure/finefrozen.8previouslyseen trainingproteins,16reuseddev,separate. ProperKabschweightedtrueblock-centerRMSD,notfullCA. Pointregression haszero ensemble spread andone inference call;not an ensemble replacement. Bothseeds/allfixedsnapshots,primaryfinalpoint;no moresteps,automaticpromotion or58/64independenttest use.',source_sha256=sources)
    atomic_json(RUN/'protocol.json',p);print('COARSE_SMALLSET_PROTOCOL_FROZEN',len(sources),flush=True)

if __name__=='__main__':main()
