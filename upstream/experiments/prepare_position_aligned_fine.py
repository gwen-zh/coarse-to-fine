"""Freeze explicit slot/coarse alignment, support-aware objective and all controls."""
import ast,json
from pathlib import Path
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from train_position_aligned_fine import RUN


def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Alignment protocol already frozen')
    previous=Path('artifacts/mixed_generator_position_support_v1');old=json.loads((previous/'protocol.json').read_text());rp=Path('artifacts/mixed_generator_repair_v1');repair=json.loads((rp/'protocol.json').read_text())
    sources=dict(old['source_sha256'])
    files=[previous/'protocol.json',previous/'comparison.json',rp/'comparison.json']
    for name in ['position_aligned_fine.py','train_position_aligned_fine.py','report_position_aligned_fine.py','sample_position_aligned_fine.py','prepare_position_aligned_fine.py']:
        path=Path('upstream/experiments')/name;ast.parse(path.read_text());files.append(path)
    files += [Path('upstream/tests/test_position_aligned_fine.py'),Path('upstream/scripts/position_aligned_fine.sbatch'),Path('upstream/scripts/position_aligned_fine_report.sbatch')]
    warmstart={};counts={};original={}
    for seed in old['seeds']:
        root=rp/'fine_ce'/f'seed{seed}'
        if json.loads((root/'complete.json').read_text())['status']!='complete':raise RuntimeError('Fixed12epoch warmstart incomplete')
        warmstart[str(seed)]=str(root/'final.pt');counts[str(seed)]=str(previous/f'train_counts_seed{seed}.npy');original[str(seed)]=str(Path(old['parent_run'])/'fine'/f'seed{seed}'/'final.pt')
        files += [root/'final.pt',root/'complete.json',Path(counts[str(seed)]),Path(original[str(seed)])]
    sources.update({str(path):sha(path) for path in files})
    p=dict(parent_protocol=old['parent_protocol'],coarse_source_run=old['parent_run'],warmstart=warmstart,counts=counts,old_generator=original,seeds=old['seeds'],arms=['support_only','position_only','position_coarse'],baseline_comparison=str(previous/'comparison.json'),
           model=dict(condition_dim=1536,coarse_dim=16,width=128,layers=2,fine_size=1024,dropout=.1),
           training=dict(epochs=12,lr=3e-4,min_lr=3e-5,weight_decay=.01,jitter_std=.05,label_smoothing=.05,evaluation_epochs=[1,3,6,12]),
           generation_probe_ids=old['generation_probe_ids'],train_probe_ids=repair['train_probe_ids'],validation_ids=repair['validation_ids'],
           scope='用户同意后续训练：固定旧coarse，修正fine训练中的块内位置与对应coarse信息。899train/101复用开发，两个种子，三组fine从同一固定12epoch普通CE检查点开始，新增12epoch，结构encoder/decoder/codebook与coarse全部冻结。',
           contrast='support_only applies train-position support during optimization;position_only additionally adds learned residue_index%4embedding;position_coarse additionally broadcasts coarse_projection(z_c[floor(i/4)]) into eachfineinput through a learned tanh gate. Slot embedding and localcoarsegate zero-initialized for exactlyidentical initialfunctions. All common weights,order,mask/jitter/dropoutstreams andbudgets paired;extras512/513parameters atwidth128.',
           objective='Variable10-100% masked CE,uniform.05smoothing ONLY across training-supported classes. StandardPyTorchlabel_smoothing over -inf forbiddenlogits would be invalid;explicit finite supported loss used. Matchingnativecoarse with.05normalizedjitter;no independently sampledcoarse paired to nativefine. AdamWfreshstate reset shared acrossarms because objective/adapterschange;not a resume of old12epochoptimizer.',
           metrics='Primary finalcomparison position_coarse versus newlytrainedsupport_only;position_only separates slotinput fromlocalcoarsebroadcast. Keep originalunconstrained/originalsupported/smallCEsupportedpretraining baselines. Same16proteinsx8draws withfrozenoldcoarse caches,actualsequence-only fine prediction,unfilteredcoordinates,no reference acceptance. Teacher/nativecoarseNLL and2draw oraclegeometry separate;full101unseenlabels explicitlyreported,never silentlydropped.',
           statistics='Save all fixed1/3/6/12 snapshots,primaryfinalepoch only,allarms/seeds,no bestcheckpoint or budgetextension. Everyepochorder/TorchRNGdigest must match acrossarms. Bootstrap units proteins,intervals descriptive after repeateddevelopmentuse. No independentconfirmation/authoradvantage/thermodynamiccoverage claim. Coarse accuracy is not improved by thisfineonlyexperiment.',
           source_sha256=sources)
    atomic_json(RUN/'protocol.json',p);print('POSITION_ALIGNED_PROTOCOL_READY',len(sources),flush=True)


if __name__=='__main__':main()
