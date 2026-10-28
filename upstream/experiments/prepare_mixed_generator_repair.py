"""Freeze one joint regularization recipe and two fine output-head controls."""
import ast
import json
from pathlib import Path
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from train_mixed_generator_repair import RUN


def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():
        raise RuntimeError('Frozen repair protocol already exists')
    parent=Path('artifacts/mixed_hierarchical_followup_v1')
    old=json.loads((parent/'protocol.json').read_text())
    files=[parent/'protocol.json',parent/'comparison.json',
           Path('artifacts/mixed_generation_failure_audit_v2/comparison.json')]
    for name in ['regularized_mixed_generator.py','train_mixed_generator_repair.py',
                 'report_mixed_generator_repair.py','sample_mixed_generator_repair.py',
                 'prepare_mixed_generator_repair.py']:
        path=Path('upstream/experiments')/name
        ast.parse(path.read_text());files.append(path)
    files.extend([Path('upstream/tests/test_regularized_mixed_generator.py'),
                  Path('upstream/scripts/mixed_generator_repair.sbatch'),
                  Path('upstream/scripts/mixed_generator_repair_report.sbatch')])
    for seed in old['seeds']:
        root=parent/'fine'/f'seed{seed}'
        files.extend([root/'final.pt',root/'final_ce.json',root/'final_samples/evaluation.json'])
        files.extend(root/'final_samples'/f'{name}_sample{i}.npz'
                     for name in old['generation_probe_ids'] for i in range(8))
    sources=dict(old['source_sha256'])
    sources.update({str(path):sha(path) for path in files})
    p=dict(parent_run=str(parent),parent_protocol=str(parent/'protocol.json'),seeds=old['seeds'],
           model=dict(condition_dim=1536,coarse_dim=16,width=128,coarse_layers=2,fine_layers=2,
                      fine_size=1024,dropout=.1),fine_heads=['ce','metric'],weight_decay=.01,
           coarse=dict(epochs=20,lr=3e-4,min_lr=3e-5,geometry_weight=.1,
                       evaluation_epochs=[1,5,10,20],sampling_steps=16),
           fine=dict(epochs=12,lr=3e-4,min_lr=3e-5,label_smoothing=.05,jitter_std=.05,
                     evaluation_epochs=[1,3,6,12],sampling_steps=16),
           final_samples=8,train_probe_ids=old['train_probe_ids'],
           generation_probe_ids=old['generation_probe_ids'],
           validation_ids=[r['id'] for r in old['targets'] if r['subset']=='validation'],
           scope='针对899train/101复用开发蛋白上的生成泛化失败：固定20epoch小coarseflow与12epoch小fine生成器，两个种子，CE与码本距离输出头并列对照。结构encoder/decoder/codebook全部冻结。',
           controls='Final six compositions per seed: original/repaired coarse x original/regularizedCE/metric fine. Same16development proteinsx8draws and original sample seed keys. Fine controls have identical common initialization, training order/masking/dropout random streams, smoothing/jitter/budget; head parameters differ. Old60epoch versus new20/12epoch is not matched training compute.',
           hypothesis='Joint smallerwidth128/two-layer capacity,dropout.1,AdamWwd.01 and .05label smoothing address overfit. Fine conditioning adds .05normalized Gaussian jitter around its matching native coarse,clipped only to structuralencoder tanh domain; no independently sampled conformation paired with original fine labels. Metric head predicts bounded8Dpoint,negative mean squared distance to frozen1024x8codes,learnedtemperature in[.02,.5],initialized.05. No proof that latent distance guarantees decoded geometry.',
           selection='Fixed final epochs for primary report; save/evaluate all predeclared snapshots1/3/6/12 or1/5/10/20. No bestcheckpoint/seed selection or automatic budget extension. Evaluate matched50%mask nativecoarse train/development NLL plus full sequence-only samples;nativecoarse generation is oracle diagnostic only.',
           audit='Original128samples/seed coarse contains no out-of-tanh-range components:2026range[-.69972,.678617],2027[-.944472,.801966]. Do not explain failure by endpoint range violations. Existing feature cache extractor pairsAAsequence andh by manifestID;all referenced immutable hashes reused.',
           interpretation='All physical outputs unfiltered. CN,clashes,angles,RMSD and spread evaluated directly. Descriptive signal target is CN decrease with CAmean deterioration<=.1A and spread>=95%baseline in bothseeds; this is engineering triage,not significance/confirmation. Report every composition and all tradeoffs. No author superiority or equilibrium ensemble claim.',
           source_sha256=sources)
    atomic_json(RUN/'protocol.json',p)
    print('GENERATOR_REPAIR_PROTOCOL_READY',len(sources),flush=True)


if __name__=='__main__':main()
