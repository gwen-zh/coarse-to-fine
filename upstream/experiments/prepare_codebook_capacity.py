"""Freeze a train-only, matched fixed-latent codebook-size protocol."""
import ast
import json
from pathlib import Path
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

RUN=Path('artifacts/codebook_capacity_v1')

def main():
    RUN.mkdir(exist_ok=True)
    if (RUN/'protocol.json').exists():raise RuntimeError('Capacity protocol already frozen')
    parent_path=Path('artifacts/mixed_hierarchical_followup_v1/protocol.json');parent=json.loads(parent_path.read_text())
    repair=json.loads(Path('artifacts/mixed_generator_repair_v1/protocol.json').read_text())
    sizes=[1024,2048,4096];seeds=[2026,2027];sources=dict(parent['source_sha256']);files=[parent_path,Path('artifacts/mixed_generator_repair_v1/protocol.json')]
    for name in ['codebook_capacity.py','run_codebook_capacity.py','prepare_codebook_capacity.py']:
        path=Path('upstream/experiments')/name;ast.parse(path.read_text());files.append(path)
    files += [Path('upstream/tests/test_codebook_capacity.py'),Path('upstream/scripts/codebook_capacity.sbatch'),Path('paper/protocols/codebook_capacity_zh.rst')]
    coarse_hashes={}
    for seed in seeds:
        for key in ['warmup','final']:files.append(Path(parent['tokenizers'][str(seed)][key]))
        for name in repair['generation_probe_ids']:
            for j in range(8):
                path=Path('artifacts/mixed_hierarchical_followup_v1')/'fine'/f'seed{seed}'/'final_samples'/f'{name}_sample{j}.npz'
                coarse_hashes[str(path)]=sha(path)
        labels=Path('artifacts/mixed_hierarchical_generator_v1/labels')/f'seed{seed}'
        for leaf in ['manifest.json','stats.json']:files.append(labels/leaf)
    sources.update({str(path):sha(path) for path in files})
    p=dict(version=1,seeds=seeds,sizes=sizes,threads=4,parent_protocol=str(parent_path),parent_labels='artifacts/mixed_hierarchical_generator_v1/labels',
        coarse_source_run='artifacts/mixed_hierarchical_followup_v1',coarse_cache_sha256=coarse_hashes,
        fit=dict(pool_vectors=65536,iterations=10,chunk=2048),
        fine_model=dict(condition_dim=1536,coarse_dim=16,width=128,layers=2,dropout=.1),
        training=dict(epochs=12,lr=3e-4,min_lr=3e-5,weight_decay=.01,jitter_std=.05,label_smoothing=.05,evaluation_epochs=[1,3,6,12]),
        sampling=dict(steps=16,samples=8,native_samples=2,temperature=1.),
        train_probe_ids=repair['train_probe_ids'],validation_ids=repair['validation_ids'],generation_probe_ids=repair['generation_probe_ids'],
        target_counts=dict(train=899,development=101),
        objective='All K refit from one identical train-only continuous-fine latent pool per seed,not quantized existing codes. Ten Lloyd passes,nested initial centers,empty-cluster restarts. Entire original structure networks stay fixed;original1024book retained separately from fresh1024control.',
        paired_training='Fresh fine models per K;common parameters initialized exactly from a seeded4096-class template. Shared ordinary row prefixes and sharedMASKrow;class count changes label semantics andhead/embeddingparametercount. Same data order,per-update masks/coarsejitter/dropout seeds;12epochs all899,no warmstart,bestepoch or automatic extension. Native matchingcoarse with normalized.05jitter;no unmatched sampledcoarse/nativefine pairs duringtraining.',
        evaluation='Reconstruction on101reuseddev plus8trainprobes;nativecoarse50%maskedCE on8train/101dev. Actual fullfine generation16reuseddevx8draws with exactlycachedoldgeneratedcoarse;nativecoarse16x2 is oracle,not deployment. No position supportfilter or reference/geometry acceptance. Report all physicalmetrics,spread andunseen slotcombinations,allK/bothseeds. RawCE acrossdifferentvocabs isnotdirectlycomparable. Readbookusingvectors directlyintoownfrozendecoder,nativebooknotmutated.',
        scope='Development experiment conditional on latent space previously learned with K1024. Not end-to-endK-specifictokenizer retraining,independentconfirmation,same-rateauthorcomparison or Boltzmannensembletest. Coarsecontinuouspath unchanged;largerfinevocab raises nominalfinebits10→11→12andparametercount/actualcompute. Equalepochs/data are not equalFLOPs. No automatedadoption or touchingconsumed58/64tests.',
        source_sha256=sources)
    atomic_json(RUN/'protocol.json',p);print('CAPACITY_PROTOCOL_FROZEN',len(sources),flush=True)

if __name__=='__main__':main()
