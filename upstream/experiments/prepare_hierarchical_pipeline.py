"""Freeze full-corpus training and automatic downstream stages before first run."""
import ast,json,hashlib
from pathlib import Path
from hierarchical_pipeline_io import RUN,sha
from diagnostic_io import atomic_json


def main():
 RUN.mkdir(exist_ok=True)
 if (RUN/'protocol.json').exists():raise RuntimeError('Protocol exists; use a new version')
 corpus=Path('artifacts/pdb_training_v2');manifest=corpus/'manifest.json';data=json.loads(manifest.read_text())
 feature_root=Path('artifacts/development/coordinate_flow_v1/sequence_features');targets=[];sources=[str(manifest),str(corpus/'protocol.json')]
 old=json.loads(Path('artifacts/development/two_level_autoencoder_v1/protocol.json').read_text())
 for row in sorted(data['targets'],key=lambda r:hashlib.sha256(('rigid-vq-full-v1:'+r['id']).encode()).hexdigest()):
  reference=corpus/row['backbone'];feature=feature_root/(row['id']+'.npz')
  if not reference.exists() or not feature.exists():raise RuntimeError('Missing paired data '+row['id'])
  targets.append(dict(id=row['id'],subset=row['subset'],length=row['length'],sequence_hash=row['sequence_hash'],reference=str(reference),features=str(feature)))
  sources.extend([str(reference),str(feature)])
 assert sum(r['subset']=='train' for r in targets)==899 and sum(r['subset']=='validation' for r in targets)==101
 code=['hierarchical_rigid_vq.py','hierarchical_token_generator.py','hierarchical_pipeline_io.py','train_hierarchical_tokenizer.py','train_hierarchical_generator.py','report_hierarchical_pipeline.py','prepare_hierarchical_pipeline.py','hierarchical_latent_contract.py','two_level_autoencoder.py','compare_two_level_ae_author.py','structural_metrics.py','backbone_quality_audit.py','diagnostic_io.py','cache_coordinate_sequence.py']
 for name in code:
  path=Path('upstream/experiments')/name;ast.parse(path.read_text());sources.append(str(path))
 sources.extend(str(Path('upstream/scripts')/f'hierarchical_pipeline_{name}.sbatch') for name in ['smoke','tokenizer','generator','report'])
 sources.extend(['upstream/tests/test_hierarchical_rigid_vq.py','upstream/tests/test_hierarchical_token_generator.py'])
 sources.extend(str(path) for path in Path('artifacts/development/coordinate_flow_v1').glob('sequence_chunk*.json'))
 sources.extend(['artifacts/development/coordinate_flow_v1/protocol.json','artifacts/development/two_level_autoencoder_v1/protocol.json'])
 p=dict(targets=targets,seeds=[2026,2027],train_probe_ids=[r['id'] for r in old['targets'] if r['subset']=='train'],generation_probe_ids=[r['id'] for r in targets if r['subset']=='validation'][:16],
 tokenizer_model=dict(width=128,layers=4,coarse_dim=16,fine_dim=8,coarse_size=256,fine_size=1024,block_size=4),
 tokenizer=dict(threads=1,warmup_epochs=20,quantized_epochs=80,lr=3e-4,min_lr=3e-5,initialization_proteins=512,kmeans_iterations=10,quantization_ramp_epochs=10,evaluate_every=5),
 weights=old['weights'],generator_model=dict(condition_dim=1536,width=256,layers=6,coarse_size=256,fine_size=1024,block_size=4),
 generator=dict(threads=4,epochs=60,lr=3e-4,min_lr=3e-5,generated_coarse_after_epoch=10,generated_coarse_probability=.2,training_coarse_steps=4,evaluate_every=10,sampling_steps=16,temperature=1.,final_samples=8),
 scope='新训练主线：899train/101已使用开发蛋白，独立结构encoder/decoder、两个真实分工的EMA码本，之后训练纯序列条件两级mask生成器；不声称已超过作者。',
 representation='Canonical first-residue frame removes arbitrary orientation. Coarse encoder reads only block CA centers with global context,ceil(N/4)x16. Fine encoder reads own-block zero-centroid residual only,Nx8. Decoder may read immediate neighboring fine blocks, outputs per-residue proper rigid poses on physical N/CA/C template. PH=0 keeps coarse centers independent of fine. Distinct256/1024codebooks see different features, not oneESMz. No ESM structure weights or decoder.',
 quantization='Continuous20epochs; train-only Euclidean kmeans initialization512trainingproteins,10iterations. JointEMA VQ80epochs,decay.99,commitment.25,dead restart belowEMAcount.01. First10quantizedepochs gradually blend toward hardcodes; evaluation always hard. Final70epochs fully hard. EMA frozen at evaluation/export. Shared models excluded from nominal rate;8ceil(N/4)+10Nbits,12bits/residue for even4-block lengths,otherwise<=12+6/N.',
 objectives='Reuse normalized coarse/fine/full coordinate losses,bond and angle terms,CAclash hinge,latent regularizer;addVQcommitment. No failed boundary-only extra loss. Fixed cosine learning rates per phase,AdamWwd1e-4,clip1;no best-checkpoint selection.',
 generator_condition='Pre-existing deterministic1536D frozen ESM3 features computed from amino-acid tokens only; structure encoder is fully ours. Frozen two-level tokens are training labels. No true structure,teacher coarse or fine is supplied to sequence-only sampling. Coarse denoising always masks all fine inputs; fine denoising uses variable masks and real fine input embeddings. Startingepoch11,20%fineupdates use independently sampled coarse conditions.',
 readiness='Before production generation training: complete100epochs and immutable1000tokenexport;allfinite;centroiderror<=2e-5A;rigiderror<=2e-4A;development coarseused>=8 and fineused>=16;developmentCA_RMSD<=3A,CN<=10%;zeroing each latent worsens own error>=5%. This new readiness gate does not override earlier failed gates or imply superiority. If not met, downstream writes blocked.json and no generator trains.',
 automation='Slurm smoke -> 2seed tokenizer array -> corresponding seed generator aftercorr -> final report. Both production stages may explicitly resume last complete epoch with full Adam/Torch/NumPy RNG state. No automatic hyperparameter scan or extra epochs after fixed schedules.',
 limitations='Static X-ray training data are not thermodynamic ensembles;101development objects have been reused, not independent confirmation. Initial generator sample evaluation16preselecteddevelopment proteins x8draws is descriptive,not author-matched. Changed model/data/canonicalization/quantization prevent attributing benefit solely to rigid output.12bitnominal code count does not equal measured entropy or general compression.',
 source_sha256={s:sha(s) for s in dict.fromkeys(sources)})
 atomic_json(RUN/'protocol.json',p);print('HIERARCHICAL_PIPELINE_PROTOCOL_READY',len(targets),len(p['source_sha256']),flush=True)


if __name__=='__main__':main()
