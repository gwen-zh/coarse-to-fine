"""Freeze ordered mixed-generation learnability screen and full schedules."""
import ast,json
from pathlib import Path
from hierarchical_pipeline_io import sha
from mixed_hierarchical_io import RUN
from diagnostic_io import atomic_json


def main():
 RUN.mkdir(exist_ok=True)
 if (RUN/'protocol.json').exists():raise RuntimeError('Frozen protocol exists')
 parent=Path('artifacts/hierarchical_rigid_pipeline_v1');old=json.loads((parent/'protocol.json').read_text());sources=dict(old['source_sha256'])
 paths=[str(parent/'protocol.json'),str(parent/'continuous_coarse_hybrid_v1/comparison.json'),str(parent/'continuous_coarse_hybrid_v1/protocol.json')]
 tokenizers={}
 for seed in old['seeds']:
  root=parent/'tokenizer'/f'seed{seed}';tokenizers[str(seed)]=dict(warmup=str(root/'continuous.pt'),final=str(root/'final.pt'))
  paths+=list(tokenizers[str(seed)].values())
  paths += [str(parent/'continuous_coarse_hybrid_v1'/f'seed{seed}'/(r['id']+'.npz')) for r in old['targets'] if r['subset']=='validation']
 code=['mixed_hierarchical_generator.py','mixed_hierarchical_io.py','prepare_mixed_hierarchical_labels.py','train_mixed_hierarchical_generator.py','report_mixed_hierarchical_generator.py','sample_mixed_hierarchical.py','prepare_mixed_hierarchical_protocol.py','latent_flow_head.py']
 for name in code:
  path=Path('upstream/experiments')/name;ast.parse(path.read_text());paths.append(str(path))
 paths += [str(Path('upstream/scripts')/f'mixed_hierarchical_{name}.sbatch') for name in ['prepare','smoke','train','report']]
 paths += ['upstream/tests/test_mixed_hierarchical_generator.py']
 sources.update({path:sha(path) for path in paths})
 p=dict(targets=old['targets'],seeds=old['seeds'],tokenizers=tokenizers,hybrid_diagnostic=str(parent/'continuous_coarse_hybrid_v1'),train_probe_ids=old['train_probe_ids'],generation_probe_ids=old['generation_probe_ids'],threads=4,
 model=dict(condition_dim=1536,coarse_dim=16,width=256,coarse_layers=4,fine_layers=6,fine_size=1024),
 coarse=dict(pilot_epochs=5,epochs=60,lr=3e-4,min_lr=3e-5,geometry_weight=.1,evaluate_every=10,sampling_steps=16),
 fine=dict(epochs=60,lr=3e-4,min_lr=3e-5,generated_coarse_after_epoch=10,generated_coarse_probability=.2,training_coarse_steps=8,evaluate_every=10,sampling_steps=16,temperature=1.,final_samples=8),
 scope='用户授权：冻结已验证混合tokenizer，在899train/101reused-development数据上训练纯序列条件连续coarse flow，随后训练条件fine mask生成器。2seeds，先固定5epoch可学性检查，再自动延续全60epoch coarse和60epoch fine。',
 frozen_representation='Warmup20epoch coarse encoder AND decoder plus final80epoch fine encoder/decoder/book. Two distinct latent levels preserved,coarse16continuous scalars/block,fine1024classID/residue. No tokenizer updates. All101prior hybrid labels and decoded backbones must replay exactly.',
 normalization='Protein-equal per-channel mean and second moment from899trainingproteins only,std floor1e-3. Development never fits stats. AA-only features mean-pooled into4residue blocks forcoarse;fullresidue features forfine.',
 coarse_objective='Rectified linear-path conditional flowmatching,v=target-noise,on normalized16Dcoarse. At t>=.5 add0.1*decoded blockcenter MSE/100 through frozen coarse decoder; geometry labels from native coarse encoded training structure,not model input. Standard Gaussian initialnoise and16Eulersteps at deployment,no target latent or coordinate input,no per-sample selection.',
 pilot_gate='Fixed5/60epochs,not extra budget:train8probe centerRMSE at least10percent below initialization;development16probe centerRMSE at least5percent below initialization and <=1.05x train-mean-latent baseline;allfinite. Same noise2draws/protein initial/final. Wrong-sequence condition control reported but not gate. Passing is engineering feasibility,not proof of conditional sequence learning,diversity or improved generation.',
 fine_objective='Variable10-100percent masked fine CE,actualfine inputembedding,continuouscoarse projection preserves channel mean/amplitude. First10epochs nativecoarse;thereafter20percent conditions from frozen learnedcoarseflow8steps. Deployment coarse always from sequence/noise and16steps,never structureencoder. Teachercoarse CE separatefrom unfiltered generatedgeometry/RMSD/spread.',
 automation='Prepare immutable labels -> real-data smoke -> two-seed first5coarseepochs -> aftercorr correspondingfullcoarse resumesepoch5to60 -> aftercorr fine60epochs -> report. If pilot fails, downstream writes explicitblocked.json, no automaticbudget/thresholdchange.',
 statistics='All fixedfinal checkpoints and bothseeds,full raw draws,16preselecteddevelopmentproteinsx8samples;no bestseed/epoch selection,no referenceacceptance or skippedfailures. Mean/shuffledsequencecontrols distinguish prior shrinkage from conditioning only descriptively.',
 limitations='Mixedrepresentation has a different informationrate from12bit fullydiscrete baseline;staticXraytrainingnot thermodynamicensemble. Reuseddevelopment,not independentconfirmation or author-matched superiority. Structuralreconstruction improvement does not establish generativeimprovement. Full60epoch predictionquality may still fail.',
 source_sha256=sources)
 atomic_json(RUN/'protocol.json',p);print('MIXED_GENERATOR_PROTOCOL_READY',len(sources),flush=True)


if __name__=='__main__':main()
