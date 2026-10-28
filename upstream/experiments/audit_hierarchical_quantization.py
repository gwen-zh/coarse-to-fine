"""Post-hoc same-weight coarse/fine quantization factorial; no parameter fitting.

Warmup codebooks replay original training-only initialization. Oracle variants
read true structure exclusively for diagnosis and cannot be deployed.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from hierarchical_pipeline_io import RUN,protocol,load_items,sha
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_latent_contract import pool_ca,compose_levels,remove_block_translation
from train_hierarchical_tokenizer import initialize_books
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json

OUT=RUN/'quantization_audit_v1'


@torch.no_grad()
def main():
 p=protocol();audit=json.loads((OUT/'protocol.json').read_text())
 for path,digest in audit['checkpoint_sha256'].items():
  if sha(path)!=digest:raise RuntimeError('Checkpoint changed '+path)
 torch.set_num_threads(1);items=load_items(p);train=[x for x in items if x['row']['subset']=='train'];val=[x for x in items if x['row']['subset']=='validation'];rows=[];timings=[];started=time.perf_counter()
 for seed in p['seeds']:
  for stage,filename in [('warmup','continuous.pt'),('final','final.pt')]:
   path=RUN/'tokenizer'/f'seed{seed}'/filename;saved=torch.load(path,map_location='cpu',weights_only=False)
   model=HierarchicalRigidVQ(**saved['model_config']);model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
   if stage=='warmup':
    torch.set_rng_state(saved['torch_rng']);initialize_books(model,train,torch.device('cpu'),p['tokenizer']['initialization_proteins'],p['tokenizer']['kmeans_iterations'])
   stage_started=time.perf_counter();directory=OUT/f'{stage}_{seed}';directory.mkdir(exist_ok=True)
   for item in val:
    x=item['bb'];c,f,target=model.encode_latents(x);qc=model.coarse_book.codes[model.coarse_book.nearest(c)];qf=model.fine_book.codes[model.fine_book.nearest(f)]
    target_centers=pool_ca(target);predictions={}
    for name,cv,fv in [('continuous_both',c,f),('quantized_coarse',qc,f),('quantized_fine',c,qf),('quantized_both',qc,qf)]:predictions[name]=model.decode(cv,fv,len(x))
    # Override only the predicted centers, leaving learned fine decoding unchanged.
    handle=model.coarse_decoder.register_forward_hook(lambda module,args,output:target_centers/10.)
    try:predictions['oracle_centers_quantized_fine']=model.decode(qc,qf,len(x))
    finally:handle.remove()
    oracle_fine=compose_levels(predictions['quantized_both']['coarse'],remove_block_translation(target))
    records={}
    for name,pred in predictions.items():
     records[name]=dict(metrics=metrics(pred['backbone'].numpy(),x.numpy()),center_rmse=float((pred['coarse']-target_centers).square().sum(-1).mean().sqrt()))
    records['oracle_fine_quantized_centers']=dict(metrics=metrics(oracle_fine.numpy(),x.numpy()),center_rmse=records['quantized_both']['center_rmse'])
    rows.append(dict(seed=seed,stage=stage,target=item['row']['id'],length=len(x),methods=records,coarse_latent_mse=float((c-qc).square().mean()),fine_latent_mse=float((f-qf).square().mean())))
   timings.append(dict(seed=seed,stage=stage,seconds=time.perf_counter()-stage_started));print('QUANTIZATION_AUDIT_STAGE',seed,stage,flush=True)
 summaries=[]
 for seed in p['seeds']:
  for stage in ['warmup','final']:
   selected=[r for r in rows if r['seed']==seed and r['stage']==stage]
   if len(selected)!=101 or len({r['target'] for r in selected})!=101:raise RuntimeError('Incomplete paired diagnostic')
   for name in selected[0]['methods']:
    summaries.append(dict(seed=seed,stage=stage,method=name,metrics={k:float(np.mean([r['methods'][name]['metrics'][k] for r in selected])) for k in selected[0]['methods'][name]['metrics']},center_rmse=float(np.mean([r['methods'][name]['center_rmse'] for r in selected]))))
 # Reproduce the frozen deployed endpoint and the continuous20epoch endpoint exactly.
 for seed in p['seeds']:
  for stage,method,relative in [('warmup','continuous_both','phase0_epoch20/evaluation.json'),('final','quantized_both','final_evaluation/evaluation.json')]:
   expected=json.loads((RUN/'tokenizer'/f'seed{seed}'/relative).read_text())['groups']['validation']['metrics']
   current=next(r['metrics'] for r in summaries if r['seed']==seed and r['stage']==stage and r['method']==method)
   for key in ['ca_rmsd','cn_rate','ca_clashes_per_residue']:
    if current[key]!=expected[key]:raise RuntimeError('Native endpoint metric mismatch '+str((seed,stage,key)))
 result=dict(status='complete',summaries=summaries,targets=rows,timings=timings,seconds=time.perf_counter()-started,scope=audit['scope'],endpoint_replay_exact=True)
 atomic_json(OUT/'comparison.json',result)
 lines=['两级量化误差拆分（不重训）','='*38,'',audit['scope'],'','seed / checkpoint / mode / CA RMSD / CN(%) / boundary CN(%) / center RMSE','']
 for row in summaries:
  m=row['metrics'];lines.append(f"{row['seed']} / {row['stage']} / {row['method']} / {m['ca_rmsd']:.4f} / {100*m['cn_rate']:.2f} / {100*m['boundary_cn_rate']:.2f} / {row['center_rmse']:.4f}")
 lines+=['','Oracle含真实结构，只是诊断。四个量化模式在每个checkpoint内共享完全相同的encoder/decoder权重；但连续/混合输入对最终硬量化训练的decoder可能是分布外，因此不能当作可部署收益。Warmup码本仅重放原训练集初始化，没有decoder/encoder优化。']
 (OUT/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('QUANTIZATION_AUDIT_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
