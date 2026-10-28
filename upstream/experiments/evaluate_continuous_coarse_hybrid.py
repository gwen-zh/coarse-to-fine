"""No-training mixed tokenizer: warmup continuous coarse + final quantized fine.

Both encoder and decoder for coarse are copied together. The fine branch sees
predicted block centers only. This is a different-rate reconstruction diagnostic,
not an oracle, a trained sequence generator or a replacement for old failed gates.
"""
import json
from pathlib import Path
import numpy as np
import torch
from hierarchical_pipeline_io import RUN,protocol,load_items,sha
from hierarchical_rigid_vq import HierarchicalRigidVQ
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json

OUT=RUN/'continuous_coarse_hybrid_v1'


@torch.no_grad()
def main():
 p=protocol();a=json.loads((OUT/'protocol.json').read_text())
 for path,digest in a['sha256'].items():
  if sha(path)!=digest:raise RuntimeError('Frozen source changed '+path)
 torch.set_num_threads(1);items=[x for x in load_items(p) if x['row']['subset']=='validation'];rows=[];summaries=[]
 for seed in p['seeds']:
  directory=RUN/'tokenizer'/f'seed{seed}';warm=torch.load(directory/'continuous.pt',map_location='cpu',weights_only=False);final=torch.load(directory/'final.pt',map_location='cpu',weights_only=False)
  model=HierarchicalRigidVQ(**final['model_config']);model.load_state_dict(final['model'])
  for name in ['coarse_encoder','coarse_decoder']:
   state={k[len(name)+1:]:v for k,v in warm['model'].items() if k.startswith(name+'.')}
   getattr(model,name).load_state_dict(state)
  model.eval().requires_grad_(False);out=OUT/f'seed{seed}';out.mkdir(exist_ok=True)
  for item in items:
   c,f,_=model.encode_latents(item['bb']);fine=model.fine_book.nearest(f)
   bb=model.decode(c,model.fine_book.codes[fine],len(f))['backbone'].numpy()
   row=dict(seed=seed,target=item['row']['id'],metrics=metrics(bb,item['bb'].numpy()),coarse_scalars=c.numel(),fine_discrete_codes=len(fine));rows.append(row)
   np.savez_compressed(out/(row['target']+'.npz'),backbone=bb,coarse_latent=c.numpy(),fine=fine.numpy())
  selected=[r for r in rows if r['seed']==seed]
  if len(selected)!=101 or len({r['target'] for r in selected})!=101:raise RuntimeError('Incomplete paired evaluation')
  summaries.append(dict(seed=seed,proteins=101,metrics={key:float(np.mean([r['metrics'][key] for r in selected])) for key in selected[0]['metrics']}))
 result=dict(status='complete',summaries=summaries,targets=rows,scope=a['scope'],no_parameter_updates=True,no_oracle_coordinates_in_decoder=True,
             limitations='Coarse now transmits16continuous scalars/block, fine1024classID/residue. Different information rate from original12bit/residue,not compression superiority. Reused101development proteins;no sequence generation trained;old readiness gates unchanged.')
 atomic_json(OUT/'comparison.json',result)
 lines=['连续coarse + 离散fine混合重建（不重训）','='*45,'',a['scope'],'']
 for r in summaries:
  m=r['metrics'];lines.append(f"seed{r['seed']}: CA RMSD{m['ca_rmsd']:.4f}Å，backbone RMSD{m['backbone_rmsd']:.4f}Å，CN{100*m['cn_rate']:.2f}%，boundary CN{100*m['boundary_cn_rate']:.2f}%，CA无碰撞率{100*m['ca_clash_valid']:.2f}%。")
 lines+=['',result['limitations']]
 (OUT/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('CONTINUOUS_COARSE_HYBRID_COMPLETE',json.dumps(summaries),flush=True)


if __name__=='__main__':main()
