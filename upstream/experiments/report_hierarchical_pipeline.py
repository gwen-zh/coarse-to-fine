"""Report full tokenizer health and generator outcome without hiding blocked stages."""
import json
from pathlib import Path
from hierarchical_pipeline_io import RUN
from diagnostic_io import atomic_json


def main():
 p=json.loads((RUN/'protocol.json').read_text());seeds=[];lines=['完整两级刚体VQ tokenizer与生成器训练','='*44,'',p['scope'],'']
 for seed in p['seeds']:
  token=json.loads((RUN/'tokenizer'/f'seed{seed}'/'complete.json').read_text());generator=RUN/'generator'/f'seed{seed}'
  if (generator/'complete.json').exists():gen=json.loads((generator/'complete.json').read_text())
  elif (generator/'blocked.json').exists():gen=json.loads((generator/'blocked.json').read_text())
  else:gen=dict(status='incomplete_or_failed')
  seeds.append(dict(seed=seed,tokenizer=token,generator=gen));m=token['validation']['metrics']
  lines.append(f"seed{seed} tokenizer: CA RMSD{m['ca_rmsd']:.4f}Å, CN{100*m['cn_rate']:.2f}%, ready={token['ready_for_generator']}; generator={gen['status']}")
  if gen['status']=='complete':lines.append('  generation metrics: '+json.dumps(gen['generation']['metrics']))
 result=dict(status='complete',seeds=seeds,scope=p['scope'],limitations=p['limitations'],all_generators_trained=all(x['generator']['status']=='complete' for x in seeds))
 atomic_json(RUN/'comparison.json',result)
 lines+=['',p['limitations'],'','Token-only真实离散重建、码本perplexity/利用率、结构约束和分支使用见每个tokenizer的final_evaluation。生成器无参考采样，teacher-coarse CE单独标明，不等同部署质量。']
 (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('HIERARCHICAL_PIPELINE_REPORT',result['all_generators_trained'],flush=True)


if __name__=='__main__':main()
