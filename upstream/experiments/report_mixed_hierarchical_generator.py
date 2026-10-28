"""Report fixed mixed-generation stages, including failed learnability screens."""
import json
from mixed_hierarchical_io import RUN
from diagnostic_io import atomic_json


def main():
 p=json.loads((RUN/'protocol.json').read_text());seeds=[];lines=['连续coarse＋离散fine：序列生成训练','='*42,'',p['scope'],'']
 for seed in p['seeds']:
  pilot=json.loads((RUN/'pilot'/f'seed{seed}'/'complete.json').read_text());stages={}
  for name in ['coarse','fine']:
   root=RUN/name/f'seed{seed}'
   if (root/'complete.json').exists():stages[name]=json.loads((root/'complete.json').read_text())
   elif (root/'blocked.json').exists():stages[name]=json.loads((root/'blocked.json').read_text())
   else:stages[name]=dict(status='incomplete_or_failed')
  seeds.append(dict(seed=seed,pilot=pilot,stages=stages));lines.append(f"seed{seed}:pilot pass={pilot['pass']};coarse={stages['coarse']['status']};fine={stages['fine']['status']}")
  if stages['fine']['status']=='complete':lines.append('  sequence-only generated metrics: '+json.dumps(stages['fine']['evaluation']))
 result=dict(status='complete',seeds=seeds,all_fine_generators_trained=all(x['stages']['fine']['status']=='complete' for x in seeds),scope=p['scope'],limitations=p['limitations'])
 atomic_json(RUN/'comparison.json',result);lines+=['',p['limitations'],'','Coarse pilot比较训练/开发固定噪声采样、均值基线和错序列条件，不以FM loss替代生成质量。Fine teacher-coarse CE单列。所有完整样本均由序列特征和随机噪声生成，没有真实coarse/fine输入或参考回退。']
 (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n');print('MIXED_GENERATOR_REPORT_COMPLETE',result['all_fine_generators_trained'],flush=True)


if __name__=='__main__':main()
