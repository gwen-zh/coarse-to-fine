import json
from pathlib import Path
from diagnostic_io import atomic_json
RUN=Path('artifacts/mixed_hierarchical_followup_v1')


def main():
 p=json.loads((RUN/'protocol.json').read_text());rows=[];lines=['混合生成器固定完整预算后续实验','='*40,'',p['scope'],'','原5epoch工程门槛失败，记录保留；本轮不是把原门槛改成通过。','']
 for seed in p['seeds']:
  original=json.loads((Path(p['label_run'])/'pilot'/f'seed{seed}'/'complete.json').read_text());stages={}
  for stage in ['coarse','fine']:
   path=RUN/stage/f'seed{seed}'/'complete.json';stages[stage]=json.loads(path.read_text()) if path.exists() else dict(status='incomplete_or_failed')
  rows.append(dict(seed=seed,original_pilot=original,stages=stages));lines.append(f"seed{seed}:originalpilot={original['pass']},coarse={stages['coarse']['status']},fine={stages['fine']['status']}")
  if stages['fine']['status']=='complete':lines.append('  sequence-only generation: '+json.dumps(stages['fine']['evaluation']))
 result=dict(status='complete',scope=p['scope'],seeds=rows,limitations=p['limitations'],all_generators_trained=all(r['stages']['fine']['status']=='complete' for r in rows))
 atomic_json(RUN/'comparison.json',result);lines+=['',p['limitations']];(RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
 print('MIXED_FOLLOWUP_REPORT_COMPLETE',result['all_generators_trained'],flush=True)


if __name__=='__main__':main()
