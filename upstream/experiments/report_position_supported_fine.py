"""Every fixed position-support intervention and no-model position prior."""
import json
from pathlib import Path
from diagnostic_io import atomic_json
from report_mixed_generator_repair import delta
RUN=Path('artifacts/mixed_generator_position_support_v1')


def main():
    p=json.loads((RUN/'protocol.json').read_text());rows=[]
    lines=['fine块内位置支持：两个种子的固定采样对照','='*46,'',p['scope'],'',
           '所有行都使用同一旧coarse、同16复用开发蛋白、每蛋白8样本。结构不经过几何门控、参考筛选或回退。',
           'seed / fine / position support / CA RMSD(Å) / CN(%) / CA clashes per residue / spread(Å)','']
    for seed in p['seeds']:
        complete=json.loads((RUN/f'seed{seed}'/'complete.json').read_text())
        old=json.loads((Path(p['parent_run'])/'fine'/f'seed{seed}'/'final_samples/evaluation.json').read_text())
        baselines={'original':old}
        for head in ['ce','metric']:
            root=Path(p['repair_run'])/f'fine_{head}'/f'seed{seed}'
            baselines[head]=json.loads((root/'complete.json').read_text())['evaluations']['original']
        variants=[(head,False,r) for head,r in baselines.items()]
        variants += [(head,True,r) for head,r in complete['arms'].items()]
        for head,support,result in variants:
            m=result['metrics'];row=dict(seed=seed,head=head,position_support=support,metrics=m,spread=result['spread'],
                                        versus_original=delta(result,old),targets=result['targets'])
            if support and head in baselines:row['versus_same_head_without_support']=delta(result,baselines[head])
            rows.append(row)
            lines.append(f"{seed} / {head} / {support} / {m['ca_rmsd']:.4f} / {100*m['cn_rate']:.2f} / {m['ca_clashes_per_residue']:.5f} / {result['spread']:.4f}")
    limitations=('Post-hoc exploratory development study motivated by tokenizer-position audit. No independent-test or '
                 'author-matched claim. Train-only support can exclude valid unseen pairs on future proteins. '
                 'All candidates and bothseeds reported, no bestsample/epoch filtering. Position prior has zero '
                 'fine-model forwards, supported model retains16forwards;prior is a mechanistic control,not '
                 'matched inferencecompute. Descriptive bootstrap intervals across16proteins are not '
                 'confirmatory tests;multipleconfigurations and repeateddevelopment use preclude such a claim.')
    atomic_json(RUN/'comparison.json',dict(status='complete',rows=rows,limitations=limitations))
    lines+=['',limitations]
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    state_path=Path('artifacts/run_state.json');state=json.loads(state_path.read_text())
    state.setdefault('mixed_generator_position_support_v1',{}).update(status='complete',run=str(RUN),report=str(RUN/'results_zh.rst'))
    atomic_json(state_path,state);print('POSITION_SUPPORT_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
