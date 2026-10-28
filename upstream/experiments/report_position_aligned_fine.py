"""All final paired alignment arms; no retrospective checkpoint selection."""
import json
from pathlib import Path
from diagnostic_io import atomic_json
from report_mixed_generator_repair import delta
from train_position_aligned_fine import RUN, protocol


def main():
    p=protocol();rows=[];pairs=[]
    baselines=json.loads(Path(p['baseline_comparison']).read_text())['rows']
    lines=['fine显式位置与coarse对应：固定预算结果','='*48,'',p['scope'],'',
           '同16复用开发蛋白，每蛋白8样本；旧coarse完全固定，所有输出直接评估，无参考选择或几何接受。',
           'seed / arm / CA RMSD(Å) / CN(%) / CA clashes per residue / spread(Å)','']
    for seed in p['seeds']:
        old=next(r for r in baselines if r['seed']==seed and r['head']=='original' and not r['position_support'])
        supported=next(r for r in baselines if r['seed']==seed and r['head']=='original' and r['position_support'])
        initial=next(r for r in baselines if r['seed']==seed and r['head']=='ce' and r['position_support'])
        complete={arm:json.loads((RUN/arm/f'seed{seed}'/'complete.json').read_text()) for arm in p['arms']}
        traces={arm:json.loads((RUN/arm/f'seed{seed}'/'trace.json').read_text()) for arm in p['arms']}
        for key in ['training_order_sha256','torch_rng_sha256']:
            values=[[r[key] for r in traces[arm]] for arm in p['arms']]
            if not all(x==values[0] for x in values):raise RuntimeError('Paired training randomness differs: '+key)
        pairs.append(dict(seed=seed,same_order_and_global_rng_every_epoch=True))
        variants=[('original_unconstrained',old),('original_supported',supported),('small_ce_supported_before_training',initial)]
        variants += [(arm,complete[arm]['evaluation']) for arm in p['arms']]
        for arm,result in variants:
            m=result['metrics'];row=dict(seed=seed,arm=arm,metrics=m,spread=result['spread'],targets=result['targets'],
                                        versus_original=delta(result,old),versus_original_supported=delta(result,supported))
            if arm in p['arms']:
                row['versus_trained_support_only']=delta(result,complete['support_only']['evaluation'])
                row['adapter_audit']=complete[arm]['adapter_audit']
                row['parameters']=complete[arm]['parameters']
            rows.append(row)
            lines.append(f"{seed} / {arm} / {m['ca_rmsd']:.5f} / {100*m['cn_rate']:.3f} / {m['ca_clashes_per_residue']:.5f} / {result['spread']:.5f}")
    limitations=('Exploratory reused development,not independent confirmation. Primary contrast is position_coarse versus '
                 'support_only with identical warmstart/common parameters,updates,optimizer,smoothing,jitter andrandomness;'
                 'position_only isolates slot embedding. The former60epochlargegenerator and12epochsmallwarmstart '
                 'differ in traininghistory/capacity. All12additionalepochs are explicit newobjective adaptation,not '
                 'an extension/relabeling of previous fixed schedules. Descriptive protein-bootstrap intervals do not '
                 'establish significance after repeateddevelopmentuse. Coarse-globalaccuracy remains unchanged by design.')
    atomic_json(RUN/'comparison.json',dict(status='complete',rows=rows,paired_training_audit=pairs,limitations=limitations))
    lines+=['',limitations,'','Nativecoarse samples are oracle diagnostics and are excluded from deployable main comparisons.']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    state_path=Path('artifacts/run_state.json');state=json.loads(state_path.read_text());state.setdefault('position_aligned_fine_v1',{}).update(status='complete',report=str(RUN/'results_zh.rst'));atomic_json(state_path,state)
    print('POSITION_ALIGNED_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
