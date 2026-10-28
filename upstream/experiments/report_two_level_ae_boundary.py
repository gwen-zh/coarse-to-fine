"""Describe complete 2x2, both seeds and all fixed checkpoints, without selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json

RUN = Path('artifacts/development/two_level_ae_boundary_v1')
METRICS = ['rmsd', 'cn_rate', 'within_rate', 'boundary_rate', 'clashes_per_residue']
ANGLES = ['peptide_planarity_deviation_degrees', 'reference_omega_circular_mae_degrees',
          'reference_ca_c_n_angle_mae_degrees', 'reference_c_n_ca_angle_mae_degrees']


def original_gate(groups):
    t, v = groups['train'], groups['validation']
    return dict(train_rmsd_below1A=t['metrics']['rmsd']<=1., train_cn_below5pct=t['metrics']['cn_rate']<=.05,
                validation_rmsd_below2A=v['metrics']['rmsd']<=2., validation_cn_below10pct=v['metrics']['cn_rate']<=.10,
                coarse_used=all(x['zero_coarse_rmse']>=1.05*x['coarse_rmse'] for x in [t,v]),
                fine_used=all(x['zero_fine_mse']>=1.05*x['fine_mse'] for x in [t,v]),
                fine_cannot_move_centers=max(t['max_centroid_drift'],v['max_centroid_drift'])<=2e-5)


def main():
    pp = RUN/'protocol.json'; p = json.loads(pp.read_text()); ph = hashlib.sha256(pp.read_bytes()).hexdigest()
    for source, digest in p['sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest: raise RuntimeError('Frozen input changed: '+source)
    parent = json.loads(Path(p['parent_protocol']).read_text()); expected = {r['id']:r['subset'] for r in parent['targets']}
    records = []; evaluations = {}; timings = []; hashes = {}
    for arm, config in p['arms'].items():
        for seed in parent['seeds']:
            if arm=='baseline':
                directory=Path(p['baseline_run'])/f'seed{seed}'
            else:
                directory=RUN/'runs'/arm/f'seed{seed}'
                complete=json.loads((directory/'complete.json').read_text())
                if complete['status']!='complete' or complete['protocol_sha256']!=ph: raise RuntimeError('Incomplete/new-protocol run')
            training=json.loads((directory/'training.json').read_text())
            timings.append(dict(arm=arm,seed=seed,seconds=training['seconds'],parameters=config['parameters']))
            if arm!='baseline': hashes[(arm,seed)]=training['augmentation_sha256']
            for step in p['checkpoints']:
                data=json.loads((directory/f'step{step}/evaluation.json').read_text())
                rows=data['targets']
                if data['status']!='complete' or data['step']!=step or len(rows)!=len(expected) or {r['target']:r['subset'] for r in rows}!=expected:
                    raise RuntimeError('Incomplete pairing; do not omit proteins')
                evaluations[(arm,seed,step)]={r['target']:r for r in rows}
                groups={}
                for subset in ['train','validation']:
                    selected=[r for r in rows if r['subset']==subset]
                    def mean(values):
                        if any(v is None or not np.isfinite(v) for v in values): raise RuntimeError('Undefined metric; do not average successful-only')
                        return float(np.mean(values))
                    metrics={k:mean([r['metrics']['full'][k] for r in selected]) for k in METRICS}
                    for k in ANGLES:
                        metrics[k]=mean([r['metrics']['full']['angles'][k] if r['metrics']['full']['angles'] is not None else None for r in selected])
                    group=dict(proteins=len(selected),metrics=metrics)
                    for k in ['coarse_rmse','zero_coarse_rmse','fine_mse','zero_fine_mse','fixed_loss']:
                        group[k]=mean([r[k] for r in selected])
                    group['max_centroid_drift']=max(r['centroid_invariance_max_error'] for r in selected)
                    groups[subset]=group
                criteria=original_gate(groups)
                records.append(dict(arm=arm,seed=seed,step=step,subsets=groups,original_criteria=criteria,original_feasibility_pass=all(criteria.values())))
    for seed in parent['seeds']:
        if len({hashes[(arm,seed)] for arm in p['arms'] if arm!='baseline'})!=1: raise RuntimeError('Training order/rotation streams differ across arms')
    final=p['steps']; contrasts=[]; per_protein=[]
    specs=dict(boundary_without_context={'boundary':1,'baseline':-1}, boundary_with_context={'both':1,'context':-1},
               context_without_boundary={'context':1,'baseline':-1}, context_with_boundary={'both':1,'boundary':-1},
               interaction={'both':1,'context':-1,'boundary':-1,'baseline':1})
    for seed in parent['seeds']:
        for subset in ['train','validation']:
            ids=[i for i,s in expected.items() if s==subset]
            for name, coefficients in specs.items():
                metrics={}
                for key in METRICS+ANGLES:
                    values=[]
                    for target in ids:
                        value=0.
                        for arm, coefficient in coefficients.items():
                            m=evaluations[(arm,seed,final)][target]['metrics']['full']
                            value+=coefficient*(m[key] if key in METRICS else m['angles'][key])
                        values.append(value)
                        per_protein.append(dict(seed=seed,subset=subset,target=target,contrast=name,metric=key,delta=value))
                    metrics[key]=float(np.mean(values))
                contrasts.append(dict(seed=seed,subset=subset,contrast=name,delta=metrics))
    last=[r for r in records if r['step']==final]
    passes={arm:all(r['original_feasibility_pass'] for r in last if r['arm']==arm) for arm in p['arms']}
    result=dict(status='complete',protocol_sha256=ph,records=records,final_contrasts=contrasts,
                paired_protein_deltas=per_protein,both_seed_original_gate=passes,timings=timings,
                matched_augmentation_streams=True,scope=p['scope'],limitations=p['limitations'],
                selection='All fixed checkpoints and seeds retained; final16k is fixed. No selected best seed/checkpoint.',
                next='No automatic extra budget, hyperparameter sweep, new data or generator training.')
    atomic_json(RUN/'comparison.json',result)
    lines=['两层autoencoder边界loss与邻块信息2×2对照','='*50,'',p['scope'],'',
           '原基线复用已完成的同种子16k结果；新三臂从相同公共权重初始化各训练16k。',
           '参数：baseline/boundary 186175；context/both 191295（+5120，+2.75%）。信息与容量未完全解耦。',
           '数据：8训练/8反复使用开发蛋白。每个蛋白等权。CN阈值1.0–1.8Å，块内与跨块分母不同。','',
           '固定16k结果：arm / seed / subset / CA RMSD(Å) / CN(%) / within(%) / boundary(%) / planarity(deg) / clashes per residue','']
    for record in last:
        for subset,g in record['subsets'].items():
            m=g['metrics'];lines.append(f"{record['arm']} / {record['seed']} / {subset} / {m['rmsd']:.4f} / {100*m['cn_rate']:.2f} / {100*m['within_rate']:.2f} / {100*m['boundary_rate']:.2f} / {m['peptide_planarity_deviation_degrees']:.3f} / {m['clashes_per_residue']:.5f}")
    lines+=['','各配置两个种子均过原门槛：'+json.dumps(passes),'',
            '主要对照为固定16k、同蛋白同种子配对差；训练旋转/顺序摘要在三新臂之间完全一致。',
            '四个主效应条件差及交互项、逐蛋白差、全部中间点保存在comparison.json。两种子八开发蛋白不足以建立独立确认或稳定交互机制。',
            p['limitations'],'',
            '几何直接参与训练；报告训练目标以外的planarity/omega、RMSD、碰撞与旋转诊断，但不据开发结果声称泛化。',
            '原失败记录保留。本轮不选最优checkpoint、不自动追加训练、改门槛或启动生成器。']
    (RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    colors=dict(baseline='#555555',boundary='#2166ac',context='#e08214',both='#238b45')
    labels=[('rmsd','CA RMSD (Å)',1),('cn_rate','C–N outliers (%)',100),('boundary_rate','Boundary C–N (%)',100),('peptide_planarity_deviation_degrees','Planarity deviation (degrees)',1)]
    for i,seed in enumerate(parent['seeds']):
        for ax,(key,label,scale) in zip(axes[i],labels):
            for arm in p['arms']:
                rows=[r for r in records if r['arm']==arm and r['seed']==seed]
                ax.plot([r['step'] for r in rows],[r['subsets']['validation']['metrics'][key]*scale for r in rows],marker='o',label=arm,color=colors[arm])
            ax.set_ylabel(label);ax.set_xlabel('Optimizer updates');ax.set_title(f'Seed {seed} · reused development')
            ax.set_xticks(p['checkpoints'],['2k','4k','8k','16k']);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(RUN/'comparison.png',dpi=170);fig.savefig(RUN/'comparison.pdf');plt.close(fig)
    print('AE_BOUNDARY_REPORT_COMPLETE',json.dumps(dict(both_seed_original_gate=passes,final=[dict(arm=r['arm'],seed=r['seed'],validation=r['subsets']['validation']['metrics']) for r in last])),flush=True)


if __name__=='__main__': main()
