"""Report fixed half-fine information-completion pilot without test claims."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from report_multiscale_tokenizer import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text())
    data={arm:json.loads((a.run/arm/'evaluation.json').read_text()) for arm in protocol['arms']}
    names=[[r['target'] for r in data[arm]['targets']] for arm in protocol['arms']]
    assert names[0]==names[1]==names[2]
    def values(arm,method,metric):return [r['methods'][method][metric] for r in data[arm]['targets']]
    comparisons={}
    for metric in ('ca_rmsd','cn_rate','long_range_distance_mae','ca_clashes_per_residue','hidden_embedding_mse'):
        for arm in ('none','local'):
            for base in ('original','half_untrained'):
                np.testing.assert_allclose(values(arm,base,metric),values('global',base,metric),rtol=1e-5,atol=1e-4)
        comparisons[metric]={
          'global_minus_local':paired(values('global','trained',metric),values('local','trained',metric)),
          'global_minus_no_coarse':paired(values('global','trained',metric),values('none','trained',metric)),
          'local_minus_no_coarse':paired(values('local','trained',metric),values('none','trained',metric)),
          'global_minus_original_fullfine':paired(values('global','trained',metric),values('global','original',metric)),
          'global_minus_shuffled':paired(values('global','trained',metric),values('global','shuffled_coarse',metric)),
          'global_minus_zero_coarse':paired(values('global','trained',metric),values('global','zero_coarse',metric))}
    g,l,n=[data[k]['methods']['trained'] for k in ('global','local','none')]
    checks=dict(global_rmsd_5percent_better_local=g['ca_rmsd']<=.95*l['ca_rmsd'],
        global_rmsd_10percent_better_none=g['ca_rmsd']<=.9*n['ca_rmsd'],
        global_long_range_no_worse_local=g['long_range_distance_mae']<=l['long_range_distance_mae'],
        global_cn_within_half_pp_local=g['cn_rate']<=l['cn_rate']+.005,
        at_least_8_codes=data['global']['code_usage']>=8,
        shuffled_rmsd_degradation_interval_positive=comparisons['ca_rmsd']['global_minus_shuffled']['bootstrap95'][1]<0)
    passed=all(checks.values())
    decision=('Eligible for three-seed development replication only; no generator or independent-test claim.' if passed else 'Stop this fixed half-fine configuration. No automatic extra epochs/variants/generator training. Does not rule out all multiscale tokenizers.')
    result=dict(status='complete',scope=protocol['scope'],means={arm:d['methods'] for arm,d in data.items()},paired=comparisons,gate=checks,gate_passed=passed,decision=decision,code_usage={k:v['code_usage'] for k,v in data.items()})
    atomic_json(a.run/'comparison.json',result)
    base=data['global']['methods']
    lines=['半数 fine 信息补全实验','======================','',
      '64个训练蛋白，24个重复使用的开发蛋白，单seed，三组各10个epoch。真实结构编码重建，不是序列生成；未使用独立确认集。','',
      '完整原fine：RMSD %.6f Å。'%base['original']['ca_rmsd'],
      '一半fine直接替换为均值、不训练：RMSD %.6f Å。'%base['half_untrained']['ca_rmsd'],'']
    for arm,label in [('none','一半fine＋训练补全器，无coarse'),('local','一半fine＋局部coarse'),('global','一半fine＋全局coarse')]:
        m=data[arm]['methods']['trained'];lines.append('%s：RMSD %.6f Å；C–N异常率 %.3f%%；长程距离MAE %.6f Å。'%(label,m['ca_rmsd'],100*m['cn_rate'],m['long_range_distance_mae']))
    lines.extend(['','全局减局部 RMSD：'+json.dumps(comparisons['ca_rmsd']['global_minus_local'],ensure_ascii=False),
      '全局减无coarse RMSD：'+json.dumps(comparisons['ca_rmsd']['global_minus_no_coarse'],ensure_ascii=False),
      '全局减打乱coarse RMSD：'+json.dumps(comparisons['ca_rmsd']['global_minus_shuffled'],ensure_ascii=False),'',
      '继续门槛：'+json.dumps(checks,ensure_ascii=False),'通过：'+str(passed),'决策：'+decision,'',
      '码率：完整原fine12bits/残基；两种coarse组偶数长度11bits/残基，奇数长度11+6/N；无coarse组约6bits/残基。仅名义离散码率，不是实际序列化压缩率。',
      '隐藏fine的码向量和decoder embedding均在融合入口屏蔽，输出不依赖其被屏蔽值。原完整结构/fine可进入coarse编码器，但新增信息必须经过32类硬量化后才能进融合器；原始几何没有旁路。',
      '局部与全局两组同容量同码率；无coarse组使用同款融合器但有效参数更少。缺失位置融合幅度上限4个原embedding标准差，可见位置0.1；与上一轮小修复结构不同，不能横跨两轮作因果比较。',
      '训练损失含隐藏embedding监督和真实decoder几何损失；标签只进入损失或结构编码器，不作为融合器输入。冻结原fine码本和decoder。',
      '限制：全局编码器能见真实结构，未证明新coarse可由序列预测。单seed/小数据/有限训练/固定融合上限，阴性结果不证明所有架构无效。打乱和置零消融也可能引入分布偏移。'])
    (a.run/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status','means','gate','gate_passed','decision')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
