import argparse,json
from pathlib import Path
import numpy as np
from report_multiscale_tokenizer import paired
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());data=json.loads((a.run/'evaluation.json').read_text())
    if not data['replay_passed']:raise RuntimeError('Replay failed; no scientific interpretation')
    rows=data['targets'];means={};comparisons={}
    def values(arm,variant,metric):return [r['arms'][arm]['methods'][variant][metric] for r in rows]
    fields=('ca_rmsd','cn_rate','ca_clashes_per_residue','long_range_distance_mae','cn_too_short_rate','cn_too_long_rate','median_cn_distance','median_adjacent_ca_distance')
    original={f:float(np.mean([r['original'][f] for r in rows])) for f in fields}
    for arm in protocol['arms']:
        means[arm]={variant:{metric:float(np.mean(values(arm,variant,metric))) for metric in fields} for variant in protocol['variants']}
    for metric in fields[:4]:
        comparisons[metric]={}
        for variant in protocol['variants']:comparisons[metric]['global_minus_local_'+variant]=paired(values('global',variant,metric),values('local',variant,metric))
        for arm in protocol['arms']:
            for left,right in [('visible_fixed','continuous'),('hidden_snapped','visible_fixed'),('hidden_snapped','continuous')]:
                comparisons[metric][arm+'_'+left+'_minus_'+right]=paired(values(arm,left,metric),values(arm,right,metric))
    g=means['global'];l=means['local']
    gate=dict(global_snap_cn_at_most_20percent=g['hidden_snapped']['cn_rate']<=.2,
        global_snap_cn_at_least_half_reduction=g['hidden_snapped']['cn_rate']<=.5*g['continuous']['cn_rate'],
        global_snap_rmsd_within_10percent_continuous=g['hidden_snapped']['ca_rmsd']<=1.1*g['continuous']['ca_rmsd'],
        global_snap_rmsd_5percent_better_local=g['hidden_snapped']['ca_rmsd']<=.95*l['hidden_snapped']['ca_rmsd'],
        global_snap_local_paired_interval_negative=comparisons['ca_rmsd']['global_minus_local_hidden_snapped']['bootstrap95'][1]<0)
    passed=all(gate.values());decision=('Eligible for a bounded discrete fine-completion training pilot; still not a model upgrade.' if passed else 'Do not automatically retrain or enlarge this completion architecture. Decode projection did not meet the joint geometry/accuracy/information-retention screen. Preserve the exploratory side-channel finding separately.')
    result=dict(status='complete',scope=protocol['scope'],original=original,means=means,paired=comparisons,gate=gate,gate_passed=passed,decision=decision,replay=data['replay'],decoder_calls=data['decoder_calls'],hidden_token_accuracy={arm:float(np.mean([r['arms'][arm]['hidden_token_accuracy'] for r in rows])) for arm in protocol['arms']})
    atomic_json(a.run/'comparison.json',result)
    lines=['半数 fine 解码方式诊断','=======================','',protocol['scope'],'','冻结原模型及已训练检查点，固定上一轮保存的coarse token；没有训练或按参考结构选择候选。局部与全局在同一GPU解码。','',
      '完整原fine：RMSD %.6f Å，C–N异常率 %.3f%%。'%(original['ca_rmsd'],100*original['cn_rate']),'']
    for arm in protocol['arms']:
        lines.append(arm)
        for variant,m in means[arm].items():lines.append('  %s：RMSD %.6f Å；CN异常 %.3f%%；长程距离MAE %.6f Å；平均每蛋白CN中位距离 %.4f Å。'%(variant,m['ca_rmsd'],100*m['cn_rate'],m['long_range_distance_mae'],m['median_cn_distance']))
        lines.append('')
    lines.extend(['配对RMSD：'+json.dumps(comparisons['ca_rmsd'],ensure_ascii=False,indent=2),'','继续门槛：'+json.dumps(gate,ensure_ascii=False),'通过：'+str(passed),decision,'',
      '限制：同一批反复使用的24开发蛋白，一个训练seed，真实结构编码；不是生成实验或独立确认。把预测映射回原fine embedding不保证组合几何合法，也不证明离散补全训练必然成功或失败。','',
      '邻居只由1280维decoder embedding标准化距离决定，候选只含4096个普通fine token。隐藏真值仅用于事后评价；可见位置恢复原embedding。该投影是对现有预测的处理，不是重新训练到词表上的模型。',
      '连续路径重放偏差：'+json.dumps(data['replay'],ensure_ascii=False),
      '解码次数：'+str(data['decoder_calls'])])
    (a.run/'results_zh.rst').write_text('\n'.join(lines)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
