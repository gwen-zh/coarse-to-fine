import json
from pathlib import Path
from diagnostic_io import atomic_json
RUN=Path('artifacts/coarse_solver_convergence_v1')


def main():
    p=json.loads((RUN/'protocol.json').read_text());results=[]
    lines=['冻结coarse的数值积分收敛诊断','='*44,'',p['scope'],'',
           '指标是按每块有效残基数加权、proper Kabsch对齐后的块中心RMSD，不是完整蛋白CA RMSD。',
           'Euler128只是本轮最细网格参照，不能宣称它是精确ODE解或真实分布。',
           'seed / subset / solver / NFE / center RMSD(Å) / delta vs Euler16(Å) / endpoint shape difference vs Euler128(Å)','']
    for seed in p['seeds']:
        r=json.loads((RUN/f'seed{seed}/complete.json').read_text());results.append(r)
        for subset,g in r['groups'].items():
            for name,m in g['solvers'].items():
                lines.append(f"{seed} / {subset} / {name} / {m['nfe']} / {m['weighted_block_center_rmsd']:.5f} / {m['delta_to_euler16']['mean']:+.5f} / {m['weighted_center_rmsd_to_euler128']:.5f}")
    limitations='Frozenweights,coarseonly,2draws per protein,8train/16reuseddevelopment,2seeds. No fullfine generation,independentconfirmation or thermodynamicensembleclaim. DifferentEulerbudgets are convergence diagnostics;Euler64 andMidpoint32match64NNcalls,butactualCPU integrationtime is separatelyrecorded and excludesdecoder/I/O.'
    atomic_json(RUN/'comparison.json',dict(status='complete',seeds=results,limitations=limitations));lines+=['',limitations];(RUN/'results_zh.rst').write_text('\n'.join(lines)+'\n')
    state_path=Path('artifacts/run_state.json');state=json.loads(state_path.read_text());state.setdefault('coarse_solver_convergence_v1',{}).update(status='complete',report=str(RUN/'results_zh.rst'));atomic_json(state_path,state)
    print('COARSE_SOLVER_REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
