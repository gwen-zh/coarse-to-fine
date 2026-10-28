"""Lock development inputs, two edit-count distributions and exact call budgets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'protocol.json').exists():raise FileExistsError('Do not overwrite locked protocol')
    parent=Path('artifacts/development/matched_bestofn_v1/protocol.json');old=json.loads(parent.read_text())
    targets=old['targets'];assert len(targets)==192 and len({x['protein'] for x in targets})==24
    assert sum(x['decoder_call_budget'] for x in targets)==8625
    for item in targets:
        path=Path(old['source'])/item['input'];assert hashlib.sha256(path.read_bytes()).hexdigest()==old['input_sha256'][item['target']]
        with np.load(path) as z:assert len(z['fine'])>1 and np.array_equal(z['logits'].argmax(-1),z['fine'])
    arms=['independent_single','independent_multi_geometric','independent_multi_uniform','random_greedy','violation_greedy']
    files=['bounded_multiedit_proposals.py','run_bounded_multiedit.py','report_bounded_multiedit.py','prepare_bounded_multiedit.py','bestofn_proposals.py','fine_search_constraints.py','structural_metrics.py','backbone_quality_audit.py','evaluate_residual_vq.py','diagnostic_io.py']
    code={str(Path('upstream/experiments')/f):hashlib.sha256((Path('upstream/experiments')/f).read_bytes()).hexdigest() for f in files}
    protocol=dict(version=1,scope='Fixed development comparator;24 reused proteins x8 original-path outputs. No training, no58/64 confirmation access. Independent bounded multi-edit BoN versus single edits and accumulating searches.',source=old['source'],corpus=old['corpus'],weights=old['weights'],targets=targets,input_sha256=old['input_sha256'],parent_protocol_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),arms=arms,chunks=16,delta_angstrom=1.,edit_fraction_cap=.1,multiedit_seed=20261026,bootstrap_seed=20261026,bootstrap_draws=20000,
        proposal='Fixed original-head top5; uniform alternative excluding initial token. Multi-edit sites uniformly sampled without replacement; each candidate independently starts from initial tokens. Counts m=1..floor(.1*N), uniform or truncated geometric proportional to2**(-m). Both distributions reported; no winner selection.',
        single_stream='Existing local_proposals seed20261022+input index; shared by independent_single and random_greedy. Greedy can accumulate edits and is gated relative to initial; proposed total edits may exceed cap and count as rejections.',
        violation_stream='Current decoded CN/CA violation sites, sorted; fixed original-head pool; reset after accepted edit. No-site remainder decodes unchanged incumbent and records padding.',
        budgets='Exactly8625decoder calls per arm, per-output counts inherited unchanged from prior experiment. Baseline decode shared; all proposals including repeats decoded. Arms rotated by input index; all5 on same CPU host per output; FP32 TF32disabled. Not matched end-to-end FLOPs.',
        selection='Reference-free initial-relative CN/CA nonincrease with one strict decrease,CA displacement<=1A,token edits<=10percent,lexicographic CN thenCA. Reference files opened after all5 arms finish selection.',
        contrasts=[[x,'violation_greedy'] for x in arms if x!='violation_greedy']+[['independent_multi_geometric','independent_single'],['independent_multi_uniform','independent_single'],['random_greedy','independent_single']],
        endpoints='CA-C-N and C-N-CA reference angle MAE as ungated outcomes; mean reference CA RMSD and8-output pairwise spread; CNrate/clash proxies separately. Paired protein bootstrap20kseed20261026; unadjusted descriptive95%intervals, no confirmatory significance claim.',
        scheduling='CPU16threads per shard; GPU preflight57668567cancelled whilependingbefore any outputs duepriorityqueue; scientific inputs/proposals/budgetsunchanged',code_sha256=code)
    atomic_json(a.run/'protocol.json',protocol)
    print('LOCKED',str(a.run),192,'inputs',8625,'calls per arm',5,'arms')

if __name__=='__main__':main()
