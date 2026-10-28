"""Report local gradient feasibility without treating training probes as a benchmark."""
import argparse
import json
from pathlib import Path

from train_residual_vq import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    rows=[]
    for target in manifest['targets']:
        arms={}
        for arm in manifest['protocol']['arms']:
            result=json.loads((args.run/target['target']/(arm+'.json')).read_text())
            arms[arm]={key:result[key] for key in ['baseline','best','final','first_gradient_norm']}
            if arm=='kl_only' and any(step['changed_tokens'] for step in result['trace']):
                raise ValueError('KL-only control changed hard tokens')
        rows.append(dict(target=target['target'],arms=arms))
    counts={}
    for arm in manifest['protocol']['arms']:
        counts[arm]=dict(
            nonzero_finite_gradient_targets=sum(0<r['arms'][arm]['first_gradient_norm']<float('inf') for r in rows),
            best_cn_improved_targets=sum(r['arms'][arm]['best']['metrics']['backbone_cn_outliers']<r['arms'][arm]['baseline']['backbone_cn_outliers'] for r in rows),
            final_cn_improved_targets=sum(r['arms'][arm]['final']['metrics']['backbone_cn_outliers']<r['arms'][arm]['baseline']['backbone_cn_outliers'] for r in rows))
    atomic_json(args.run/'comparison.json',dict(protocol=manifest['protocol'],targets=rows,counts=counts,
        interpretation='Four selected TRAIN examples; biased hard-forward surrogate gradient and local logit optimization only. Best uses repeated geometry selection. No learned-generator, held-out quality, or publication claim. Compare final and shuffled control as well as best.'))
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
