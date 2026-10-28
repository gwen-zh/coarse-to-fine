"""Report the next feedback variant without treating its optimized CN score as independent."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.root / "baseline/summary.json").read_text())["targets"]
    ca = json.loads((args.root / "feedback/summary.json").read_text())["targets"]
    backbone = json.loads((args.root / "feedback_backbone/summary.json").read_text())["targets"]
    ids = {t["target"] for t in baseline}
    rows = []
    strategies = {"baseline": baseline, "CA_geometry": [t for t in ca if t["policy"] == "geometry"]}
    strategies.update({f"backbone_{policy}": [t for t in backbone if t["policy"] == policy]
                       for policy in ["geometry", "random", "global"]})
    for method, targets in strategies.items():
        if {t["target"] for t in targets} != ids:
            raise ValueError(f"Incomplete backbone comparison: {method}")
        samples = sum(t["samples"] for t in targets)
        references = [v for t in targets for v in t["reference_best_ca_rmsd"] if v is not None]
        rows.append(dict(method=method, samples=samples,
                         ca_clash_validity=1 - sum(t["upstream_clash_invalid_samples"] for t in targets) / samples,
                         no_cn_outlier_fraction=1 - sum(t["cn_outlier_samples"] for t in targets) / samples,
                         mean_pairwise_ca_rmsd=float(np.mean([t["mean_pairwise_ca_rmsd"] for t in targets])),
                         mean_best_available_reference_ca_rmsd=float(np.mean(references))))
    result = dict(note="Development only. CA and C-N constraints are optimized by the backbone variant; reference RMSD and diversity are needed to assess tradeoffs. No ensemble-distribution or independent-test claim.",
                  methods=rows)
    (args.root / "backbone_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
