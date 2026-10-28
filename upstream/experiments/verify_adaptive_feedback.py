"""Verify that early stopping saves compute without changing the fixed-budget output."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    reference_root = args.root / "feedback/geometry"
    adaptive_root = args.root / "feedback_adaptive/geometry"
    count, mismatches = 0, []
    for reference in sorted(reference_root.glob("*/seed*.npz")):
        adaptive = adaptive_root / reference.relative_to(reference_root)
        with np.load(reference) as original, np.load(adaptive) as candidate:
            if any(not np.array_equal(original[key], candidate[key]) for key in ["fine", "coarse"]):
                mismatches.append(str(reference))
        count += 1
    baseline = json.loads((args.root / "baseline/summary.json").read_text())["targets"]
    if count != sum(t["samples"] for t in baseline):
        raise ValueError("Incomplete adaptive comparison")
    fixed = [t for t in json.loads((args.root / "feedback/summary.json").read_text())["targets"] if t["policy"] == "geometry"]
    adaptive = json.loads((args.root / "feedback_adaptive/summary.json").read_text())["targets"]
    result = dict(samples=count, token_mismatches=mismatches,
                  baseline_network_calls=sum(t["network_calls"] for t in baseline),
                  fixed_extra_network_calls=sum(t["network_calls"] for t in fixed),
                  adaptive_extra_network_calls=sum(t["network_calls"] for t in adaptive),
                  baseline_elapsed_seconds=sum(t["elapsed_seconds"] for t in baseline),
                  fixed_extra_elapsed_seconds=sum(t["elapsed_seconds"] for t in fixed),
                  adaptive_extra_elapsed_seconds=sum(t["elapsed_seconds"] for t in adaptive))
    (args.root / "adaptive_equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if mismatches:
        raise RuntimeError("Adaptive and fixed-budget outputs differ; inspect before claiming equivalence")


if __name__ == "__main__":
    main()
