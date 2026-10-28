"""Aggregate paired exploratory results and export a standalone comparison figure."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(baseline, feedback, direct=None):
    base = {row["target"]: row for row in baseline["targets"]}
    strategies = {"baseline": base}
    for policy in ["geometry", "random", "global"]:
        strategies[policy] = {row["target"]: row for row in feedback["targets"] if row["policy"] == policy}
        if set(strategies[policy]) != set(base):
            raise ValueError(f"Incomplete comparison for {policy}")
    if direct is not None:
        strategies["direct19"] = {row["target"]: row for row in direct["targets"]}
        if set(strategies["direct19"]) != set(base):
            raise ValueError("Incomplete direct19 control")
    rng = np.random.default_rng(20261018)
    resamples = rng.integers(0, len(base), size=(2000, len(base)))
    rows = []
    for policy, targets in strategies.items():
        values = list(targets.values())
        samples = sum(t["samples"] for t in values)
        row = dict(policy=policy, proteins=len(values), samples=samples,
                   proxy_invalid_samples=sum(t["proxy_invalid_samples"] for t in values),
                   upstream_clash_invalid_samples=sum(t["upstream_clash_invalid_samples"] for t in values),
                   cn_outlier_samples=sum(t["cn_outlier_samples"] for t in values),
                   median_diversity_ratio=float(np.median([
                       targets[key]["mean_pairwise_ca_rmsd"] / base[key]["mean_pairwise_ca_rmsd"]
                       for key in base if base[key]["mean_pairwise_ca_rmsd"] > 1e-8])),
                   initial_network_calls=sum(t["network_calls"] for t in (values if policy == "direct19" else base.values())),
                   additional_network_calls=0 if policy in ("baseline", "direct19") else sum(t["network_calls"] for t in values),
                   sampling_elapsed_seconds=sum(t.get("elapsed_seconds", 0) for t in values)
                       + (sum(t.get("elapsed_seconds", 0) for t in base.values()) if policy not in ("baseline", "direct19") else 0))
        row["total_network_calls"] = row["initial_network_calls"] + row["additional_network_calls"]
        available = [float(np.mean([v for v in t.get("reference_best_ca_rmsd", []) if v is not None]))
                     for t in values if any(v is not None for v in t.get("reference_best_ca_rmsd", []))]
        row["mean_best_available_reference_ca_rmsd"] = float(np.mean(available)) if available else None
        deltas = np.array([(base[key]["upstream_clash_invalid_samples"] - targets[key]["upstream_clash_invalid_samples"])
                           / base[key]["samples"] for key in base])
        estimates = deltas[resamples].mean(axis=1)
        row["ca_validity_gain_mean"] = float(deltas.mean())
        row["ca_validity_gain_ci_low"] = float(np.quantile(estimates, 0.025))
        row["ca_validity_gain_ci_high"] = float(np.quantile(estimates, 0.975))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.root / "baseline/summary.json").read_text())
    feedback = json.loads((args.root / "feedback/summary.json").read_text())
    direct_path = args.root / "direct19/summary.json"
    direct = json.loads(direct_path.read_text()) if direct_path.exists() else None
    rows = summarize(baseline, feedback, direct)
    geometry = {t["target"]: t for t in feedback["targets"] if t["policy"] == "geometry"}
    controls = {policy: {t["target"]: t for t in feedback["targets"] if t["policy"] == policy}
                for policy in ["random", "global"]}
    if direct is not None:
        controls["direct19"] = {t["target"]: t for t in direct["targets"]}
    rng = np.random.default_rng(20261018)
    resamples = rng.integers(0, len(geometry), size=(2000, len(geometry)))
    paired = {}
    for policy, control in controls.items():
        delta = np.array([(control[k]["upstream_clash_invalid_samples"] - geometry[k]["upstream_clash_invalid_samples"])
                          / geometry[k]["samples"] for k in sorted(geometry)])
        estimates = delta[resamples].mean(1)
        paired[f"geometry_vs_{policy}"] = dict(mean_gain=float(delta.mean()),
            ci_low=float(np.quantile(estimates, 0.025)), ci_high=float(np.quantile(estimates, 0.975)))
    result = dict(note=baseline["note"],
                  uncertainty="Exploratory 95% percentile bootstrap intervals resample proteins, not individual structures; 12 selected proteins are not a representative population sample.",
                  compute_note="Direct19 matches feedback's total denoiser calls; feedback uses additional structure decodes. Elapsed times include sampling/decoding, exclude model loading, and are not hardware-normalized benchmarks.",
                  geometry_paired_ca_validity_gain=paired, methods=rows)
    (args.root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.root / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labels = [r["policy"] for r in rows]
    panels = [("CA clash validity", [1 - r["upstream_clash_invalid_samples"] / r["samples"] for r in rows]),
              ("No C-N length outliers", [1 - r["cn_outlier_samples"] / r["samples"] for r in rows]),
              ("Median diversity / baseline", [r["median_diversity_ratio"] for r in rows])]
    for axis, (title, values) in zip(axes, panels):
        axis.bar(labels, values, color=["#666666", "#2670b5", "#e59535", "#659c48", "#9165ab"])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
        axis.set_ylim(0, max(1.05, max(values) * 1.1))
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory development: 12 proteins, 144 paired starting structures")
    fig.tight_layout()
    fig.savefig(args.root / "comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
