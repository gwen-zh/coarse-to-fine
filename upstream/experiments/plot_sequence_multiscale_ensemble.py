"""Descriptive paired-outlet plots for the fixed single-protein MD pilot."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/sequence_multiscale_ensemble_v1")
DESTINATION = Path("paper/figures/sequence_multiscale_ensemble_v1")
ARMS = ["latent16_teacher", "centers3_teacher", "centers3_scheduled"]
LABELS = ["Learned coarse\nPaired context", "3D centers\nPaired context", "3D centers\nScheduled context"]


def main():
    report = json.loads((ROOT / "report.json").read_text())
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4))
    measures = [
        ("mean_sample_nearest_reference_ca_rmsd_A", "Sample to nearest R2 frame: CA RMSD (Å)"),
        ("cn_rate", "C–N abnormal bonds (%)"),
        ("mean_pairwise_ca_rmsd_A", "Ensemble spread: pairwise CA RMSD (Å)"),
        ("js_pairwise_distance_divergence_nats", "Pairwise-distance JS divergence (nats)"),
    ]
    lines = ["Fixed single-protein spatial multiscale generation pilot",
             "R1 training; reused R2 development; all 64 samples retained per model.",
             "Two training seeds; descriptive values, no independent-protein or equilibrium claim.", ""]
    lookup = {(r["arm"], r["seed"]): r for r in report["rows"]}
    for axis, (key, label) in zip(axes.flat, measures):
        for index, arm in enumerate(ARMS):
            for seed, marker, offset in [(2026, "o", -.06), (2027, "^", .06)]:
                row = lookup[(arm, seed)]
                entries = {r["outlet"]: r for r in row["evaluation"]["outlets"]}
                values = []
                for outlet, x, color in [("full_continuous", index - .17 + offset, "#2369a5"),
                                          ("full_quantized", index + .17 + offset, "#c68124")]:
                    data = entries[outlet]
                    value = data["geometry"][key] * 100 if key == "cn_rate" else data[key]
                    values.append((x, value))
                    axis.scatter(x, value, marker=marker, color=color, s=46, zorder=3)
                axis.plot([x for x, _ in values], [y for _, y in values], color=".72", lw=.8, zorder=1)
        if key == "mean_pairwise_ca_rmsd_A":
            reference = report["rows"][0]["evaluation"]["outlets"][0]["reference_mean_pairwise_ca_rmsd_A"]
            axis.axhline(reference, color=".4", linestyle="--", linewidth=1, label="R2 reference spread")
            axis.legend(frameon=False, fontsize=8)
        axis.set_xticks(range(3), LABELS)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.16)
        axis.set_xlim(-.6, 2.6)
    handles = [plt.Line2D([], [], color="#2369a5", marker="o", ls="", label="Continuous fine"),
               plt.Line2D([], [], color="#c68124", marker="o", ls="", label="Quantized outlet"),
               plt.Line2D([], [], color=".4", marker="o", ls="", label="Seed 2026"),
               plt.Line2D([], [], color=".4", marker="^", ls="", label="Seed 2027")]
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .96))
    figure.suptitle("Sequence-conditioned spatial multiscale ensembles", y=.997, fontsize=14)
    figure.text(.5, .016, "One 90-residue ATLAS protein · Reused R2 development · 64 raw samples per model · No geometry filtering", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .91))
    for suffix in ["pdf", "png", "svg"]:
        figure.savefig(DESTINATION / ("Sequence_Multiscale_Pilot." + suffix), dpi=200)
    plt.close(figure)
    for arm in ARMS:
        for seed in [2026, 2027]:
            entry = lookup[(arm, seed)]
            lines.append(f"{arm}, seed {seed}, {entry['complete']['parameters']} trainable generator parameters")
            for row in entry["evaluation"]["outlets"]:
                lines.append(f"  {row['outlet']}: CA nearest R2 {row['mean_sample_nearest_reference_ca_rmsd_A']:.4f} Å; "
                    f"R2 nearest sample {row['mean_reference_nearest_sample_ca_rmsd_A']:.4f} Å; "
                    f"CN {row['geometry']['cn_rate']*100:.3f}%; "
                    f"spread {row['mean_pairwise_ca_rmsd_A']:.4f} Å; "
                    f"JS-PwD {row['js_pairwise_distance_divergence_nats']:.5f} nats")
            lines.append("  Measured mean seconds: " + json.dumps(entry["evaluation"]["mean_latency_seconds"]))
            lines.append("")
    (ROOT / "result_summary.txt").write_text("\n".join(lines) + "\n")
    print(DESTINATION / "Sequence_Multiscale_Pilot.pdf")


if __name__ == "__main__":
    main()
