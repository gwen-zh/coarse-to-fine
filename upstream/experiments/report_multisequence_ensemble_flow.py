"""Descriptive new-sequence results, kept separate from seen-protein diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json

RUN = Path("artifacts/multisequence_ensemble_flow_v1")
DESTINATION = Path("paper/figures/multisequence_ensemble_flow_v1")


def main():
    report = json.loads((RUN / "report.json").read_text())
    records = []
    for trial in report["rows"]:
        records.extend(dict(r, arm=trial["arm"], seed=trial["seed"])
                       for r in trial["evaluation"]["rows"])
    proteins = sorted({r["protein"] for r in records if r["split"] == "validation"})
    primary = [r for r in records if r["split"] == "validation" and r["outlet"] == "full"]
    aggregates = []
    for arm in ["MD_only", "MD_plus_PDB"]:
        for seed in [2026, 2027]:
            selected = [r for r in primary if r["arm"] == arm and r["seed"] == seed]
            if len(selected) != 2:
                raise RuntimeError("Expected exactly two new development proteins")
            result = {key: float(np.mean([r[key] for r in selected])) for key in [
                "mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A",
                "js_pairwise_distance_divergence_nats", "rmsf_mae_A", "mean_full_generation_seconds"]}
            result["cn_rate"] = float(np.mean([r["geometry"]["cn_rate"] for r in selected]))
            result["mean_spread_ratio"] = float(np.mean([r["mean_pairwise_ca_rmsd_A"] /
                r["reference_mean_pairwise_ca_rmsd_A"] for r in selected]))
            aggregates.append(dict(arm=arm, seed=seed, **result))
    lines = ["Two new-sequence MD development proteins; not the sealed test set.",
        "Five MD training proteins; final fixed checkpoints; all raw samples retained.",
        "Fixed physical histogram bins differ from earlier per-sequence adaptive bins.",
        "Only two development proteins: descriptive early generalization check, not broad validation.", ""]
    for row in aggregates:
        lines.append("PRIMARY " + json.dumps(row, sort_keys=True))
    lines.append("")
    for row in records:
        lines.append(f"{row['arm']} seed{row['seed']} {row['split']} {row['protein']} {row['outlet']}: "
            f"sample->ref {row['mean_sample_nearest_reference_ca_rmsd_A']:.4f} Å; "
            f"ref->sample {row['mean_reference_nearest_sample_ca_rmsd_A']:.4f} Å; "
            f"CN {row['geometry']['cn_rate']*100:.3f}%; "
            f"spread {row['mean_pairwise_ca_rmsd_A']:.4f} / ref {row['reference_mean_pairwise_ca_rmsd_A']:.4f} Å; "
            f"JS {row['js_pairwise_distance_divergence_nats']:.5f}")
    (RUN / "result_summary.txt").write_text("\n".join(lines) + "\n")
    atomic_json(RUN / "primary_summary.json", dict(status="complete", macro_per_seed=aggregates,
        validation_proteins=proteins, heldout_test_opened=False, scope=report["scope"]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (lambda r: r["mean_sample_nearest_reference_ca_rmsd_A"], "Sample to nearest reference: CA RMSD (Å)"),
        (lambda r: 100*r["geometry"]["cn_rate"], "C–N abnormal bonds (%)"),
        (lambda r: r["js_pairwise_distance_divergence_nats"], "Pairwise-distance JS divergence (nats)"),
        (lambda r: r["mean_pairwise_ca_rmsd_A"] / r["reference_mean_pairwise_ca_rmsd_A"], "Generated / reference ensemble spread"),
    ]
    for axis, (value, label) in zip(axes.flat, panels):
        for index, protein in enumerate(proteins):
            for seed, marker, delta in [(2026, "o", -.04), (2027, "^", .04)]:
                points = []
                for arm, offset, color in [("MD_only", -.16, "#2369a5"), ("MD_plus_PDB", .16, "#c68124")]:
                    r = next(v for v in primary if v["protein"] == protein and v["arm"] == arm and v["seed"] == seed)
                    x, y = index+offset+delta, value(r)
                    axis.scatter(x, y, marker=marker, color=color, s=48, zorder=3)
                    points.append((x, y))
                axis.plot([x for x, _ in points], [y for _, y in points], color=".7", lw=.8)
        if label.startswith("Generated"):
            axis.axhline(1., color=".4", ls="--", lw=1)
        axis.set_xticks(range(len(proteins)), proteins)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.17)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#2369a5", label="MD only"),
        plt.Line2D([], [], marker="o", ls="", color="#c68124", label="MD + PDB replay"),
        plt.Line2D([], [], marker="o", ls="", color=".4", label="Seed 2026"),
        plt.Line2D([], [], marker="^", ls="", color=".4", label="Seed 2027")]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, .95))
    fig.suptitle("Spatial multiscale generation on two new sequences", y=.995, fontsize=14)
    fig.text(.5, .017, "Exploratory development results · 64 raw samples per model/protein · Test coordinates remain unopened", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .9))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(DESTINATION / ("Multi_Sequence_Development."+extension), dpi=200)
    plt.close(fig)
    print("MULTISEQUENCE_FIGURES_READY", DESTINATION, flush=True)


if __name__ == "__main__":
    main()
