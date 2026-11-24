"""Report the matched sequence-condition comparison without changing checkpoints."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json

RUN = Path("artifacts/sequence_condition_probe_v1")
DESTINATION = Path("paper/figures/sequence_condition_probe_v1")
ARMS = ["AA_lookup", "ESM3_context"]


def main():
    report = json.loads((RUN / "report.json").read_text())
    rows = []
    for trial in report["rows"]:
        rows.extend(dict(row, arm=trial["arm"], seed=trial["seed"])
                    for row in trial["evaluation"]["rows"])
    primary = [row for row in rows if row["split"] == "validation" and row["outlet"] == "full"]
    proteins = sorted({row["protein"] for row in primary})
    macro = []
    for arm in ARMS:
        for seed in [2026, 2027]:
            values = [row for row in primary if row["arm"] == arm and row["seed"] == seed]
            if len(values) != 2:
                raise RuntimeError("Expected two new-sequence development cases")
            record = {key: float(np.mean([v[key] for v in values])) for key in [
                "mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A",
                "js_pairwise_distance_divergence_nats", "rmsf_mae_A", "mean_full_generation_seconds"]}
            record["cn_rate"] = float(np.mean([v["geometry"]["cn_rate"] for v in values]))
            record["mean_spread_ratio"] = float(np.mean([v["mean_pairwise_ca_rmsd_A"] /
                v["reference_mean_pairwise_ca_rmsd_A"] for v in values]))
            macro.append(dict(arm=arm, seed=seed, **record))
    result = dict(status="complete", macro_per_seed=macro,
        all_examples_and_noise_draws_matched=report["all_examples_and_noise_draws_matched"],
        trainable_parameters_equal=report["trainable_parameters_equal"],
        foundation_feature_extraction_seconds=report["feature_extraction"]["seconds"],
        scope=report["scope"], no_automatic_model_replacement=True)
    atomic_json(RUN / "primary_summary.json", result)
    lines = ["Matched sequence-condition pilot; both small generators have the same trainable capacity.",
        "Frozen ESM3 adds offline extraction and foundation pretraining; total compute is not matched.",
        "Two development sequences are held out from our training, not certified absent from ESM3 pretraining.", ""]
    lines.extend("PRIMARY " + json.dumps(row, sort_keys=True) for row in macro)
    lines.append("")
    for row in rows:
        lines.append(f"{row['arm']} seed{row['seed']} {row['split']} {row['protein']} {row['outlet']}: "
            f"CA {row['mean_sample_nearest_reference_ca_rmsd_A']:.4f} Å; "
            f"coverage error {row['mean_reference_nearest_sample_ca_rmsd_A']:.4f} Å; "
            f"CN {row['geometry']['cn_rate']*100:.3f}%; "
            f"spread {row['mean_pairwise_ca_rmsd_A']:.4f}/ref {row['reference_mean_pairwise_ca_rmsd_A']:.4f} Å; "
            f"JS {row['js_pairwise_distance_divergence_nats']:.5f}")
    (RUN / "result_summary.txt").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (lambda r: r["mean_sample_nearest_reference_ca_rmsd_A"], "Sample to nearest reference: CA RMSD (Å)"),
        (lambda r: 100*r["geometry"]["cn_rate"], "C–N abnormal bonds (%)"),
        (lambda r: r["js_pairwise_distance_divergence_nats"], "Pairwise-distance JS divergence (nats)"),
        (lambda r: r["mean_pairwise_ca_rmsd_A"]/r["reference_mean_pairwise_ca_rmsd_A"], "Generated / reference ensemble spread"),
    ]
    for axis, (value, label) in zip(axes.flat, panels):
        for index, protein in enumerate(proteins):
            for seed, marker, delta in [(2026, "o", -.04), (2027, "^", .04)]:
                points = []
                for arm, offset, color in [(ARMS[0], -.16, "#2369a5"), (ARMS[1], .16, "#c68124")]:
                    row = next(r for r in primary if r["protein"] == protein and r["arm"] == arm and r["seed"] == seed)
                    x, y = index+offset+delta, value(row)
                    axis.scatter(x, y, marker=marker, color=color, s=48, zorder=3)
                    points.append((x, y))
                axis.plot([x for x, _ in points], [y for _, y in points], color=".7", lw=.8)
        if label.startswith("Generated"):
            axis.axhline(1., ls="--", color=".4", lw=1)
        axis.set_xticks(range(len(proteins)), proteins)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.17)
    handles = [plt.Line2D([], [], color="#2369a5", marker="o", ls="", label="AA lookup control"),
        plt.Line2D([], [], color="#c68124", marker="o", ls="", label="Frozen ESM3 features"),
        plt.Line2D([], [], color=".4", marker="o", ls="", label="Seed 2026"),
        plt.Line2D([], [], color=".4", marker="^", ls="", label="Seed 2027")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(.5, .95))
    fig.suptitle("Sequence conditions for spatial multiscale generation", y=.995, fontsize=14)
    fig.text(.5, .016, "Two development proteins · Same small-generator capacity and training draws · Additional frozen foundation-model compute", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .9))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(DESTINATION / ("Sequence_Condition_Comparison."+extension), dpi=200)
    plt.close(fig)
    print("SEQUENCE_CONDITION_FIGURE_READY", DESTINATION, flush=True)


if __name__ == "__main__":
    main()
