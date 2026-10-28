"""Export the measured utilization, reconstruction ceiling and capacity controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    usage = json.loads((args.root / "encoder_broad/summary.json").read_text())
    reconstruction = json.loads((args.root / "tokenizer/summary.json").read_text())
    control = json.loads((args.root / "quantization_control/summary.json").read_text())
    levels = ["coarse", "medium", "fine"]
    groups = {"apo/codnas (n=9)": [t for t in reconstruction["targets"] if not t["target"].startswith("PED")],
              "PED (n=3)": [t for t in reconstruction["targets"] if t["target"].startswith("PED")]}
    means = {group: {level: float(np.mean([t["levels"][level]["ca_rmsd"] for t in rows]))
                     for level in levels} for group, rows in groups.items()}
    result = dict(codebooks={name: {k: v for k, v in stats.items() if k != "counts"}
                             for name, stats in usage["codebooks"].items()},
                  mean_ca_reconstruction_rmsd=means,
                  quantization_validation={name: {k: v for k, v in stats.items() if k != "targets"}
                                           for name, stats in control["validation"].items()},
                  note="Exploratory development measurements; quantization gains do not establish generation improvements.")
    (args.root / "diagnostic_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    values = [usage["codebooks"][level] for level in levels]
    bars = axes[0].bar(levels, [v["utilization"] for v in values], color=["#659c48", "#e59535", "#2670b5"])
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                     f"{value['used']}/{value['vocabulary']}", ha="center")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Utilization: 204 proteins, 36,579 residues")
    x = np.arange(3)
    for i, (group, data) in enumerate(means.items()):
        axes[1].bar(x + (i - 0.5) * 0.36, [data[level] for level in levels], 0.36, label=group)
    axes[1].set_xticks(x, levels)
    axes[1].set_ylabel("Mean aligned CA RMSD (A)")
    axes[1].set_title("Reconstruction without generation")
    axes[1].legend(frameon=False)
    methods = ["released512", "data_init512", "kmeans512"]
    axes[2].bar(["released", "data init", "k-means"],
                [control["validation"][method]["residue_weighted_mse"] for method in methods],
                color=["#666666", "#e59535", "#2670b5"])
    axes[2].set_ylabel("Latent MSE")
    axes[2].set_title("512 codes: 40 validation proteins")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Tokenizer diagnostics and latent-space capacity control")
    fig.tight_layout()
    fig.savefig(args.root / "tokenizer_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
