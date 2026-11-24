"""Render the frozen read-only decomposition without modifying audit inputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("artifacts/fine_state_difference_v1")
    report = json.loads((root / "report.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    labels = ["Parent", "Two-state\ntraining", "Added raw\nXYZ", "Added local\ngeometry"]
    order = ["parent", "two_state", "raw_context", "local_geometry"]
    for ax, name in zip(axes, ["1b2s_E", "1ef1_D"]):
        rows = [next(x for x in report["summary"] if x["protein"] == name and x["version"] == v) for v in order]
        means = np.array([r["mean_mse"] for r in rows])
        contrasts = np.array([r["contrast_mse"] for r in rows])
        ax.bar(range(4), means, color="#a3afb9", label="Error in shared mean")
        ax.bar(range(4), contrasts, bottom=means, color="#286b9d", label="Error in state difference")
        for i, row in enumerate(rows):
            ax.annotate(f"{row['total_mse']:.3f}", (i, row["total_mse"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
            individual = [x["total_mse"] for x in report["records"] if x["protein"] == name and x["version"] == order[i]]
            ax.scatter([i - .08, i + .08], individual, color="black", s=12, zorder=3)
        ax.set_title(name); ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_ylabel("Fine velocity MSE at noise start (t = 0)")
        ax.set_ylim(0, max(r["total_mse"] for r in rows) * 1.19)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("What improved during two-state training?\nTwo selected native training pairs; two seeds; not an ensemble-collapse test", fontsize=11)
    folder = Path("paper/figures/fine_state_difference_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Fine_State_Difference." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
