"""Report solver sensitivity without selecting on the spread statistic alone."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("artifacts/multiscale_solver_probe_v1")
DESTINATION = Path("paper/figures/multiscale_solver_probe_v1")


def main():
    data = json.loads((ROOT / "report.json").read_text())["rows"]
    lookup = {(r["solver"], r["seed"]): r for r in data}
    solvers = ["euler16", "heun16", "euler64", "heun64"]
    labels = ["Euler\n16 calls", "Heun\n16 calls", "Euler\n64 calls", "Heun\n64 calls"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    functions = [
        (lambda r: r["coarse_spread"]["generated_A"], "Coarse-center pairwise RMSD (Å)", data[0]["coarse_spread"]["R2_A"]),
        (lambda r: r["mean_pairwise_ca_rmsd_A"], "Backbone ensemble spread (Å)", data[0]["reference_mean_pairwise_ca_rmsd_A"]),
        (lambda r: r["mean_sample_nearest_reference_ca_rmsd_A"], "Sample to nearest R2 frame: CA RMSD (Å)", None),
        (lambda r: 100*r["geometry"]["cn_rate"], "C–N abnormal bonds (%)", None),
    ]
    for axis, (value, ylabel, reference) in zip(axes.flat, functions):
        for index, solver in enumerate(solvers):
            for seed, marker, offset in [(2026, "o", -.065), (2027, "^", .065)]:
                color = "#2369a5" if solver.startswith("euler") else "#c68124"
                axis.scatter(index+offset, value(lookup[(solver, seed)]), marker=marker, color=color, s=47)
        if reference is not None:
            axis.axhline(reference, ls="--", color=".4", lw=1, label="R2 reference")
            axis.legend(frameon=False, fontsize=9)
        axis.set_xticks(range(4), labels)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.17)
    handles = [plt.Line2D([], [], color=".4", marker="o", ls="", label="Seed 2026"),
               plt.Line2D([], [], color=".4", marker="^", ls="", label="Seed 2027")]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(.5, .955))
    fig.suptitle("Coarse sampling accuracy changes the ensemble trade-off", y=.995, fontsize=14)
    fig.text(.5, .017, "Same frozen weights and noise seeds · Fine sampler fixed at 16 Euler calls · One reused development protein", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg"]:
        fig.savefig(DESTINATION / ("Coarse_Solver_Probe."+extension), dpi=200)
    plt.close(fig)
    print(DESTINATION / "Coarse_Solver_Probe.pdf")


if __name__ == "__main__":
    main()
