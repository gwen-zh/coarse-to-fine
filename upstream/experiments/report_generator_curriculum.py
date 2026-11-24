"""Summarize matched PDB pretraining curricula on reused development proteins."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json

RUN = Path("artifacts/generator_curriculum_v1")
DESTINATION = Path("paper/figures/generator_curriculum_v1")
ARMS = ["interleaved", "PDB_then_MD"]


def main():
    report = json.loads((RUN / "report.json").read_text())
    rows, probes = [], []
    for trial in report["rows"]:
        rows.extend(dict(row, arm=trial["arm"], seed=trial["seed"])
                    for row in trial["evaluation"]["rows"])
        probes.append(dict(arm=trial["arm"], seed=trial["seed"],
                           **trial["pretrain_probe"]["mean_metrics"]))
    primary = [r for r in rows if r["split"] == "validation" and r["outlet"] == "full"]
    proteins = sorted({r["protein"] for r in primary})
    macro = []
    for arm in ARMS:
        for seed in [2026, 2027]:
            values = [r for r in primary if r["arm"] == arm and r["seed"] == seed]
            if len(values) != 2:
                raise RuntimeError("Expected two development proteins")
            record = {key: float(np.mean([v[key] for v in values])) for key in [
                "mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A",
                "js_pairwise_distance_divergence_nats", "rmsf_mae_A", "mean_full_generation_seconds"]}
            record["cn_rate"] = float(np.mean([v["geometry"]["cn_rate"] for v in values]))
            record["mean_spread_ratio"] = float(np.mean([v["mean_pairwise_ca_rmsd_A"] /
                v["reference_mean_pairwise_ca_rmsd_A"] for v in values]))
            macro.append(dict(arm=arm, seed=seed, **record))
    atomic_json(RUN / "primary_summary.json", dict(status="complete", macro_per_seed=macro,
        fixed_training_probe_at_8192=probes, scope=report["scope"], automatic_adoption=False))
    lines = ["Matched training curricula: same initial model, LR per update, examples and per-event noise.",
        "Staged: 8192 PDB-only then 4096 MD-heavy updates. Control: 12288 interleaved updates.",
        "Each arm: 36864 PDB and 12288 MD examples; 899 static PDB proteins, five MD train sequences.",
        "Final new-sequence results are reused development, not independent test.", ""]
    lines.extend("PRIMARY " + json.dumps(r, sort_keys=True) for r in macro)
    lines.append("")
    lines.extend("TRAIN_ONLY_PROBE_AT_8192 " + json.dumps(r, sort_keys=True) for r in probes)
    lines.append("")
    for r in rows:
        lines.append(f"{r['arm']} seed{r['seed']} {r['split']} {r['protein']} {r['outlet']}: "
            f"CA {r['mean_sample_nearest_reference_ca_rmsd_A']:.4f} Å; "
            f"coverage error {r['mean_reference_nearest_sample_ca_rmsd_A']:.4f} Å; "
            f"CN {r['geometry']['cn_rate']*100:.3f}%; "
            f"spread {r['mean_pairwise_ca_rmsd_A']:.4f}/ref {r['reference_mean_pairwise_ca_rmsd_A']:.4f} Å; "
            f"JS {r['js_pairwise_distance_divergence_nats']:.5f}")
    (RUN / "result_summary.txt").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42,
        "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    panels=[
        (lambda r:r["mean_sample_nearest_reference_ca_rmsd_A"],"Sample to nearest reference: CA RMSD (Å)"),
        (lambda r:100*r["geometry"]["cn_rate"],"C–N abnormal bonds (%)"),
        (lambda r:r["js_pairwise_distance_divergence_nats"],"Pairwise-distance JS divergence (nats)"),
        (lambda r:r["mean_pairwise_ca_rmsd_A"]/r["reference_mean_pairwise_ca_rmsd_A"],"Generated / reference ensemble spread")]
    for axis,(value,label) in zip(axes.flat,panels):
        for index,protein in enumerate(proteins):
            for seed,marker,delta in [(2026,"o",-.04),(2027,"^",.04)]:
                points=[]
                for arm,offset,color in [(ARMS[0],-.16,"#2369a5"),(ARMS[1],.16,"#c68124")]:
                    row=next(r for r in primary if r["protein"]==protein and r["arm"]==arm and r["seed"]==seed)
                    x,y=index+offset+delta,value(row)
                    axis.scatter(x,y,marker=marker,color=color,s=48,zorder=3)
                    points.append((x,y))
                axis.plot([x for x,_ in points],[y for _,y in points],color=".7",lw=.8)
        if label.startswith("Generated"):
            axis.axhline(1.,ls="--",color=".4",lw=1)
        axis.set_xticks(range(len(proteins)),proteins)
        axis.set_ylabel(label)
        axis.grid(axis="y",alpha=.17)
    handles=[plt.Line2D([],[],color="#2369a5",marker="o",ls="",label="Interleaved"),
        plt.Line2D([],[],color="#c68124",marker="o",ls="",label="PDB then MD"),
        plt.Line2D([],[],color=".4",marker="o",ls="",label="Seed 2026"),
        plt.Line2D([],[],color=".4",marker="^",ls="",label="Seed 2027")]
    fig.legend(handles=handles,frameon=False,loc="upper center",ncol=4,bbox_to_anchor=(.5,.95))
    fig.suptitle("PDB pretraining versus interleaved generator training",y=.995,fontsize=14)
    fig.text(.5,.016,"Same training-event and noise multiset · Fixed representation and ESM3 features · Reused development proteins",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.9))
    DESTINATION.mkdir(parents=True,exist_ok=True)
    for extension in ["pdf","png","svg"]:
        fig.savefig(DESTINATION/("Generator_Curriculum_Comparison."+extension),dpi=200)
    plt.close(fig)
    print("GENERATOR_CURRICULUM_FIGURE_READY",DESTINATION,flush=True)


if __name__ == "__main__":
    main()
