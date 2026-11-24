"""Readable completed eight-target results, including the reconstruction floor."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha


def main():
    run=Path("artifacts/fine_few_targets_v1");report=json.loads((run/"report.json").read_text())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for seed,color,marker in [(2026,"#24679e","o"),(2027,"#c27b24","^")]:
        trials={r["arm"]:r for r in report["rows"] if r["seed"]==seed}
        phases=[trials["fixed_total"]["initial"],trials["fixed_total"]["final"],trials["matched_presentations"]["final"]]
        ca=[];cn=[];mse=[]
        for rows in phases:
            values=[next(v for v in row["evaluation"]["rows"] if v["nfe"]==16) for row in rows]
            ca.append(np.mean([v["metrics"]["ca_rmsd"] for v in values]));cn.append(100*np.mean([v["metrics"]["cn_rate"] for v in values]))
            mse.append(np.mean([row["evaluation"]["mean_teacher_velocity_mse"] for row in rows]))
        for ax,values in [(axes[0,0],ca),(axes[0,1],cn),(axes[1,1],mse)]:ax.plot(range(3),values,marker=marker,color=color,label=f"Seed {seed}")
        axes[1,0].plot([0,1],ca[1:],marker=marker,color=color)
    native=.31234474724316136
    for ax in [axes[0,0],axes[1,0]]:ax.axhline(native,color=".5",ls="--",lw=1)
    for ax in [axes[0,0],axes[0,1],axes[1,1]]:ax.set_xticks(range(3),["Parent","2,048 updates","16,384 updates"])
    axes[1,0].set_xticks([0,1],["2,048 updates","16,384 updates"])
    axes[1,0].set_title("Final errors near native reconstruction",fontsize=10)
    axes[0,0].set_ylabel("CA RMSD (Å)");axes[0,1].set_ylabel("C–N abnormalities (%)")
    axes[1,0].set_ylabel("CA RMSD (Å), expanded scale");axes[1,1].set_ylabel("Fine velocity MSE (new noise)");axes[1,1].set_yscale("log")
    axes[0,0].legend(frameon=False)
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18)
    fig.suptitle("Eight training structures: longer training reaches the reconstruction floor",fontsize=13)
    fig.text(.5,.018,"Native coarse supplied · Training targets only · Dashed line: native encode/decode · No ensemble-quality claim",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));folder=Path("paper/figures/fine_few_targets_v1")
    for ext in ["pdf","png","svg"]:fig.savefig(folder/("Fine_Eight_Targets_Readable."+ext),dpi=180)
    plt.close(fig)
    atomic_json(run/"figure_rendering.json",dict(status="complete",source_sha256=sha(__file__),
        input_report_sha256=sha(run/"report.json"),metrics_changed=False,
        figure=str(folder/"Fine_Eight_Targets_Readable.pdf")))


if __name__=="__main__":main()
