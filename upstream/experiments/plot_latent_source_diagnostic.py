"""Readable rendering of already frozen results; no metrics or models change."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha


def main():
    run=Path("artifacts/latent_source_diagnostic_v1")
    report=json.loads((run/"report.json").read_text())
    methods=["native_reconstruction","true_coarse_generated_fine16",
             "generated_coarse_reencoded_native_fine","full16"]
    labels=["Native coarse\nNative fine","Native coarse\nSampled fine",
            "Sampled coarse\nNative fine\n(re-encoded)","Sampled coarse\nSampled fine"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(14,8.5))
    for index,domain in enumerate(["PDB_train","MD_train"]):
        for axis,(key,scale,title) in zip(axes[index],[("ca_rmsd",1.,"CA RMSD to paired structure (Å)"),
                ("cn_rate",100.,"C–N abnormalities (%)"),("boundary_cn_rate",100.,"Block-boundary C–N abnormalities (%)")]):
            for seed,color,marker in [(2026,"#24679e","o"),(2027,"#c27b24","^")]:
                vals=[next(r for r in report["summaries"] if r["seed"]==seed and
                    r["domain"]==domain and r["method"]==method)["metrics"][key]*scale for method in methods]
                axis.plot(range(4),vals,marker=marker,color=color,label=f"Seed {seed}",lw=1.2)
            axis.set_xticks(range(4),labels,fontsize=8);axis.set_xlim(-.22,3.22)
            axis.set_ylabel(title);axis.grid(axis="y",alpha=.17)
            axis.set_title("8 PDB training proteins" if domain=="PDB_train" else "5 MD training proteins")
    axes[0,0].legend(frameon=False)
    fig.suptitle("Coarse and fine source controls",fontsize=16)
    fig.text(.5,.028,"Training proteins only · Oracle paths use native structural inputs · 4 fixed draws per protein and seed",ha="center",fontsize=10)
    fig.text(.5,.008,"MD reference error uses the paired frame here; it is not the sample-to-nearest-ensemble error from the main evaluation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95),h_pad=2.5,w_pad=2.)
    folder=Path("paper/figures/latent_source_diagnostic_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(folder/("Latent_Source_Diagnostic_Readable."+ext),dpi=180)
    plt.close(fig)
    atomic_json(run/"figure_rendering.json",dict(status="complete",
        source=str(Path(__file__)),source_sha256=sha(__file__),
        input_report_sha256=sha(run/"report.json"),metrics_changed=False,
        figure=str(folder/"Latent_Source_Diagnostic_Readable.pdf")))


if __name__=="__main__":main()
