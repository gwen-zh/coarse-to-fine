"""Presentation-only redraw with readable labels; original frozen report kept."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hierarchical_pipeline_io import sha


def main():
    root=Path("artifacts/aa_structural_coarse_condition_v1")
    data=json.loads((root/"report.json").read_text());summary=data["summary"]
    methods=["own","raw_author","encoded_author","prior_raw","prior_first_residue","prior_align_to_own"]
    labels=["Own","Author","Encoded\nauthor","Prior\nraw","Prior\nfirst-res.","Prior\naligned"]
    panels=[("mean_sample_nearest_reference_ca_rmsd_A","Generated → reference (Å)",False,1),
        ("mean_reference_nearest_sample_ca_rmsd_A","Reference → generated (Å)",False,1),
        ("cn_rate","C–N outliers (%)",True,100),
        ("reference_ca_c_n_angle_mae_degrees","CA–C–N angle error (degrees)",True,1),
        ("reference_c_n_ca_angle_mae_degrees","C–N–CA angle error (degrees)",True,1),
        ("js_pairwise_distance_divergence_nats","Pair-distance JS divergence (nats)",False,1)]
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
    for ax,(key,label,geom,scale) in zip(axes.flat,panels):
        values=[(summary[m]["geometry"] if geom else summary[m])[key]*scale for m in methods]
        ax.bar(range(6),values,color=["#737373","#b5b5b5","#a67c52","#3577a7","#4e9b87","#8064a2"])
        for i,value in enumerate(values):ax.annotate(f"{value:.2f}",(i,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
        ax.set_xticks(range(6),labels,fontsize=9);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.2+1e-9)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("AA-only physical coarse prompts with our frozen fine generator\n"
        "2 reused development proteins · 2 seeds · 64 raw samples/group · no training or selection",fontsize=13)
    folder=Path("paper/figures/aa_structural_coarse_condition_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["pdf","png","svg"]:fig.savefig(folder/("AA_Structural_Coarse_Condition_v2."+ext),dpi=170)
    plt.close(fig)
    audit=dict(presentation_only=True,metric_values_unchanged=True,
        source_sha256={str(p):sha(p) for p in [Path(__file__),root/"protocol.json"]},
        figure=str(folder/"AA_Structural_Coarse_Condition_v2.pdf"))
    (root/"figure_v2.json").write_text(json.dumps(audit,indent=2)+"\n")


if __name__=="__main__":main()
