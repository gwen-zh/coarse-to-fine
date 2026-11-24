"""Report every predeclared physical coarse prompt, with the source baselines."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_author_codec_interface import METRIC_KEYS, GEOMETRY_KEYS, load
from run_aa_structural_coarse_condition import ROOT, METHODS, verify


def main():
    p=load(ROOT/"protocol.json");verify(p);rows=[];completed=[]
    for seed in p["seeds"]:
        x=load(ROOT/f"seed{seed}/complete.json")
        if len(x["rows"])!=12 or not x["model_and_codec_unchanged"] or x["optimizer_updates"]!=0:
            raise RuntimeError("Incomplete frozen structural-coarse run")
        rows.extend(x["rows"]);completed.append({k:x[k] for k in ["seed","host","cpu_threads","optimizer_updates","model_and_codec_unchanged"]})
    summary={}
    for method in METHODS:
        selected=[r for r in rows if r["method"]==method]
        if len(selected)!=4:raise RuntimeError("Missing protein/seed group")
        good=all(r["metrics"]["metrics_available"] for r in selected)
        out=dict(metrics_available=good,failure_count=sum(r["metrics"]["failure_count"] for r in selected))
        if good:
            out.update({k:float(np.mean([r["metrics"][k] for r in selected])) for k in METRIC_KEYS})
            out["geometry"]={k:float(np.mean([r["metrics"]["geometry"][k] for r in selected])) for k in GEOMETRY_KEYS}
        out["coarse_metrics"]={k:float(np.mean([r["coarse_metrics"][k] for r in selected])) for k in [
            "mean_sample_nearest_reference_center_rmsd_A","mean_reference_nearest_sample_center_rmsd_A",
            "generated_conditional_span_flag_rate","js_pairwise_center_distance_nats"]}
        summary[method]=out
    # All gauge variants must describe the same physical coarse distance field.
    gauge_checks=[]
    for seed in p["seeds"]:
        for target in p["targets"]:
            selected=[r for r in rows if r["seed"]==seed and r["protein"]==target["protein"] and r["method"].startswith("prior_")]
            for key in ["mean_sample_nearest_reference_center_rmsd_A","mean_reference_nearest_sample_center_rmsd_A"]:
                values=[r["coarse_metrics"][key] for r in selected]
                np.testing.assert_allclose(values,values[0],atol=2e-4,rtol=2e-5)
            gauge_checks.append(dict(seed=seed,protein=target["protein"],physical_coarse_error_unchanged_by_rotation=True))
    result=dict(status="complete_awaiting_review",summary=summary,rows=rows,completed=completed,gauge_checks=gauge_checks,
        all_samples_retained=True,optimizer_updates=0,independent_test_claim=False,test_coordinates_opened=False,
        R3_opened=False,model_default_replaced=False,scope=p["scope"],protocol_sha256=sha(ROOT/"protocol.json"),
        report_source_sha256=sha(__file__))
    if all(x["metrics_available"] for x in summary.values()):
        labels=["Own","Author","Encoded\nauthor","Prior:\nraw","Prior:\nfirst residue","Prior:\nalign to own"]
        panels=[("mean_sample_nearest_reference_ca_rmsd_A","Generated → reference (Å)",False,1),
            ("mean_reference_nearest_sample_ca_rmsd_A","Reference → generated (Å)",False,1),
            ("cn_rate","C–N outliers (%)",True,100),
            ("reference_ca_c_n_angle_mae_degrees","CA–C–N angle error (degrees)",True,1),
            ("reference_c_n_ca_angle_mae_degrees","C–N–CA angle error (degrees)",True,1),
            ("js_pairwise_distance_divergence_nats","Pair-distance JS divergence (nats)",False,1)]
        fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
        for ax,(key,label,geom,scale) in zip(axes.flat,panels):
            values=[(summary[m]["geometry"] if geom else summary[m])[key]*scale for m in METHODS]
            ax.bar(range(6),values,color=["#737373","#b5b5b5","#a67c52","#3577a7","#4e9b87","#8064a2"])
            for i,value in enumerate(values):ax.annotate(f"{value:.2f}",(i,value),xytext=(0,3),textcoords="offset points",ha="center",fontsize=8)
            ax.set_xticks(range(6),labels,fontsize=8);ax.set_ylabel(label);ax.set_ylim(0,max(values)*1.2+1e-9)
            ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Can stronger AA-only coarse prompts rescue our frozen fine generator?\n"
            "2 reused development proteins · 2 seeds · 64 raw samples/group · original weights/decoder unchanged",fontsize=12)
        folder=Path("paper/figures/aa_structural_coarse_condition_v1");folder.mkdir(parents=True,exist_ok=True)
        for ext in ["pdf","png","svg"]:fig.savefig(folder/("AA_Structural_Coarse_Condition."+ext),dpi=170)
        plt.close(fig);result["figure"]=str(folder/"AA_Structural_Coarse_Condition.pdf")
    atomic_json(ROOT/"report.json",result)
    print("AA_STRUCTURAL_COARSE_REPORT",json.dumps(summary),flush=True)


if __name__=="__main__":main()
