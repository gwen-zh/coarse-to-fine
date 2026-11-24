"""Matched decoder-only adaptation report; every fixed endpoint is retained."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from decoder_angle_adaptation import ARMS
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_author_codec_interface import METRIC_KEYS, GEOMETRY_KEYS
from run_decoder_angle_adaptation import ROOT, PRIOR, load, verify, identity


def summarize(rows,key):
    good=all(r[key]["metrics_available"] for r in rows)
    result=dict(metrics_available=good,failure_count=sum(r[key]["failure_count"] for r in rows))
    if good:
        result.update({k:float(np.mean([r[key][k] for r in rows])) for k in METRIC_KEYS})
        result["geometry"]={k:float(np.mean([r[key]["geometry"][k] for r in rows])) for k in GEOMETRY_KEYS}
    return result


def main():
    p=load(ROOT/"protocol.json");verify(p);all_rows=[];runs=[];paired=[]
    for seed in p["seeds"]:
        traces=[]
        for arm in ARMS:
            folder=ROOT/arm/f"seed{seed}";done=load(folder/"complete.json");evaluation=load(folder/"evaluation.json")
            if done["updates"]!=2048 or not done["frozen_encoder_book_coarse_unchanged"] or len(evaluation["rows"])!=4:
                raise RuntimeError("Incomplete fixed decoder endpoint")
            rows=load(folder/"trace.json")
            if len(rows)!=2048:raise RuntimeError("Incomplete decoder training trace")
            traces.append([identity(r) for r in rows]);all_rows.extend(evaluation["rows"])
            runs.append(dict(**done,native_oracle=evaluation["native_oracle"]))
        if traces[0]!=traces[1]:raise RuntimeError("Decoder-arm training draws/LR differ")
        paired.append(dict(seed=seed,all2048_actual_training_events_and_LR_equal=True))
    summaries={}
    for outlet in ["author_roundtrip","own_frozen_generator"]:
        selected=[r for r in all_rows if r["outlet"]==outlet]
        baseline_rows=[r for r in selected if r["arm"]==ARMS[0]]
        for r in baseline_rows:
            other=next(x for x in selected if x["arm"]==ARMS[1] and x["seed"]==r["seed"] and x["protein"]==r["protein"])
            for key in METRIC_KEYS:np.testing.assert_allclose(r["baseline"][key],other["baseline"][key],atol=1e-10,rtol=1e-10)
        summaries[outlet]={"original_decoder":summarize(baseline_rows,"baseline")}
        for arm in ARMS:summaries[outlet][arm]=summarize([r for r in selected if r["arm"]==arm],"adapted")
    author=load(PRIOR/"complete.json")["two_seed_development_macro"]["original"]
    report=dict(status="complete_awaiting_review",scope=p["scope"],summaries=summaries,
        original_author=author,paired_audit=paired,runs=runs,rows=all_rows,
        all_samples_retained=True,independent_test_claim=False,test_coordinates_opened=False,R3_opened=False,
        model_default_replaced=False,protocol_sha256=sha(ROOT/"protocol.json"),report_source_sha256=sha(__file__))
    if all(v["metrics_available"] for group in summaries.values() for v in group.values()):
        panels=[("mean_sample_nearest_reference_ca_rmsd_A","Generated → reference (Å)",False,1),
            ("mean_reference_nearest_sample_ca_rmsd_A","Reference → generated (Å)",False,1),
            ("cn_rate","C–N outliers (%)",True,100),
            ("reference_ca_c_n_angle_mae_degrees","CA–C–N error (degrees)",True,1),
            ("reference_c_n_ca_angle_mae_degrees","C–N–CA error (degrees)",True,1),
            ("js_pairwise_distance_divergence_nats","Pair-distance JS (nats)",False,1)]
        fig,axes=plt.subplots(2,6,figsize=(18,7),constrained_layout=True)
        for row_index,outlet in enumerate(summaries):
            names=["Original decoder","Continuation","Angle objective"];groups=list(summaries[outlet].values())
            colors=["#737373","#699c74","#3577a7"]
            if outlet=="author_roundtrip":names=["Raw author"]+names;groups=[author]+groups;colors=["#b5b5b5"]+colors
            for ax,(key,label,geom,scale) in zip(axes[row_index],panels):
                values=[(g["geometry"] if geom else g)[key]*scale for g in groups]
                ax.bar(range(len(values)),values,color=colors,width=.7)
                for index,value in enumerate(values):ax.annotate(f"{value:.2f}",(index,value),xytext=(0,3),textcoords="offset points",ha="center",fontsize=7)
                ax.set_xticks(range(len(values)),names,rotation=45,ha="right",fontsize=7)
                ax.set_title(label,fontsize=9);ax.set_ylim(0,max(values)*1.23+1e-9)
                ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True);ax.spines[["top","right"]].set_visible(False)
            axes[row_index,0].set_ylabel("Author → own representation" if row_index==0 else "Own frozen generator")
        fig.suptitle("Decoder-only adaptation with frozen coarse/fine representations\n"
            "2 reused development proteins · 2 paired training-draw seeds · 2048 updates · no inference selection",fontsize=12)
        folder=Path("paper/figures/decoder_angle_adaptation_v1");folder.mkdir(parents=True,exist_ok=True)
        for ext in ["pdf","png","svg"]:fig.savefig(folder/("Decoder_Angle_Adaptation."+ext),dpi=160)
        plt.close(fig);report["figure"]=str(folder/"Decoder_Angle_Adaptation.pdf")
    atomic_json(ROOT/"report.json",report)
    print("DECODER_ANGLE_REPORT",json.dumps(summaries),flush=True)


if __name__=="__main__":main()
