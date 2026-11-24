"""Matched fine-flow/geometry continuation, with the fixed parent as reference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from run_fine_geometry_flow import RUN,PARENT,DIAGNOSTIC,ARMS,verify


def average(rows):
    keys=set.intersection(*(set(row) for row in rows))
    return {key:float(np.mean([row[key] for row in rows])) for key in sorted(keys)}


def main():
    p=json.loads((RUN/"protocol.json").read_text());verify(p)
    trials=[];pairing=[];pdb=[]
    for seed in p["seeds"]:
        initial=[];traces=[]
        trials.append(dict(arm="parent",seed=seed,
            evaluation=json.loads((PARENT/"interleaved"/f"seed{seed}"/"evaluation.json").read_text())))
        old=json.loads((DIAGNOSTIC/f"seed{seed}"/"complete.json").read_text())
        before={(r["case_id"],r["draw"]):next(m["metrics"] for m in r["methods"]
            if m["method"]=="true_coarse_generated_fine16") for r in old["records"] if r["domain"]=="PDB_train"}
        for arm in ARMS:
            out=RUN/arm/f"seed{seed}"
            done=json.loads((out/"complete.json").read_text())
            if done["updates"]!=p["updates"] or not done["coarse_weights_unchanged"] or not done["codec_weights_unchanged"]:
                raise RuntimeError("Incorrect fine training state")
            if not done["normalization_buffers_unchanged"]:raise RuntimeError("Changed normalization")
            init=json.loads((out/"initial.json").read_text());initial.append(init)
            trace=json.loads((out/"trace.json").read_text())
            if len(trace)!=p["updates"]:raise RuntimeError("Incomplete trace")
            traces.append([(r["batch"],r["draws"],r["rollout_case"],r["rollout_draw"],r["lr"]) for r in trace])
            trial=dict(arm=arm,seed=seed,complete=done,
                evaluation=json.loads((out/"evaluation.json").read_text()))
            trials.append(trial)
            probe=json.loads((out/"PDB_oracle_probe.json").read_text())["rows"]
            if set(before)!={(r["case_id"],r["draw"]) for r in probe}:raise RuntimeError("PDB probe pairing differs")
            for protein in sorted({r["protein"] for r in probe}):
                after_rows=[r for r in probe if r["protein"]==protein]
                after=average([r["metrics"] for r in after_rows])
                prior=average([before[(r["case_id"],r["draw"])] for r in after_rows])
                pdb.append(dict(arm=arm,seed=seed,protein=protein,before=prior,after=after,
                    delta={k:after[k]-prior[k] for k in after if k in prior}))
        keys=["fine_sha256","coarse_sha256","codec_sha256","trainable_parameters"]
        if any(initial[0][k]!=initial[1][k] for k in keys) or traces[0]!=traces[1]:
            raise RuntimeError("Fine training arms are not paired")
        if not all(i["epoch0_sampler_bitwise_replayed"] for i in initial):raise RuntimeError("Epoch0 mismatch")
        pairing.append(dict(seed=seed,updates=p["updates"],initial_weights_and_examples_noise_LR_identical=True,
                            coarse_codec_and_normalization_frozen=True))
    macro=[]
    for trial in trials:
        for outlet in ["full","oracle_coarse"]:
            values=[r for r in trial["evaluation"]["rows"] if r["split"]=="validation" and r["outlet"]==outlet]
            if len(values)!=2:raise RuntimeError("Expected both development proteins")
            entry={key:float(np.mean([v[key] for v in values])) for key in [
                "mean_sample_nearest_reference_ca_rmsd_A","mean_reference_nearest_sample_ca_rmsd_A",
                "js_pairwise_distance_divergence_nats","rmsf_mae_A","mean_full_generation_seconds"]}
            entry.update({key:float(np.mean([v["geometry"][key] for v in values])) for key in [
                "cn_rate","boundary_cn_rate","ca_clashes_per_residue",
                "reference_ca_c_n_angle_mae_degrees","reference_c_n_ca_angle_mae_degrees"]})
            entry["mean_spread_ratio"]=float(np.mean([v["mean_pairwise_ca_rmsd_A"]/
                v["reference_mean_pairwise_ca_rmsd_A"] for v in values]))
            macro.append(dict(arm=trial["arm"],seed=trial["seed"],outlet=outlet,**entry))
    result=dict(status="complete",rows=trials,macro_per_seed=macro,PDB_training_oracle_probe=pdb,
        matching=pairing,scope=p["scope"],geometry_weight=p["geometry_weights"],
        same_raw_samples_retained=True,no_reference_based_acceptance=True,
        independent_test_claim=False,model_default_replaced=False,test_coordinates_opened=False,R3_opened=False)
    atomic_json(RUN/"report.json",result)
    atomic_json(RUN/"primary_summary.json",dict(status="complete",macro_per_seed=macro,
        matching=pairing,scope=p["scope"],automatic_adoption=False))
    lines=["Fine-only continuation with immutable coarse and structural codec.",p["objective"],
        "Reference is the fixed parent. Both continued arms see the same data, Gaussian/time draws and updates.",
        "All samples retained. Two reused development proteins; no independent-test claim.",""]
    lines += [json.dumps(r,sort_keys=True) for r in macro]
    lines += ["","PDB TRAINING / ORACLE-COARSE PROBE"]
    for arm in ARMS:
        for seed in p["seeds"]:
            rows=[r for r in pdb if r["arm"]==arm and r["seed"]==seed]
            lines.append(json.dumps(dict(arm=arm,seed=seed,before=average([r["before"] for r in rows]),
                after=average([r["after"] for r in rows])),sort_keys=True))
    (RUN/"result_summary.txt").write_text("\n".join(lines)+"\n")
    primary=[dict(v,arm=t["arm"],seed=t["seed"]) for t in trials for v in t["evaluation"]["rows"]
             if v["split"]=="validation" and v["outlet"]=="full"]
    proteins=sorted({r["protein"] for r in primary})
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    panels=[(lambda r:r["mean_sample_nearest_reference_ca_rmsd_A"],"CA error to nearest reference (Å)"),
        (lambda r:100*r["geometry"]["cn_rate"],"C–N abnormal bonds (%)"),
        (lambda r:r["js_pairwise_distance_divergence_nats"],"Pairwise-distance JS divergence (nats)"),
        (lambda r:r["mean_pairwise_ca_rmsd_A"]/r["reference_mean_pairwise_ca_rmsd_A"],"Generated / reference ensemble spread")]
    configs=[("parent",-.22,"#777777","Parent"),(ARMS[0],0.,"#24679e","Fine flow only"),
             (ARMS[1],.22,"#c27b24","Fine flow + geometry")]
    for ax,(value,label) in zip(axes.flat,panels):
        for index,protein in enumerate(proteins):
            for seed,marker,delta in [(2026,"o",-.035),(2027,"^",.035)]:
                points=[]
                for arm,offset,color,_ in configs:
                    r=next(r for r in primary if r["arm"]==arm and r["seed"]==seed and r["protein"]==protein)
                    x,y=index+offset+delta,value(r);ax.scatter(x,y,color=color,marker=marker,s=45,zorder=3);points.append((x,y))
                ax.plot([x for x,y in points],[y for x,y in points],color=".8",lw=.8)
        if label.startswith("Generated"):ax.axhline(1.,ls="--",color=".5",lw=1)
        ax.set_xticks(range(len(proteins)),proteins);ax.set_ylabel(label);ax.grid(axis="y",alpha=.17)
    handles=[plt.Line2D([],[],color=color,marker="o",ls="",label=label) for _,_,color,label in configs]
    handles += [plt.Line2D([],[],color=".4",marker=m,ls="",label=f"Seed {seed}") for seed,m in [(2026,"o"),(2027,"^")]]
    fig.legend(handles=handles,frameon=False,loc="upper center",ncol=3,bbox_to_anchor=(.5,.95))
    fig.suptitle("Fine generation: flow versus exact geometry supervision",fontsize=14,y=.995)
    fig.text(.5,.018,"Fixed coarse sampler and structure maps · Matched continued training · Reused development proteins",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.87))
    folder=Path("paper/figures/fine_geometry_flow_v1");folder.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(folder/("Fine_Geometry_Flow_Comparison."+ext),dpi=200)
    plt.close(fig)
    print("FINE_GEOMETRY_REPORT_COMPLETE",len(trials),flush=True)


if __name__=="__main__":main()
