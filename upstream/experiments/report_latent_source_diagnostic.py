"""Report every frozen source-swap, fine-support and solver diagnostic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from latent_source_diagnostic import METHODS

RUN=Path("artifacts/latent_source_diagnostic_v1")
FIGURE=Path("paper/figures/latent_source_diagnostic_v1")


def mean_metrics(rows):
    keys=set.intersection(*(set(v) for v in rows))
    return {key:float(np.mean([v[key] for v in rows])) for key in sorted(keys)}


def main():
    from run_latent_source_diagnostic import verify
    p=json.loads((RUN/"protocol.json").read_text());verify(p)
    summaries=[];support=[];endpoint=[];coarse=[];timing=[]
    for seed in p["seeds"]:
        result=json.loads((RUN/f"seed{seed}"/"complete.json").read_text())
        if len(result["records"])!=p["cases_per_model"]:raise RuntimeError("Incomplete diagnostic")
        if not result["generator_unchanged"] or not result["exact_production_replay_verified"]:
            raise RuntimeError("Incorrect fixed model / sampler")
        timing.append(dict(seed=seed,seconds=result["seconds"]))
        for domain in ["PDB_train","MD_train"]:
            rows=[r for r in result["records"] if r["domain"]==domain]
            proteins=sorted({r["protein"] for r in rows})
            for method in METHODS:
                per_protein=[]
                for name in proteins:
                    values=[m["metrics"] for row in rows if row["protein"]==name
                            for m in row["methods"] if m["method"]==method]
                    if len(values)!=4:raise RuntimeError("Missing fixed draws")
                    per_protein.append(dict(protein=name,metrics=mean_metrics(values)))
                summaries.append(dict(seed=seed,domain=domain,method=method,
                    proteins=len(proteins),metrics=mean_metrics([r["metrics"] for r in per_protein]),
                    per_protein=per_protein))
            for name in ["native","true_coarse_generated","generated_coarse_generated","cycle"]:
                values=[mean_metrics([r["support"][name] for r in rows if r["protein"]==protein])
                        for protein in proteins]
                support.append(dict(seed=seed,domain=domain,source=name,**mean_metrics(values)))
            for t in [.1,.5,.9]:
                values=[mean_metrics([v for r in rows if r["protein"]==protein
                                     for v in r["endpoint_probe"] if v["time"]==t]) for protein in proteins]
                endpoint.append(dict(seed=seed,domain=domain,**mean_metrics(values)))
            values=[mean_metrics([{key:r[key] for key in ["coarse_aligned_ca_center_rmsd_A",
                "generated_adjacent_center_distance_mean_A","native_adjacent_center_distance_mean_A",
                "fine_cycle_latent_rmse"]} for r in rows if r["protein"]==protein]) for protein in proteins]
            coarse.append(dict(seed=seed,domain=domain,**mean_metrics(values)))
    atomic_json(RUN/"report.json",dict(status="complete",summaries=summaries,
        fine_support=support,endpoint_prediction=endpoint,coarse_diagnostics=coarse,timing=timing,
        scope=p["scope"],oracle_policy=p["oracle_policy"],controls=p["controls"],
        model_default_replaced=False,training_performed=False,
        test_coordinates_opened=False,validation_coordinates_evaluated=False,R3_opened=False))
    lines=["Frozen generator diagnostic: training proteins only.",p["oracle_policy"],p["controls"],
           "C-N abnormal means bond length <1.0 or >1.8 A; all fixed paths retained.",
           "Within each domain every protein has equal weight; two seeds reported separately.",""]
    for row in summaries:
        m=row["metrics"]
        lines.append(f"{row['domain']} seed{row['seed']} {row['method']}: CA {m['ca_rmsd']:.4f} A; "
            f"CN {100*m['cn_rate']:.3f}%; within {100*m['within_cn_rate']:.3f}%; "
            f"boundary {100*m['boundary_cn_rate']:.3f}%; CA clashes/residue {m['ca_clashes_per_residue']:.5f}")
    lines += ["", "SUPPORT"]+[json.dumps(v,sort_keys=True) for v in support]
    lines += ["", "COARSE"]+[json.dumps(v,sort_keys=True) for v in coarse]
    lines += ["", "ONE-STEP ENDPOINT DIAGNOSTICS (paired corrupted native training example)"]
    lines += [json.dumps(v,sort_keys=True) for v in endpoint]
    (RUN/"result_summary.txt").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
        "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,3,figsize=(12,7.5))
    methods=METHODS[:4]
    labels=["Native C\nNative F","Native C\nGenerated F","Generated C\nRe-encoded native F","Generated C\nGenerated F"]
    for index,domain in enumerate(["PDB_train","MD_train"]):
        for axis,(key,scale,title) in zip(axes[index],[("ca_rmsd",1.,"CA RMSD (Å)"),
                ("cn_rate",100.,"C–N abnormalities (%)"),("boundary_cn_rate",100.,"Block-boundary C–N abnormalities (%)")]):
            for seed,color,marker in [(2026,"#24679e","o"),(2027,"#c27b24","^")]:
                vals=[next(r for r in summaries if r["seed"]==seed and r["domain"]==domain and r["method"]==method)["metrics"][key]*scale for method in methods]
                axis.plot(range(4),vals,marker=marker,color=color,label=f"Seed {seed}",lw=1.2)
            axis.set_xticks(range(4),labels,fontsize=8)
            axis.set_ylabel(title);axis.grid(axis="y",alpha=.17)
            axis.set_title(("8 PDB training proteins" if domain=="PDB_train" else "5 MD training proteins"))
    axes[0,0].legend(frameon=False)
    fig.suptitle("Where does generation fail? Coarse and fine source controls",fontsize=14)
    fig.text(.5,.018,"Training examples only · Native/oracle paths use structural inputs · Same frozen representation and generators",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.95));FIGURE.mkdir(parents=True,exist_ok=True)
    for ext in ["png","pdf","svg"]:fig.savefig(FIGURE/("Latent_Source_Diagnostic."+ext),dpi=180)
    plt.close(fig)
    print("LATENT_SOURCE_DIAGNOSTIC_COMPLETE",len(summaries),flush=True)


if __name__=="__main__":main()
