"""Lightweight complete-cohort paired report; no Torch imports."""
import argparse
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from structural_metrics import aligned_rmsd


def interval(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all():raise ValueError("Nonfinite paired difference")
    rng=np.random.default_rng(2026)
    draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),bootstrap95=np.quantile(draws,[.025,.975]).tolist(),proteins=len(values))


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/"protocol.json").read_text())
    manifest=json.loads((Path(protocol["source"])/"manifest.json").read_text())
    targets=manifest["targets"];groups={};records=[]
    for item in targets:
        result=json.loads((a.run/item["target"]/"summary.json").read_text())
        assert result["target"]==item["target"]
        group=item.get("protein",item["target"])
        groups.setdefault(group,[]).append((item,result))
    output={}
    for arm in protocol["arms"]:
        per_protein=[]
        for protein,items in groups.items():
            differences=[];base_errors=[];selected_errors=[];cn=[];ca=[];edit=[];motion=[];seconds=[];rolled=0;base_bb=[];selected_bb=[]
            for item,r in items:
                x=r["arms"][arm];b=x["posthoc_baseline_reference_metrics"];s=x["posthoc_reference_metrics"]
                differences.append(s["closest_available_reference_ca_rmsd"]-b["closest_available_reference_ca_rmsd"])
                base_errors.append(b["closest_available_reference_ca_rmsd"]);selected_errors.append(s["closest_available_reference_ca_rmsd"])
                with np.load(Path(protocol["source"])/item["input"]) as z:n=len(z["fine"])
                cn.append(100*(s["backbone_cn_outliers"]-b["backbone_cn_outliers"])/max(1,n-1))
                ca.append((s["upstream_ca_clashes"]-b["upstream_ca_clashes"])/n)
                edit.append(x["gate"]["changed_fraction"]);motion.append(x["gate"]["aligned_ca_displacement"])
                seconds.append(x["search_seconds"]);rolled+=int(x["returned_original"])
                if len(items)>1:
                    with np.load(a.run/item["target"]/(arm+".npz")) as z:selected_bb.append(z["backbone"][:,1].copy())
                    with np.load(a.run/item["target"]/"baseline.npz") as z:base_bb.append(z["backbone"][:,1].copy())
            row=dict(protein=protein,samples=len(items),reference_rmsd_delta=float(np.mean(differences)),
                     gated_cn_rate_delta_pp=float(np.mean(cn)),gated_ca_clash_density_delta=float(np.mean(ca)),
                     mean_edited_fraction=float(np.mean(edit)),mean_ca_displacement=float(np.mean(motion)),
                     mean_search_seconds=float(np.mean(seconds)),rollback_count=rolled)
            if len(items)>1:
                def spread(arrays):return float(np.mean([aligned_rmsd(x,y) for i,x in enumerate(arrays) for y in arrays[i+1:]]))
                row.update(baseline_spread=spread(base_bb),selected_spread=spread(selected_bb),
                           baseline_best_reference_rmsd=min(base_errors),selected_best_reference_rmsd=min(selected_errors))
                for label,arrays in [("baseline",base_bb),("selected",selected_bb)]:
                    pairs=[aligned_rmsd(x,y) for i,x in enumerate(arrays) for y in arrays[i+1:]]
                    row[label+"_pairwise_ca_rmsd_quantiles"]=np.quantile(pairs,[.1,.5,.9]).tolist()
                row["spread_delta"]=row["selected_spread"]-row["baseline_spread"]
            per_protein.append(row)
        output[arm]=dict(per_protein=per_protein,paired_reference_rmsd_delta=interval([x["reference_rmsd_delta"] for x in per_protein]),
                         proteins_with_lower_mean_reference_error=sum(x["reference_rmsd_delta"]<0 for x in per_protein),
                         mean_gated_cn_rate_delta_pp=float(np.mean([x["gated_cn_rate_delta_pp"] for x in per_protein])),
                         returned_original=sum(x["rollback_count"] for x in per_protein),samples=len(targets))
        if all("spread_delta" in x for x in per_protein):output[arm]["paired_spread_delta"]=interval([x["spread_delta"] for x in per_protein])
    atomic_json(a.run/"comparison.json",dict(status="complete",scope=protocol["scope"],expected_samples=len(targets),completed_samples=len(targets),arms=output,
        limitations="New fixed confirmation cohort; secondary reference/spread intervals over proteins, not proof of equivalence. Primary ungated quality endpoints are reported separately. Gated proxy gains follow acceptance policy. Same nominal search time is not equal FLOPs; final operation overruns reported in raw results."))
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":main()
