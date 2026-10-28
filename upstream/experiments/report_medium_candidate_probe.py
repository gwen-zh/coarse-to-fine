"""Complete paired proposal ablation; gated proxies versus ungated quality separated."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from report_bounded_fine_search import interval
from structural_metrics import aligned_rmsd
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());targets=protocol["targets"];records=[]
    for item in targets:
        d=json.loads((a.run/item["target"]/"summary.json").read_text());assert d["status"]=="complete";n=d["length"]
        for arm in protocol["arms"]:
            x=d["arms"][arm];assert x["decoder_calls"]==item["decoder_call_budget"];m=x["posthoc_metrics"];b=d["baseline"];q=x["posthoc_quality"];bq=d["baseline_quality"]
            records.append(dict(protein=item["protein"],target=item["target"],arm=arm,reference_ca_rmsd=m.get("closest_available_reference_ca_rmsd"),reference_rmsd_delta=m["closest_available_reference_ca_rmsd"]-b["closest_available_reference_ca_rmsd"] if m.get("closest_available_reference_ca_rmsd") is not None else None,cn_rate=m["backbone_cn_outliers"]/(n-1) if "backbone_cn_outliers" in m else None,cn_delta_pp=100*(m["backbone_cn_outliers"]-b["backbone_cn_outliers"])/(n-1) if "backbone_cn_outliers" in m else None,ca_clash_density=m["upstream_ca_clashes"]/n if "upstream_ca_clashes" in m else None,ca_c_n_angle_error=q["reference_ca_c_n_angle_mae_degrees"] if q else None,c_n_ca_angle_error=q["reference_c_n_ca_angle_mae_degrees"] if q else None,ca_c_n_angle_delta=q["reference_ca_c_n_angle_mae_degrees"]-bq["reference_ca_c_n_angle_mae_degrees"] if q else None,c_n_ca_angle_delta=q["reference_c_n_ca_angle_mae_degrees"]-bq["reference_c_n_ca_angle_mae_degrees"] if q else None,search_seconds=x["search_seconds"],decoder_calls=x["decoder_calls"],padding_calls=x["padding_calls"],fallback_proposal_calls=x["fallback_proposal_calls"],accepted_fallback_proposals=sum(t["accepted"] and t["cluster_fallback"] for t in x["trace"]),returned_original=x["returned_original"],ca_displacement=x["gate"]["aligned_ca_displacement"],edited_fraction=x["gate"]["changed_fraction"],overlap_with_head=d["candidate_pool_overlap"][arm]["alternative_overlap_with_head"],overlap_with_fine_neighbors=d["candidate_pool_overlap"][arm]["alternative_overlap_with_fine_neighbors"]))
    assert len(records)==192*4;proteins=sorted(set(x["protein"] for x in records));assert len(proteins)==24
    per_protein={};summary={}
    numeric=[key for key in records[0] if key not in ["protein","target","arm"]]
    for arm in protocol["arms"]:
        selected=[r for r in records if r["arm"]==arm];values=[]
        for protein in proteins:
            group=[r for r in selected if r["protein"]==protein];assert len(group)==8
            row=dict(protein=protein)
            for key in numeric:
                data=[x[key] for x in group];row[key]=float(np.mean(data)) if all(x is not None for x in data) else None
            arrays=[];baselines=[]
            for r in group:
                with np.load(a.run/r["target"]/(arm+".npz")) as z:arrays.append(z["backbone"][:,1].copy())
                with np.load(a.run/r["target"]/"baseline.npz") as z:baselines.append(z["backbone"][:,1].copy())
            def spread(items):
                pairs=[aligned_rmsd(x,y) for i,x in enumerate(items) for y in items[i+1:]]
                return float(np.mean(pairs)) if all(x is not None for x in pairs) else None
            row["spread"]=spread(arrays);row["baseline_spread"]=spread(baselines);row["spread_delta"]=row["spread"]-row["baseline_spread"] if row["spread"] is not None else None
            values.append(row)
        per_protein[arm]={r["protein"]:r for r in values};summary[arm]={}
        for key in values[0]:
            if key=="protein":continue
            data=[r[key] for r in values];summary[arm][key]=dict(mean=float(np.mean(data)) if all(x is not None for x in data) else None,defined_proteins=sum(x is not None for x in data))
        summary[arm]["exact_total_decoder_calls"]=sum(x["decoder_calls"] for x in selected);summary[arm]["total_returned_original"]=sum(x["returned_original"] for x in selected)
    def paired(left,right):
        result={}
        for key in ["reference_ca_rmsd","cn_rate","ca_clash_density","ca_c_n_angle_error","c_n_ca_angle_error","spread","search_seconds"]:
            data=[]
            for protein in proteins:
                x,y=per_protein[left][protein][key],per_protein[right][protein][key];data.append(x-y if x is not None and y is not None else None)
            result[key]=interval(data) if all(x is not None for x in data) else dict(mean=None,undefined_proteins=sum(x is None for x in data))
        return result
    comparisons={"repaired_minus_head":paired("repaired_cluster","head_topk"),"repaired_minus_direct_fine_neighbors":paired("repaired_cluster","fine_neighbors"),"repaired_minus_original_medium":paired("repaired_cluster","original_cluster"),"fine_neighbors_minus_head":paired("fine_neighbors","head_topk")}
    result=dict(status="complete",scope=protocol["scope"],samples=192,search_outputs=768,methods=summary,paired_comparisons=comparisons,codebook_geometry=protocol["geometry"],per_protein=per_protein,limitations="Repeated development cohort. CN/CA improvement relative to initial output is enforced by gate and not independent efficacy evidence. Same-medium-center assignment is not nested fine-cell geometry and not guaranteed spatial similarity. Repaired decoder embedding patch not used. All four methods share GPUFP32 and exact stored CPUcall budget; no cross-hardware equivalence/speed claim. No-op budget padding and fallback counts explicit.")
    atomic_json(a.run/"comparison.json",result)
    with (a.run/"per_sample.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True);names=protocol["arms"]
    for axis,key,label,factor in [(axes[0],"reference_ca_rmsd","Reference CA RMSD (A)",1),(axes[1],"cn_rate","Gated CN outliers (%)",100),(axes[2],"ca_c_n_angle_error","Ungated reference CA-C-N angle MAE (degrees)",1)]:
        vals=[summary[name][key]["mean"] for name in names];axis.bar(range(4),[v*factor if v is not None else float("nan") for v in vals]);axis.set_xticks(range(4),names,rotation=30,ha="right");axis.set_ylabel(label)
    fig.suptitle("Same192 original outputs, exact decoder-call budget; reused development cohort")
    for ext in ["png","svg"]:fig.savefig(a.run/("candidate_comparison."+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(methods=summary,paired_comparisons=comparisons),indent=2))


if __name__=="__main__":main()
