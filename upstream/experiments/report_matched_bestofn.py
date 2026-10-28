"""Mechanism report, retaining zero-budget cases and all proteins."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from report_bounded_fine_search import interval
from structural_metrics import aligned_rmsd


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());rows=[]
    parent=Path(protocol["parent_search"]);qa={x["target"]:x for x in json.loads((parent/"independent_backbone_quality.json").read_text())["records"]}
    for item in protocol["targets"]:
        d=json.loads((a.run/item["target"]/"summary.json").read_text());
        with np.load(Path(protocol["source"])/item["input"]) as z:n=len(z["fine"])
        for arm,v in d["arms"].items():
            assert v["decoder_calls"]==item["decoder_call_budget"]
            m=v["posthoc_metrics"];rows.append(dict(protein=item["protein"],target=item["target"],arm=arm,reference_rmsd_delta=m["closest_available_reference_ca_rmsd"]-d["baseline"]["closest_available_reference_ca_rmsd"],cn_delta_pp=100*(m["backbone_cn_outliers"]-d["baseline"]["backbone_cn_outliers"])/(n-1),ca_c_n_angle_delta=v["posthoc_quality"]["reference_ca_c_n_angle_mae_degrees"]-d["baseline_quality"]["reference_ca_c_n_angle_mae_degrees"],c_n_ca_angle_delta=v["posthoc_quality"]["reference_c_n_ca_angle_mae_degrees"]-d["baseline_quality"]["reference_c_n_ca_angle_mae_degrees"],search_seconds=v["search_seconds"],returned_original=v["returned_original"],calls=v["decoder_calls"],rejection_diagnostics_overlapping=v["rejection_diagnostics_overlapping"]))
        old=json.loads((parent/item["target"]/"summary.json").read_text())["arms"]["discrete_topk"]
        assert old["forward_calls"]==item["decoder_call_budget"]
        bm,sm=old["posthoc_baseline_reference_metrics"],old["posthoc_reference_metrics"]
        q=qa[item["target"]]["metrics"]
        reasons=dict(edit_cap_exceeded=0,displacement_exceeded=0,geometry_pareto_failed=0,eligible=0)
        for row in old["trace"]:
            g=row["gate"];m=row["metrics"]
            reasons["edit_cap_exceeded"]+=int(g["changed_fraction"]>protocol["edit_fraction_cap"])
            reasons["displacement_exceeded"]+=int(g["aligned_ca_displacement"] is None or g["aligned_ca_displacement"]>protocol["delta_angstrom"])
            eligible_geometry=m["backbone_cn_outliers"]<=bm["backbone_cn_outliers"] and m["upstream_ca_clashes"]<=bm["upstream_ca_clashes"] and (m["backbone_cn_outliers"]<bm["backbone_cn_outliers"] or m["upstream_ca_clashes"]<bm["upstream_ca_clashes"])
            reasons["geometry_pareto_failed"]+=int(not eligible_geometry);reasons["eligible"]+=int(g["eligible"])
        rows.append(dict(protein=item["protein"],target=item["target"],arm="existing_targeted",reference_rmsd_delta=sm["closest_available_reference_ca_rmsd"]-bm["closest_available_reference_ca_rmsd"],cn_delta_pp=100*(sm["backbone_cn_outliers"]-bm["backbone_cn_outliers"])/(n-1),ca_c_n_angle_delta=q["discrete_topk"]["reference_ca_c_n_angle_mae_degrees"]-q["baseline"]["reference_ca_c_n_angle_mae_degrees"],c_n_ca_angle_delta=q["discrete_topk"]["reference_c_n_ca_angle_mae_degrees"]-q["baseline"]["reference_c_n_ca_angle_mae_degrees"],search_seconds=old["search_seconds"],returned_original=old["returned_original"],calls=old["forward_calls"],rejection_diagnostics_overlapping=reasons))
    proteins=sorted(set(x["protein"] for x in rows));summary={};per_protein={}
    for arm in protocol["arms"]+["existing_targeted"]:
        selected=[x for x in rows if x["arm"]==arm];means={}
        for key in ["reference_rmsd_delta","cn_delta_pp","ca_c_n_angle_delta","c_n_ca_angle_delta"]:
            means[key]={protein:float(np.mean([x[key] for x in selected if x["protein"]==protein])) for protein in proteins}
        summary[arm]={k:interval(list(v.values())) for k,v in means.items()}
        spread_changes=[]
        for protein in proteins:
            targets=[x["target"] for x in selected if x["protein"]==protein]
            baseline_arrays=[];selected_arrays=[]
            for target in targets:
                with np.load((parent if arm=="existing_targeted" else a.run)/target/"baseline.npz") as z:baseline_arrays.append(z["backbone"][:,1].copy())
                with np.load((parent if arm=="existing_targeted" else a.run)/target/(("discrete_topk" if arm=="existing_targeted" else arm)+".npz")) as z:selected_arrays.append(z["backbone"][:,1].copy())
            def spread(arrays):return float(np.mean([aligned_rmsd(x,y) for i,x in enumerate(arrays) for y in arrays[i+1:]]))
            spread_changes.append(spread(selected_arrays)-spread(baseline_arrays))
        summary[arm]["paired_spread_delta"]=interval(spread_changes)
        summary[arm].update(returned_original=sum(x["returned_original"] for x in selected),samples=len(selected),mean_search_seconds=float(np.mean([x["search_seconds"] for x in selected])),calls=sum(x["calls"] for x in selected),rejections={key:sum(x["rejection_diagnostics_overlapping"][key] for x in selected) for key in selected[0]["rejection_diagnostics_overlapping"]})
        per_protein[arm]=means
    paired={key:interval([per_protein["local_greedy"][key][protein]-per_protein["local_independent"][key][protein] for protein in proteins]) for key in per_protein["local_greedy"]}
    versus_existing={arm:{key:interval([per_protein[arm][key][protein]-per_protein["existing_targeted"][key][protein] for protein in proteins]) for key in per_protein[arm]} for arm in protocol["arms"]}
    atomic_json(a.run/"comparison.json",dict(status="complete",scope=protocol["scope"],arms=summary,greedy_minus_independent_shared_proposals=paired,new_arm_minus_existing_targeted=versus_existing,records=rows,note="All decoder budgets exactly matched. Whole-sequence proposal rejection may be structural to edit/drift caps; local shared-stream comparison separates accumulation. Development-only mechanism analysis, no independent-test or gradient claim."))
    print(json.dumps(dict(arms=summary,greedy_minus_independent_shared_proposals=paired),indent=2))


if __name__=="__main__":main()
