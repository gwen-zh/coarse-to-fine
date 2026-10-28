"""Complete-cohort pilot summary with parent and repair effects separated."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from structural_metrics import aligned_rmsd
from report_bounded_fine_search import interval


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"])
    items=json.loads((source/"manifest.json").read_text())["targets"];assert len(items)==len(protocol["proteins"])*4
    results={};records=[]
    for item in items:
        d=json.loads((a.run/item["target"]/"summary.json").read_text());assert d["protein"]==item["protein"] and d["parent"]==item["parent"]
        with np.load(source/item["input"]) as z:n=len(z["fine"])
        methods={"baseline":dict(metrics=d["baseline"],quality=d["baseline_ungated_quality"],search_seconds=0.,motion=0.,edits=0.,returned_original=True)}
        for arm,v in d["arms"].items():methods[arm]=dict(metrics=v["posthoc_metrics"],quality=v["posthoc_ungated_quality"],search_seconds=v["search_seconds"],motion=v["gate"]["aligned_ca_displacement"],edits=v["gate"]["changed_fraction"],returned_original=v["returned_original"])
        for method,v in methods.items():
            m=v["metrics"];row=dict(protein=item["protein"],parent=item["parent"],sample=item["sample"],method=method,reference_ca_rmsd=m["closest_available_reference_ca_rmsd"],cn_rate_percent=100*m["backbone_cn_outliers"]/(n-1),ca_clashes_per_residue=m["upstream_ca_clashes"]/n,search_seconds=v["search_seconds"],motion=v["motion"],edited_fraction=v["edits"],returned_original=v["returned_original"],**v["quality"]);records.append(row)
    for parent in protocol["parent_runs"]:
        results[parent]={}
        for method in ["baseline"]+protocol["arms"]:
            group=[x for x in records if x["parent"]==parent and x["method"]==method]
            keys=["reference_ca_rmsd","cn_rate_percent","ca_clashes_per_residue","search_seconds","motion","edited_fraction","reference_ca_c_n_angle_mae_degrees","reference_c_n_ca_angle_mae_degrees"]
            results[parent][method]={k:float(np.mean([x[k] for x in group])) for k in keys}
            results[parent][method]["returned_original_count"]=sum(x["returned_original"] for x in group)
    def difference(parent1,method1,parent0,method0,key):
        values=[]
        for protein in protocol["proteins"]:
            def mean(parent,method):return float(np.mean([x[key] for x in records if x["protein"]==protein and x["parent"]==parent and x["method"]==method]))
            values.append(mean(parent1,method1)-mean(parent0,method0))
        return interval(values)
    effects={}
    for arm in protocol["arms"]:
        effects[arm]=dict(adapted_repair_minus_adapted_baseline=difference("adapted",arm,"adapted","baseline","reference_ca_rmsd"),control_repair_minus_control_baseline=difference("control",arm,"control","baseline","reference_ca_rmsd"),adapted_repaired_minus_control_repaired=difference("adapted",arm,"control",arm,"reference_ca_rmsd"))
    effects["lora_baseline_minus_control_baseline"]=difference("adapted","baseline","control","baseline","reference_ca_rmsd")
    atomic_json(a.run/"comparison.json",dict(status="complete",scope=protocol["scope"],completed_inputs=len(items),methods=results,paired_effects=effects,records=records,note="Four reused development proteins; intervals exploratory only. Geometry proxies gated; true bridge distance candidates are not original fine logits. Neither LoRA interaction nor full-generator superiority is confirmed by pilot."))
    print(json.dumps(dict(methods=results,paired_effects=effects),indent=2),flush=True)


if __name__=="__main__":main()
