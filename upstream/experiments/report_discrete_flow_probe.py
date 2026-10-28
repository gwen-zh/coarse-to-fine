"""Equal-NFE development report, no geometry gates or best-arm selection."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());rows=[json.loads((a.run/"decoded"/(x["id"]+".json")).read_text()) for x in protocol["targets"]];assert len(rows)==24
    results={};paired={};records=[]
    for steps in protocol["coarse_steps"]:
        results[str(steps)]={};paired[str(steps)]={};protein_values={}
        for arm in protocol["arms"]:
            name=arm["name"];values=[]
            for row in rows:
                current=row["methods"][str(steps)][name];samples=current["samples"];n=row["length"];assert len(samples)==8
                def complete_mean(key):
                    data=[x["metrics"].get(key) for x in samples]
                    return float(np.mean(data)) if all(x is not None for x in data) else None
                q=[]
                for key in ["reference_ca_c_n_angle_mae_degrees","reference_c_n_ca_angle_mae_degrees"]:
                    data=[x["quality"].get(key) if x["quality"] else None for x in samples];q.append(float(np.mean(data)) if all(x is not None for x in data) else None)
                cn=complete_mean("backbone_cn_outliers");ca=complete_mean("upstream_ca_clashes")
                values.append(dict(protein=row["target"],reference_ca_rmsd=complete_mean("closest_available_reference_ca_rmsd"),cn_rate=cn/(n-1) if cn is not None else None,ca_clashes_per_residue=ca/n if ca is not None else None,ca_clash_validity=float(np.mean([bool(x["metrics"].get("upstream_ca_clash_valid")) for x in samples])),finite_ca_fraction=float(np.mean([bool(x["metrics"].get("all_ca_finite")) for x in samples])),ca_c_n_angle_error=q[0],c_n_ca_angle_error=q[1],spread=current["mean_pairwise_ca_rmsd"],best_reference_ca_rmsd=current["best_reference_ca_rmsd"],generation_seconds=float(np.mean([x["sampling_seconds"] for x in samples])),remask_events_per_sample=float(np.mean([x["remask_events"] for x in samples])),revised_tokens_per_sample=float(np.mean([x["revised_token_events"] for x in samples]))))
            def summarize(key):
                data=[x[key] for x in values]
                return dict(mean=float(np.mean(data)) if all(x is not None for x in data) else None,defined_proteins=sum(x is not None for x in data),expected_proteins=24)
            results[str(steps)][name]={key:summarize(key) for key in values[0] if key!="protein"};results[str(steps)][name]["backbone_calls_per_sample"]=steps+2
            protein_values[name]=values;records.extend([dict(steps=steps,arm=name,**x) for x in values])
        for arm in protocol["arms"][1:]:
            name=arm["name"];paired[str(steps)][name]={}
            for key in ["reference_ca_rmsd","cn_rate","ca_clashes_per_residue","ca_c_n_angle_error","c_n_ca_angle_error","spread","generation_seconds"]:
                values=[]
                for x,y in zip(protein_values[name],protein_values["native"]):
                    assert x["protein"]==y["protein"]
                    values.append(x[key]-y[key] if x[key] is not None and y[key] is not None else None)
                paired[str(steps)][name][key]=interval(values) if all(x is not None for x in values) else dict(mean=None,undefined_proteins=sum(x is None for x in values),note="No silent exclusion or successful paired claim")
    primary=paired["20"]["corrector_b0p5"]["reference_ca_rmsd"]
    atomic_json(a.run/"comparison.json",dict(status="complete",scope=protocol["scope"],primary_endpoint=protocol["primary"],primary=primary,methods=results,paired_differences_from_native=paired,per_protein=records,expected_outputs=24*8*4*2,native_base_full_exact_replay=True,limitations=protocol["limits"]+" Reused development cohort; intervals exploratory. Reference/spread computed after averaging8samples per protein; no inferred thermal ensemble or independent confirmation."))
    with (a.run/"per_protein.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for arm in protocol["arms"]:
        name=arm["name"];x=[steps+2 for steps in protocol["coarse_steps"]]
        for axis,key,factor in [(axes[0],"reference_ca_rmsd",1),(axes[1],"cn_rate",100),(axes[2],"spread",1)]:
            y=[results[str(steps)][name][key]["mean"] for steps in protocol["coarse_steps"]]
            axis.plot(x,[value*factor if value is not None else float("nan") for value in y],"o--" if name=="base_fm" else "o-",label=name,alpha=.8)
            axis.set_xticks(x);axis.set_xlabel("Backbone forward calls per sample")
    axes[0].set_ylabel("Mean reference CA RMSD (A)");axes[1].set_ylabel("CN outliers (%)");axes[2].set_ylabel("Within-protein mean pairwise CA RMSD (A)")
    axes[0].legend(fontsize=8)
    fig.suptitle("24 reused development proteins, 8 samples each; fixed weights")
    for ext in ["png","svg"]:fig.savefig(a.run/("sampler_comparison."+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(primary=primary,methods=results),indent=2))


if __name__=="__main__":main()
