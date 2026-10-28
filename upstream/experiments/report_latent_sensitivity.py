"""Complete descriptive robustness curves; no generator-feasibility verdict."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json
from report_bounded_fine_search import interval


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());targets=[];records=[]
    for item in protocol["targets"]:
        d=json.loads((a.run/item["target"]/"summary.json").read_text());assert d["status"]=="complete" and d["requested_conditions"]==55
        base=d["records"][0]
        for row in d["records"]:
            m=row["metrics"];b=base["metrics"];q=row["ungated_quality"];bq=base["ungated_quality"]
            record=dict(target=d["target"],length=d["length"],kind=row["kind"],family=row["family"],alpha=row.get("alpha"),replicate=row["replicate"],changed_fraction=row["changed_fraction"],ca_displacement=row["ca_displacement"],max_ca_displacement=row["max_ca_displacement"],noise_rms=row["noise_rms"],noise_relative_quantization_rms=row["noise_relative_quantization_rms"],reference_rmsd_delta=(m.get("closest_available_reference_ca_rmsd")-b["closest_available_reference_ca_rmsd"]) if m.get("closest_available_reference_ca_rmsd") is not None else None,cn_rate_delta_pp=100*(m["backbone_cn_outliers"]-b["backbone_cn_outliers"])/(d["length"]-1) if "backbone_cn_outliers" in m else None,ca_clash_density_delta=(m["upstream_ca_clashes"]-b["upstream_ca_clashes"])/d["length"] if "upstream_ca_clashes" in m else None,ca_c_n_angle_delta=q["reference_ca_c_n_angle_mae_degrees"]-bq["reference_ca_c_n_angle_mae_degrees"] if q is not None and bq is not None else None,side=row.get("side"),margin_quantile=row.get("margin_quantile"),expected_crossing_observed=row.get("expected_crossing_observed"),quality_error=row["ungated_quality_error"])
            records.append(record)
        targets.append(dict(target=d["target"],requested=d["requested_conditions"],unique_decodes=d["unique_decoder_calls"],decoder_seconds=d["decoder_seconds"],quantization_rms=d["quantization_rms"],cell_margin_quantiles=d["cell_margin_quantiles"]))
    def summarize(group,key,only_changed=False):
        values=[];defined=0
        for item in protocol["targets"]:
            sample=[r[key] for r in group if r["target"]==item["target"] and r[key] is not None and (not only_changed or r["changed_fraction"]>0)]
            defined+=len(sample)
            if sample:values.append(float(np.mean(sample)))
        return dict(expected_conditions=len(group),defined_conditions=defined,defined_proteins=len(values),estimate=interval(values) if values else None)
    curves=[]
    for family in protocol["noise_families"]:
        for alpha in protocol["alphas"]:
            group=[r for r in records if r["kind"]=="gaussian_noise" and r["family"]==family and r["alpha"]==alpha];assert len(group)==len(targets)*protocol["replicates"]
            curves.append(dict(family=family,alpha=alpha,metrics={key:summarize(group,key) for key in ["changed_fraction","ca_displacement","max_ca_displacement","reference_rmsd_delta","cn_rate_delta_pp","ca_clash_density_delta","ca_c_n_angle_delta"]},conditional_on_token_change_ca_displacement=summarize(group,"ca_displacement",True),zero_change_conditions=sum(r["changed_fraction"]==0 for r in group)))
    boundary=[]
    for quantile in [.1,.5,.9]:
        for side in ["below","above"]:
            group=[r for r in records if r["kind"]=="cell_face" and r["margin_quantile"]==quantile and r["side"]==side]
            boundary.append(dict(margin_quantile=quantile,side=side,conditions=len(group),expected_crossing_observed=sum(r["expected_crossing_observed"] for r in group),ca_displacement=summarize(group,"ca_displacement"),max_ca_displacement=summarize(group,"max_ca_displacement"),reference_rmsd_delta=summarize(group,"reference_rmsd_delta")))
    result=dict(status="complete",scope=protocol["scope"],proteins=len(targets),conditions=len(records),unique_decoder_calls=sum(x["unique_decodes"] for x in targets),training_centered_rms=protocol["training_centered_rms"],noise_curves=curves,boundary_probes=boundary,per_protein=targets,quality_error_conditions=sum(r["quality_error"] is not None for r in records),limitations=["Oracle true-structure encoder neighborhood, not samples from a trained generator","Near-zero drift without token changes is a quantization plateau, not evidence of continuous geometric smoothness","Gaussian covariance alignment still need not stay on a physical latent manifold","Boundary sensitivity does not prove continuous FM impossible; good local robustness does not prove conditional generation learnable","Intervals descriptive on reused development proteins; no holdout or FM training performed"])
    atomic_json(a.run/"comparison.json",result)
    with (a.run/"conditions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    colors=dict(isotropic="tab:blue",covariance_oriented="tab:orange")
    for family in protocol["noise_families"]:
        group=[r for r in curves if r["family"]==family]
        axes[0].plot([r["alpha"] for r in group],[100*r["metrics"]["changed_fraction"]["estimate"]["mean"] for r in group],"o-",label=family,color=colors[family])
        points=[r for r in records if r["kind"]=="gaussian_noise" and r["family"]==family and r["ca_displacement"] is not None]
        axes[1].scatter([100*r["changed_fraction"] for r in points],[r["ca_displacement"] for r in points],s=8,alpha=.25,color=colors[family])
    axes[0].set(xscale="log",xlabel="Noise RMS / TRAIN centered latent RMS",ylabel="Changed fine tokens (%)");axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Changed fine tokens (%)",ylabel="Aligned CA displacement from initial decode (A)")
    groups=[];labels=[]
    for q in [.1,.5,.9]:
        for side in ["below","above"]:
            groups.append([r["ca_displacement"] for r in records if r["kind"]=="cell_face" and r["margin_quantile"]==q and r["side"]==side and r["ca_displacement"] is not None]);labels.append(f"q{int(100*q)} {side}")
    axes[2].boxplot(groups,tick_labels=labels);axes[2].set(ylabel="CA displacement: single-site boundary probe (A)",yscale="linear",ylim=(0,None));axes[2].tick_params(axis="x",rotation=40)
    for ext in ["png","svg"]:fig.savefig(a.run/("latent_sensitivity."+ext),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(status="complete",proteins=len(targets),conditions=len(records),unique_decoder_calls=result["unique_decoder_calls"],quality_errors=result["quality_error_conditions"]),indent=2))


if __name__=="__main__":main()
