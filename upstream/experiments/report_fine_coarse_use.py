"""Paired within-coarse effects for the frozen 2x2 fine-conditioning control."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from fine_coarse_use import CELLS, COMPARISONS, validate_cells
from hierarchical_pipeline_io import sha
from run_author_codec_interface import GEOMETRY_KEYS, METRIC_KEYS, load
from run_fine_coarse_use import ROOT, verify


def summarize(rows):
    good=all(r["metrics"]["metrics_available"] for r in rows)
    result=dict(metrics_available=good,failure_count=sum(r["metrics"]["failure_count"] for r in rows))
    if good:
        result.update({key:float(np.mean([r["metrics"][key] for r in rows])) for key in METRIC_KEYS})
        for family,field in [("geometry","geometry"),("fixed_coarse_reference_geometry",None)]:
            result[family]={key:float(np.mean([r["metrics"][field][key] if field else r[family][key] for r in rows])) for key in GEOMETRY_KEYS}
    return result


def main():
    p=load(ROOT/"protocol.json");verify(p);rows=[];pairs=[];runs=[]
    for seed in p["seeds"]:
        value=load(ROOT/f"seed{seed}/complete.json")
        if len(value["rows"])!=24 or len(value["pairs"])!=12 or not value["codec_unchanged"] or not value["diagonal_backbones_bitwise_replayed"]:
            raise RuntimeError("Incomplete crossed evaluation")
        for target in p["targets"]:
            for gauge in p["gauges"]:
                validate_cells([r for r in value["rows"] if r["protein"]==target["protein"] and r["gauge"]==gauge])
        rows.extend(value["rows"]);pairs.extend(value["pairs"])
        runs.append({k:value[k] for k in ["seed","host","cpu_threads","optimizer_updates","new_scored_fine_sampler_calls","diagonal_backbones_bitwise_replayed"]})
    summaries={};deltas={}
    for gauge in p["gauges"]:
        summaries[gauge]={cell:summarize([r for r in rows if r["gauge"]==gauge and r["cell"]==cell]) for cell in CELLS}
        deltas[gauge]={}
        for comparison in COMPARISONS:
            selected=[r for r in pairs if r["gauge"]==gauge and r["comparison"]==comparison]
            if len(selected)!=4:raise RuntimeError("Missing paired protein/seed groups")
            good=all(r["metrics_available"] for r in selected);value=dict(metrics_available=good)
            if good:
                value.update({key:float(np.mean([r["changed_minus_held"][key] for r in selected])) for key in METRIC_KEYS})
                for field in ["geometry","fixed_coarse_reference_geometry"]:
                    value[field]={key:float(np.mean([r["changed_minus_held"][field][key] for r in selected])) for key in GEOMETRY_KEYS}
                value["mean_direct_CA_displacement_A"]=float(np.mean([r["mean_direct_CA_displacement_A"] for r in selected]))
                value["fine_response"]={key:float(np.mean([r["fine_response"][key] for r in selected])) for key in selected[0]["fine_response"]}
            deltas[gauge][comparison]=value
    # The original ownC/ownF cell is the same cached sample in all gauge repetitions.
    for key in METRIC_KEYS:
        values=[summaries[g]["ownC_ownF"][key] for g in p["gauges"]]
        np.testing.assert_allclose(values,values[0],atol=0,rtol=0)
    result=dict(status="complete_awaiting_review",summaries=summaries,changed_minus_held=deltas,
        rows=rows,pairs=pairs,runs=runs,primary_gauge=p["primary_gauge"],scope=p["scope"],
        optimizer_updates=0,new_scored_fine_sampler_calls=0,all_samples_retained=True,
        independent_test_claim=False,test_coordinates_opened=False,R3_opened=False,model_default_replaced=False,
        protocol_sha256=sha(ROOT/"protocol.json"),report_source_sha256=sha(__file__))
    with (ROOT/"paired_differences.csv").open("w",newline="") as stream:
        fields=["seed","protein","gauge","comparison","mean_direct_CA_displacement_A",
            *METRIC_KEYS,*GEOMETRY_KEYS]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in pairs:
            if not row["metrics_available"]:continue
            value={key:row[key] for key in fields[:5]}
            value.update({key:row["changed_minus_held"][key] for key in METRIC_KEYS})
            value.update({key:row["changed_minus_held"]["fixed_coarse_reference_geometry"][key] for key in GEOMETRY_KEYS})
            writer.writerow(value)
    if all(v["metrics_available"] for gauge in summaries.values() for v in gauge.values()):
        panels=[("mean_sample_nearest_reference_ca_rmsd_A","Reference CA RMSD (Å)",False,1,3),
            ("cn_rate","C–N outliers (%)",True,100,2),
            ("reference_ca_c_n_angle_mae_degrees","CA–C–N error (degrees)",True,1,2),
            ("reference_c_n_ca_angle_mae_degrees","C–N–CA error (degrees)",True,1,2)]
        fig,axes=plt.subplots(3,4,figsize=(14,11),constrained_layout=True)
        for column,(key,label,geometry,scale,digits) in enumerate(panels):
            values=[(v["geometry"] if geometry else v)[key]*scale for group in summaries.values() for v in group.values()]
            lo,hi=min(values),max(values)
            for index,gauge in enumerate(p["gauges"]):
                ax=axes[index,column];group=summaries[gauge]
                def metric(cell):return (group[cell]["geometry"] if geometry else group[cell])[key]*scale
                matrix=np.array([[metric("ownC_ownF"),metric("ownC_priorF")],
                    [metric("priorC_ownF"),metric("priorC_priorF")]])
                im=ax.imshow(matrix,vmin=lo,vmax=hi,cmap="YlGnBu")
                for (i,j),value in np.ndenumerate(matrix):
                    ax.text(j,i,f"{value:.{digits}f}",ha="center",va="center",
                        color="white" if value>(lo+hi)/2 else "black",fontsize=11)
                ax.set_xticks([0,1],["Fine | own C","Fine | prior C"],fontsize=8)
                ax.set_yticks([0,1],["Own C","Prior C"],fontsize=9)
                if column==0:ax.set_ylabel({"raw":"Raw orientation","first_residue":"First-residue frame","align_to_own":"Aligned to own"}[gauge],fontsize=10)
                if index==0:ax.set_title(label,fontsize=10)
                ax.spines[:].set_visible(False)
            fig.colorbar(im,ax=axes[:,column].tolist(),shrink=.85,pad=.02)
        fig.suptitle("Does regenerating fine under the new coarse condition help?\n"
            "Rows: decoder coarse · Columns: fine sampling condition · Same AA / paired noise / frozen decoder\n"
            "2 reused development proteins · 2 seeds · 64 paired samples/group · all three gauges reported",fontsize=12)
        folder=Path("paper/figures/fine_coarse_use_v1");folder.mkdir(parents=True,exist_ok=True)
        for ext in ["pdf","png","svg"]:fig.savefig(folder/("Fine_Coarse_Use."+ext),dpi=160)
        plt.close(fig);result["figure"]=str(folder/"Fine_Coarse_Use.pdf")
    atomic_json(ROOT/"report.json",result)
    print("FINE_COARSE_USE_REPORT",json.dumps(dict(summaries=summaries,changed_minus_held=deltas)),flush=True)


if __name__=="__main__":main()
