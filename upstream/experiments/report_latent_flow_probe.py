"""Complete development feasibility report, not matched-compute model superiority."""
import argparse,json
from pathlib import Path
import numpy as np
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/"protocol.json").read_text());results={}
    for mode in ["fm","regression"]:
        data=[json.loads((a.run/mode/"decoded"/(name+".json")).read_text()) for name in protocol["geometry_targets"]];assert len(data)==24
        effects=[];errors=[];cn=[];ca=[];latent=[];accuracy=[];quality_errors=0
        for row in data:
            samples=row["samples"];n=row["length"];current=[x["metrics"]["closest_available_reference_ca_rmsd"] for x in samples]
            if any(v is None for v in current):raise ValueError("Undefined primary reference metric")
            error=float(np.mean(current));errors.append(error);effects.append(error-row["baseline"]["closest_available_reference_ca_rmsd"])
            cn.append(float(np.mean([x["metrics"]["backbone_cn_outliers"]/(n-1) for x in samples])));ca.append(float(np.mean([x["metrics"]["upstream_ca_clashes"]/n for x in samples])));latent.append(float(np.mean([x["latent_mse_to_oracle"] for x in samples])));accuracy.append(float(np.mean([x["fine_match_to_oracle"] for x in samples])));quality_errors+=sum(x["quality_error"] is not None for x in samples)
        results[mode]=dict(training=json.loads((a.run/mode/"summary.json").read_text()),mean_reference_ca_rmsd=float(np.mean(errors)),mean_original_native_ca_rmsd=float(np.mean([x["baseline"]["closest_available_reference_ca_rmsd"] for x in data])),mean_oracle_reconstruction_ca_rmsd=float(np.mean([x["oracle"]["closest_available_reference_ca_rmsd"] for x in data])),paired_reference_error_minus_original=interval(effects),mean_cn_rate=float(np.mean(cn)),mean_ca_clash_density=float(np.mean(ca)),mean_latent_mse=float(np.mean(latent)),mean_fine_accuracy=float(np.mean(accuracy)),quality_error_samples=quality_errors,records=data)
    differences=[]
    for fm_row,reg_row in zip(results["fm"]["records"],results["regression"]["records"]):
        if fm_row["target"]!=reg_row["target"]:raise ValueError("Paired target order mismatch")
        fm_error=float(np.mean([sample["metrics"]["closest_available_reference_ca_rmsd"] for sample in fm_row["samples"]]))
        reg_error=float(np.mean([sample["metrics"]["closest_available_reference_ca_rmsd"] for sample in reg_row["samples"]]))
        differences.append(fm_error-reg_error)
    atomic_json(a.run/"comparison.json",dict(status="complete",scope=protocol["scope"],methods=results,fm_minus_regression_reference_error=interval(differences),limitations="One training seed per method, five epochs, reused validation cohort. FM16 ODE steps/two noise samples versus regression one forward/one output; not matched inference compute. Frozen condition comes from one sequence-only original coarse draw per protein, not a new full-generation ensemble. Training objectives differ and their absolute values are not comparable. No58/64 test access or backbone/decoder/codebook updates."))
    print(json.dumps({k:{kk:vv for kk,vv in x.items() if kk not in ["records","training"]} for k,x in results.items()},indent=2))


if __name__=="__main__":main()
