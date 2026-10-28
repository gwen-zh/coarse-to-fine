"""Two prespecified primary comparisons, protein-cluster bootstrap."""
import argparse,json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    d=json.loads((a.run/"independent_backbone_quality.json").read_text());rows=d["records"];proteins=sorted(set(r["protein"] for r in rows));assert len(proteins)==64 and len(rows)==512
    result={}
    for key in ["reference_ca_c_n_angle_mae_degrees","reference_c_n_ca_angle_mae_degrees"]:
        values=np.array([np.mean([r["metrics"]["discrete_topk"][key]-r["metrics"]["baseline"][key] for r in rows if r["protein"]==protein]) for protein in proteins])
        rng=np.random.default_rng(2026);draws=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(1)
        result[key]=dict(mean_delta=float(values.mean()),bootstrap95=np.quantile(draws,[.025,.975]).tolist(),bootstrap97_5=np.quantile(draws,[.0125,.9875]).tolist(),proteins=64)
    atomic_json(a.run/"primary_confirmation.json",dict(status="complete",scope="Two prespecified angle-error endpoints; Bonferroni nominal intervals, not exact finite-sample guarantee. Reference-error noninferiority and full-model superiority not established by these endpoints alone.",primary=result))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
