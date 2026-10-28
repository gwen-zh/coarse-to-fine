"""Audit all saved original and repaired structures without fitting or generation."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from backbone_quality_audit import measurements
from report_bounded_fine_search import interval
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--corpus",type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"])
    manifest=json.loads((source/"manifest.json").read_text());refs={x["id"]:x for x in json.loads((a.corpus/"manifest.json").read_text())["targets"]}
    rows=[];hashes={}
    for item in manifest["targets"]:
        target=item["target"];protein=item["protein"]
        refpath=a.corpus/refs[protein]["backbone"]
        with np.load(refpath) as z:reference=z["backbone"].copy()
        directory=a.run/target
        paths={"baseline":directory/"baseline.npz",**{arm:directory/(arm+".npz") for arm in protocol["arms"]}}
        metrics={}
        for arm,path in paths.items():
            with np.load(path) as z:metrics[arm]=measurements(z["backbone"],reference)
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(target=target,protein=protein,metrics=metrics))
    assert len(rows)==len(manifest["targets"])
    proteins=sorted(set(x["protein"] for x in rows));summary={}
    for arm in protocol["arms"]:
        summary[arm]={}
        for key in rows[0]["metrics"]["baseline"]:
            changes=[];base=[];selected=[]
            for protein in proteins:
                group=[x for x in rows if x["protein"]==protein]
                b=float(np.mean([x["metrics"]["baseline"][key] for x in group]));s=float(np.mean([x["metrics"][arm][key] for x in group]))
                base.append(b);selected.append(s);changes.append(s-b)
            summary[arm][key]=dict(baseline_mean=float(np.mean(base)),selected_mean=float(np.mean(selected)),paired_delta=interval(changes))
    atomic_json(a.run/"independent_backbone_quality.json",dict(status="complete",samples=len(rows),proteins=len(proteins),scope="Posthoc ungated diagnostics on reused development proteins, not independent test confirmation; reference-relative angle errors are conformation dependent; no all-atom or Ramachandran validity claim. Intra-residue bonds are decoder controls.",arms=summary,records=rows,input_sha256=hashes))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
