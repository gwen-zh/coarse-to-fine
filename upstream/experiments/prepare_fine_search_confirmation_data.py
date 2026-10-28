"""Prepare a new outcome-blind sequence-screened cohort; no model evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from Bio import Align
from prepare_pdb_training import fetch,close_match
from diagnostic_io import atomic_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    out=a.output
    for name in ["raw","metadata","backbone"]:(out/name).mkdir(parents=True,exist_ok=True)
    corpus=Path("artifacts/pdb_training_v2");benchmark=Path("artifacts/development/manifest.json")
    old=json.loads((corpus/"manifest.json").read_text())["targets"];bench=json.loads(benchmark.read_text())["all_groups"]
    references=[(x["id"],x["sequence"]) for x in old+bench]
    snapshot=corpus/"search_snapshot.json";candidates=json.loads(snapshot.read_text())["identifiers"]
    attempted=set(x.stem for x in (corpus/"metadata").glob("*.json"))
    attempted.update(x["id"] for x in old)
    rejectfile=corpus/"rejections.json"
    if rejectfile.exists():
        rejected=json.loads(rejectfile.read_text())
        if isinstance(rejected,list):attempted.update(x["id"] for x in rejected if "id" in x)
    candidates=sorted(set(candidates)-attempted,key=lambda x:hashlib.sha256(("fine-search-confirmation-v1:"+x).encode()).hexdigest())
    protocol=dict(version=1,target=64,candidate_order="SHA256 fine-search-confirmation-v1:<ID>; unused identifiers from original5000 RCSB snapshot",candidate_ids=candidates,
                  exclusions="All attempted metadata identifiers from prior corpus and all prior retained corpus sequences; compare sequences against all1000 training/validation and262 original benchmark groups plus current retained test sequences",
                  source_sha256={str(snapshot):sha(snapshot),str(corpus/"manifest.json"):sha(corpus/"manifest.json"),str(benchmark):sha(benchmark)},
                  selection="No model scores. Same Xray40-256-residue <=3A source query and complete contiguous N/CA/C checks as corpus",
                  similarity="Local alignment >=30percent identity among aligned pairs at >=80percent shorter-sequence coverage; heuristic not certified homology separation or pretraining-unseen",
                  scope="New short experimental protein cohort for a frozen inference method; not thermodynamic ensemble/IDP evaluation")
    pp=out/"protocol.json"
    if pp.exists():assert json.loads(pp.read_text())==protocol
    atomic_json(pp,protocol)
    aligner=Align.PairwiseAligner(mode="local",match_score=2,mismatch_score=-1,open_gap_score=-5,extend_gap_score=-.5)
    retained=[];rejected=[]
    for identifier in candidates:
        row=fetch(identifier,out)
        if "error" in row:
            rejected.append(dict(id=identifier,reason=row["error"]))
        else:
            match=next((name for name,seq in references if close_match(row["sequence"],seq,aligner)),None)
            if match:rejected.append(dict(id=identifier,reason="sequence_similarity",matched=match))
            else:
                path=Path("backbone")/(identifier+".npz");np.savez_compressed(out/path,backbone=row["backbone"])
                retained.append(dict(id=identifier,entry=row["entry"],auth_chain=row["auth_chain"],sequence=row["sequence"],sequence_hash=hashlib.sha256(row["sequence"].encode()).hexdigest(),length=len(row["sequence"]),backbone=str(path),raw=row["raw"],raw_sha256=row["raw_sha256"],subset="confirmation"))
                references.append((identifier,row["sequence"]))
                print("NEW_CONFIRMATION_RETAINED",identifier,len(retained),flush=True)
        atomic_json(out/"manifest.json",dict(protocol_sha256=sha(pp),targets=retained,status="complete" if len(retained)==64 else "collecting"))
        atomic_json(out/"rejections.json",rejected)
        if len(retained)==64:break
    if len(retained)!=64:raise RuntimeError("Insufficient screened candidates; no evaluation authorized on incomplete cohort")
    print("NEW_CONFIRMATION_DATA_COMPLETE",64,flush=True)


if __name__=="__main__":main()
