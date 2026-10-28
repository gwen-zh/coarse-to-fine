"""Build locked inference protocols after outcome-blind data collection."""
import argparse,json,hashlib
from pathlib import Path
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();run=a.run
    manifest=json.loads((run/"manifest.json").read_text());assert manifest["status"]=="complete" and len(manifest["targets"])==64
    plan=json.loads((run/"evaluation_plan.json").read_text())
    for file,digest in plan["source_code_sha256"].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    source=run/"inputs";source.mkdir(exist_ok=True)
    sampling=dict(version=1,source=str(source),corpus=str(run),generator="artifacts/mp_rank_00_model_states.pt",targets=manifest["targets"],samples_per_protein=8,coarse_steps=5,chunks=8,scope=plan["scope"])
    atomic_json(run/"sampling_protocol.json",sampling)
    rows=[dict(target=t["id"]+"_sample"+str(i),protein=t["id"],sample=i,input=t["id"]+"_sample"+str(i)+".npz") for t in manifest["targets"] for i in range(8)]
    atomic_json(source/"manifest.json",dict(targets=rows))
    protocol=json.loads(Path("artifacts/development/bounded_fine_search_ensemble_30s_v1/protocol.json").read_text())
    protocol.update(source=str(source),scope=plan["scope"]+"; new fixed independent confirmation, no tuning",input_sha256={},sampling_protocol_sha256=hashlib.sha256((run/"sampling_protocol.json").read_bytes()).hexdigest(),evaluation_plan_sha256=hashlib.sha256((run/"evaluation_plan.json").read_bytes()).hexdigest())
    atomic_json(run/"protocol.json",protocol)
    print("CONFIRMATION_SETUP_COMPLETE",len(rows),flush=True)


if __name__=="__main__":main()
