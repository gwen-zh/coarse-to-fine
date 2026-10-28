"""Prepare full40 and safely reuse exact first4 pilot inputs/results."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import prepare_lora_fine_search
from diagnostic_io import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);args=parser.parse_args()
    prepare_lora_fine_search.main()
    protocol=json.loads((args.run/"protocol.json").read_text());pilot=Path(protocol["reuse_pilot"]);old=json.loads((pilot/"protocol.json").read_text())
    for key in ["topk","delta_angstrom","edit_fraction_cap","seconds_per_arm","arms","bridge","weights","parent_runs"]:assert protocol[key]==old[key]
    now=json.loads((Path(protocol["source"])/"manifest.json").read_text())["targets"];before=json.loads((Path(old["source"])/"manifest.json").read_text())["targets"]
    assert now[:len(before)]==before and len(before)==16 and len(now)==160
    for item in before:
        with np.load(Path(protocol["source"])/item["input"]) as x,np.load(Path(old["source"])/item["input"]) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:assert np.array_equal(x[key],y[key],equal_nan=True)
        destination=args.run/item["target"]
        if destination.exists():raise ValueError("Refuse to replace existing output")
        shutil.copytree(pilot/item["target"],destination)
    atomic_json(args.run/"reused_pilot.json",dict(inputs=16,proteins=4,source=str(pilot),verification="All input arrays identical, same parent configuration and same indices for random seeds; not counted as additional independent observations"))
    print("FULL40_PREPARED_REUSED16",flush=True)


if __name__=="__main__":main()
