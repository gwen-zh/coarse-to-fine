"""Seal generated input hashes after all sequence-only sampling finishes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"])
    manifest=json.loads((source/"manifest.json").read_text());hashes={}
    for item in manifest["targets"]:
        path=source/item["input"]
        with np.load(path) as z:
            assert np.array_equal(z["logits"].argmax(-1),z["fine"])
            assert np.isfinite(z["logits"]).all()
        hashes[item["target"]]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert not protocol["input_sha256"] or protocol["input_sha256"]==hashes
    protocol["input_sha256"]=hashes
    atomic_json(a.run/"protocol.json",protocol)
    print("ENSEMBLE_INPUTS_LOCKED",len(hashes),flush=True)


if __name__=="__main__":main()
