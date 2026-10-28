"""Prepare four fixed development proteins, control/adapted bridge outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ESMProtein,ROOT
from esm.models.residual_bridge import load_residual_bridge
from bridge_search_candidates import ranked_candidates
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();torch.set_num_threads(8)
    protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"]);source.mkdir(exist_ok=True)
    bridge,_=load_residual_bridge(protocol["bridge"])
    groups={x["id"]:x for x in json.loads(Path("artifacts/development/manifest.json").read_text())["all_groups"]}
    targets=[];hashes={}
    for name in protocol["proteins"]:
        assert groups[name]["split"]=="development"
        protein=ESMProtein.from_pdb(ROOT/groups[name]["paths"][0]);assert protein.sequence==groups[name]["sequence"]
        reference=protein.coordinates[:,:3].numpy()
        for parent,folder in protocol["parent_runs"].items():
            for sample in range(2):
                key=f"{name}_{parent}_sample{sample}"
                pred=Path(folder)/"predictions"/f"{name}_sample{sample}.npz"
                saved=Path(folder)/"decoded"/f"{name}_sample{sample}.npz"
                with np.load(pred) as z:tokens=torch.from_numpy(z["tokens"].copy()).long()
                fine=bridge.fine_tokens(tokens).numpy();latent=bridge.prefix_latent(tokens);codes=bridge.fine_codes
                distance=latent.square().sum(-1,keepdim=True)-2*latent@codes.T+codes.square().sum(-1)
                candidates=ranked_candidates(distance.numpy(),fine,protocol["topk"])
                with np.load(saved) as z:
                    if not np.array_equal(z["fine"],fine):raise ValueError("Fine bridge differs from cached initial output")
                    expected=z["backbone"].copy()
                path=source/(key+".npz")
                np.savez_compressed(path,fine=fine,candidates=candidates,reference_backbone=reference,expected_backbone=expected)
                hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
                targets.append(dict(target=key,protein=name,parent=parent,sample=sample,input=path.name))
                print("LORA_SEARCH_PREPARED",key,flush=True)
    atomic_json(source/"manifest.json",dict(targets=targets))
    protocol["input_sha256"]=hashes
    atomic_json(a.run/"protocol.json",protocol)


if __name__=="__main__":main()
