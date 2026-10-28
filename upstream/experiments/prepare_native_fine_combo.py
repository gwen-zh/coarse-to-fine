"""Native original-coarse path, original/plain/retained fine projections only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from evaluate_residual_vq import ESMProtein,ROOT
from fine_retention import build_head
from coarse_lora import restore_trainable_state
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();a.run.mkdir(exist_ok=True,parents=True);torch.set_num_threads(8)
    cache=Path("artifacts/pdb_training_v2/fine_retention_features");fine_run=Path("artifacts/development/fine_retention");projection_path=cache/"fine_projection.pt";projection=torch.load(projection_path,map_location="cpu",weights_only=True)
    rows=[x for x in json.loads((cache/"manifest.json").read_text())["targets"] if x["dataset"]=="diagnostic"];rows=sorted(rows,key=lambda x:(x["sample"],x["id"]));assert len(rows)==80 and len(set(x["id"] for x in rows))==40
    groups={x["id"]:x for x in json.loads(Path("artifacts/development/manifest.json").read_text())["all_groups"]};heads={};snapshots={}
    for parent in ["plain","retained"]:
        path=fine_run/parent/"best_adapter.pt";snapshot=torch.load(path,map_location="cpu",weights_only=True);assert snapshot["epoch"]==5
        head=build_head(projection,rank=snapshot["protocol"]["rank"]);restore_trainable_state(head,snapshot["state"]);head.eval().requires_grad_(False);heads[parent]=head;snapshots[parent]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),epoch=5)
    source=a.run/"inputs";source.mkdir(exist_ok=True);targets=[];hashes={};refs={}
    for row in rows:
        name=row["id"];sample=row["sample"];assert groups[name]["split"]=="development"
        if name not in refs:
            protein=ESMProtein.from_pdb(ROOT/groups[name]["paths"][0]);assert protein.sequence==groups[name]["sequence"];refs[name]=protein.coordinates[:,:3].numpy()
        with np.load(cache/row["features"]) as z:h=torch.from_numpy(z["h"].copy());coarse=z["coarse"].copy();expected=z["fine"].copy()
        logits={"native":F.linear(h,projection["weight"][:4096],projection["bias"][:4096]).numpy()}
        if not np.array_equal(logits["native"].argmax(-1),expected):raise ValueError("Original fine feature replay failed")
        for parent,head in heads.items():
            logits[parent]=head(h).numpy()
            with np.load(fine_run/parent/"predictions"/f"{name}_sample{sample}.npz") as z:
                if not np.array_equal(z["coarse"],coarse) or not np.array_equal(logits[parent].argmax(-1),z["tokens"][:,0]):raise ValueError("Original-semantic LoRA replay failed")
        key=f"{name}_sample{sample}";path=source/(key+".npz")
        np.savez_compressed(path,logits=np.stack([logits[parent] for parent in ["native","plain","retained"]]),fine=np.stack([logits[parent].argmax(-1) for parent in ["native","plain","retained"]]),coarse=coarse,reference_backbone=refs[name])
        targets.append(dict(target=key,protein=name,sample=sample,input=path.name));hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest();print("NATIVE_FINE_COMBO_INPUT",key,flush=True)
    atomic_json(source/"manifest.json",dict(targets=targets))
    files=["upstream/experiments/fixed_pool_geometry_search.py","upstream/experiments/run_native_fine_combo.py","upstream/experiments/fine_retention.py","upstream/experiments/coarse_lora.py"]
    protocol=dict(version=1,scope="Original semantic/native coarse->fine projection-LoRA plus geometry search;40reused development proteins x2 fixed original draws. No RVQ, medium remapping, model retraining or58/64test access. Distinct from earlier coarse-LoRA/RVQ pipeline.",source=str(source),targets=targets,input_sha256=hashes,parents=["native","plain","retained"],snapshots=snapshots,feature_manifest_sha256=hashlib.sha256((cache/"manifest.json").read_bytes()).hexdigest(),projection_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),weights="artifacts/tokenizer/decoder_checkpoint_110.pth",chunks=4,decoder_budget=64,delta_angstrom=1.,edit_fraction_cap=.1,topk=5,budget="Fixed64 decoder candidate evaluations for every parent/context, plus separately decoded initial output. Samepool search, same displacement/edit gates, unchanged-parent fallback. No-op padding after violation exhaustion reported, not early-stop speed claim.",comparison="Native/plain/retained each before/after same64-call search. Their own fixed fine probabilities propose top5, their own initial output is coordinate anchor. Same original coarse and originalfine semantics for every parent.",focus="Reference RMSD, ungated reference angles, CA/CN, actual search time, two-sample spread as warning only. Do not treat mean CN outlier counts as percentages. Old plain LoRA RMSD CI crosseszero and two-sample spread drops~12percent, so no existing overall superiority claim.",hardware="Same GPUFP32 decoder per context, TF32 disabled; frozen checkpoints exactly replay oldCPU token predictions at preparation. No CPU/GPU exact-coordinate claim.",code_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files})
    atomic_json(a.run/"protocol.json",protocol);print("NATIVE_FINE_COMBO_PREPARED",80,flush=True)


if __name__=="__main__":main()
