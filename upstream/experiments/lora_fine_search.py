"""Discrete geometry search using RVQ bridge distances, no invented fine logits."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from fine_search_constraints import evaluate_candidate
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--index",type=int,required=True);a=p.parse_args();torch.set_num_threads(16)
    protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"]);item=json.loads((source/"manifest.json").read_text())["targets"][a.index];path=source/item["input"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["input_sha256"][item["target"]]
    with np.load(path) as z:original=z["fine"].copy();candidates=z["candidates"].copy();expected=z["expected_backbone"].copy()
    if not np.array_equal(candidates[:,0],original):raise ValueError("Candidate origin mismatch")
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state
    decoder.eval().requires_grad_(False)
    with torch.no_grad():base=decode_embedding(decoder,decoder.embed(torch.from_numpy(original).long())).numpy()
    # Cached original runs used8 CPU threads. Exact bits can differ: verify replay with explicit tolerance.
    replay_max=float(np.max(np.abs(base-expected)))
    if not np.allclose(base,expected,rtol=1e-4,atol=1e-3):raise ValueError(f"Original decoded output replay drift {replay_max}")
    baseline=summarize_backbone(base);directory=a.run/item["target"];directory.mkdir(exist_ok=True);np.savez_compressed(directory/"baseline.npz",fine=original,backbone=base)
    results={}
    for arm in (protocol["arms"] if a.index%2==0 else list(reversed(protocol["arms"]))):
        current=original.copy();best_bb=base.copy();best_metrics=baseline;best_key=(baseline["backbone_cn_outliers"],baseline["upstream_ca_clashes"])
        rng=np.random.default_rng(2026+a.index);proposals=[];cursor=0;trace=[];started=time.perf_counter()
        while time.perf_counter()-started<protocol["seconds_per_arm"]:
            if not proposals or cursor>=len(proposals):
                cn=np.linalg.norm(best_bb[:-1,2]-best_bb[1:,0],axis=-1);positions=set(np.where((cn<1)|(cn>1.8))[0].tolist());positions.update(i+1 for i in list(positions))
                ca=best_bb[:,1];x,y=np.where(np.triu(np.linalg.norm(ca[:,None]-ca[None],axis=-1)<3,1));positions.update(x.tolist());positions.update(y.tolist())
                if arm=="bridge_random_sites":positions=set(range(len(original)))
                proposals=[(i,int(t)) for i in sorted(positions) for t in candidates[i] if t!=current[i]]
                if arm=="bridge_random_sites":rng.shuffle(proposals)
                cursor=0
                if not proposals:break
            i,t=proposals[cursor];cursor+=1;ids=current.copy();ids[i]=t
            with torch.no_grad():bb=decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long())).numpy()
            m=summarize_backbone(bb);gate=evaluate_candidate(ids,bb,original,base,m,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"]);key=(m.get("backbone_cn_outliers",10**9),m.get("upstream_ca_clashes",10**9));accepted=gate["eligible"] and key<best_key
            if accepted:current=ids;best_bb=bb.copy();best_metrics=m;best_key=key;proposals=[]
            trace.append(dict(call=len(trace)+1,elapsed_seconds=time.perf_counter()-started,accepted=accepted,gate=gate,metrics=m))
        elapsed=time.perf_counter()-started;np.savez_compressed(directory/(arm+".npz"),fine=current,backbone=best_bb)
        results[arm]=dict(selected=best_metrics,gate=evaluate_candidate(current,best_bb,original,base,best_metrics,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"]),returned_original=bool(np.array_equal(current,original)),search_seconds=elapsed,forward_calls=len(trace),trace=trace)
    # Reference is first read only after both search decisions have finished.
    with np.load(path) as z:reference=z["reference_backbone"].copy()
    reference_ca=reference[:,1]
    for arm,d in results.items():
        with np.load(directory/(arm+".npz")) as z:bb=z["backbone"].copy()
        d["posthoc_metrics"]=summarize_backbone(bb,[reference_ca]);d["posthoc_ungated_quality"]=measurements(bb,reference)
    atomic_json(directory/"summary.json",dict(target=item["target"],protein=item["protein"],parent=item["parent"],sample=item["sample"],baseline=summarize_backbone(base,[reference_ca]),baseline_ungated_quality=measurements(base,reference),replay_max_coordinate_error=replay_max,arms=results,scope=protocol["scope"]))
    print("LORA_BRIDGE_SEARCH_COMPLETE",item["target"],flush=True)


if __name__=="__main__":main()
