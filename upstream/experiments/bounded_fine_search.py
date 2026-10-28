"""Pilot reference-free fine-token search on four reused TRAIN inputs."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder, decode_embedding
from geometry_objective import hard_token_embedding, backbone_penalties, mix_logit_gradients
from structural_metrics import summarize_backbone
from fine_search_constraints import evaluate_candidate
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--index",type=int,required=True)
    a=p.parse_args();torch.set_num_threads(16)
    protocol=json.loads((a.run/"protocol.json").read_text())
    source=Path(protocol["source"])
    manifest=json.loads((source/"manifest.json").read_text()); item=manifest["targets"][a.index]
    path=source/item["input"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["input_sha256"][item["target"]]
    with np.load(path) as data:
        initial=torch.from_numpy(data["logits"].copy()); original=data["fine"].copy()
        # Reference is withheld until all searches and acceptance decisions finish.
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False)
    decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state
    decoder.eval().requires_grad_(False)
    with torch.no_grad():
        base=decode_embedding(decoder,decoder.embed(torch.from_numpy(original).long())).numpy()
    baseline=summarize_backbone(base)
    teacher=initial.softmax(-1)
    directory=a.run/item["target"];directory.mkdir(exist_ok=True)
    results={}
    for arm in protocol["arms"]:
        logits=torch.nn.Parameter(initial.clone());opt=torch.optim.Adam([logits],lr=protocol["lr"])
        rng=torch.Generator().manual_seed(2026+a.index)
        best_ids=original.copy();best_bb=base.copy();best_metrics=baseline;best_key=(baseline["backbone_cn_outliers"],baseline["upstream_ca_clashes"])
        trace=[];calls=0;backwards=0;current=torch.from_numpy(original).long();search_positions=[];cursor=0
        started=time.perf_counter()
        while time.perf_counter()-started < protocol["seconds_per_arm"] and calls < protocol["max_forward_calls"]:
            if arm=="discrete_topk":
                if not search_positions or cursor>=len(search_positions):
                    # Prioritize sites incident to initial/current decoded violations. No reference access.
                    bb=best_bb; cn=np.linalg.norm(bb[:-1,2]-bb[1:,0],axis=-1)
                    positions=set(np.where((cn<1.)|(cn>1.8))[0].tolist())
                    positions.update(i+1 for i in list(positions))
                    ca=bb[:,1];dist=np.linalg.norm(ca[:,None]-ca[None],axis=-1)
                    x,y=np.where(np.triu(dist<3.,1));positions.update(x.tolist());positions.update(y.tolist())
                    if not positions:break
                    search_positions=[]
                    for site in sorted(positions):
                        for token in initial[site].topk(protocol["topk"]).indices.tolist():
                            if token != int(current[site]):search_positions.append((site,token))
                    cursor=0
                    if not search_positions:break
                site,token=search_positions[cursor];cursor+=1
                ids=current.clone();ids[site]=token
                with torch.no_grad():bb_tensor=decode_embedding(decoder,decoder.embed(ids))
            else:
                opt.zero_grad(set_to_none=True)
                emb,ids,probability=hard_token_embedding(logits,decoder.embed.weight[:4096])
                bb_tensor=decode_embedding(decoder,emb)
            calls+=1;bb=bb_tensor.detach().numpy();metrics=summarize_backbone(bb)
            gate=evaluate_candidate(ids.detach().numpy(),bb,original,base,metrics,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"])
            key=(metrics.get("backbone_cn_outliers",10**9),metrics.get("upstream_ca_clashes",10**9))
            accepted=gate["eligible"] and key<best_key
            if accepted:
                best_key=key;best_ids=ids.detach().numpy().copy();best_bb=bb.copy();best_metrics=metrics
                if arm=="discrete_topk":current=ids.clone();search_positions=[]
            trace.append(dict(call=calls,elapsed_seconds=time.perf_counter()-started,accepted=accepted,gate=gate,metrics=metrics))
            if arm!="discrete_topk":
                cn,clash=backbone_penalties(bb_tensor)
                grad=torch.autograd.grad(cn+clash,logits)[0];backwards+=1
                if not torch.isfinite(grad).all():raise RuntimeError("Nonfinite logit gradient")
                order=torch.randperm(initial.shape[-1],generator=rng) if arm=="shuffled_gradient" else None
                logits.grad=mix_logit_gradients(grad,probability.detach(),teacher,permutation=order)
                torch.nn.utils.clip_grad_norm_([logits],1.);opt.step()
            if calls%10==0:print(item["target"],arm,calls,best_key,flush=True)
        elapsed=time.perf_counter()-started
        np.savez_compressed(directory/(arm+".npz"),fine=best_ids,backbone=best_bb)
        results[arm]=dict(baseline=baseline,selected=best_metrics,gate=evaluate_candidate(best_ids,best_bb,original,base,best_metrics,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"]),
                          returned_original=bool(np.array_equal(best_ids,original)),forward_calls=calls,backward_calls=backwards,
                          search_seconds=elapsed,budget_overrun_seconds=max(0.,elapsed-protocol["seconds_per_arm"]),
                          selection_opportunities=len(trace),trace=trace)
        atomic_json(directory/(arm+".json"),results[arm])
    # Independent reference evaluation happens only after every arm is finalized.
    with np.load(path) as data:reference=data["reference_ca"].copy()
    for arm,result in results.items():
        with np.load(directory/(arm+".npz")) as data:bb=data["backbone"].copy()
        result["posthoc_reference_metrics"]=summarize_backbone(bb,[reference])
        result["posthoc_baseline_reference_metrics"]=summarize_backbone(base,[reference])
    atomic_json(directory/"summary.json",dict(target=item["target"],protocol=protocol,arms=results,
        scope="Four reused selected TRAIN probes; feasibility/timing only. Equal nominal wall budget, operation overruns and selection opportunities explicitly reported; not a matched-compute efficacy claim."))
    print("BOUNDED_SEARCH_COMPLETE",item["target"],flush=True)


if __name__=="__main__":main()
