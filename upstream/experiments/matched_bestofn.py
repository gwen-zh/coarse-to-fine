"""Fixed decoder-call comparison of global BoN, local BoN and greedy search."""
import argparse,hashlib,json,time,socket
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from fine_search_constraints import evaluate_candidate
from backbone_quality_audit import measurements
from bestofn_proposals import local_proposals,apply_proposal
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--index",type=int,required=True);a=p.parse_args();torch.set_num_threads(16)
    protocol=json.loads((a.run/"protocol.json").read_text());item=protocol["targets"][a.index];source=Path(protocol["source"]);path=source/item["input"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["input_sha256"][item["target"]]
    with np.load(path) as z:logits=torch.from_numpy(z["logits"].copy());original=z["fine"].copy()
    if not np.array_equal(logits.argmax(-1).numpy(),original):raise ValueError("Original argmax replay mismatch")
    budget=item["decoder_call_budget"];load_start=time.perf_counter();decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30)
    state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.eval().requires_grad_(False);load_seconds=time.perf_counter()-load_start
    def decode(ids):
        with torch.no_grad():return decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long())).numpy()
    t=time.perf_counter();base=decode(original);baseline_decode_seconds=time.perf_counter()-t;baseline=summarize_backbone(base)
    directory=a.run/item["target"];directory.mkdir(exist_ok=True);np.savez_compressed(directory/"baseline.npz",fine=original,backbone=base)
    stream=local_proposals(logits.topk(5).indices.numpy(),original,budget,20261022+a.index)
    arms=list(protocol["arms"]);arms=arms[a.index%len(arms):]+arms[:a.index%len(arms)]
    results={}
    for arm in arms:
        started=time.perf_counter();current=original.copy();best_bb=base.copy();best_metrics=baseline;best_key=(baseline["backbone_cn_outliers"],baseline["upstream_ca_clashes"]);trace=[];reasons=dict(edit_cap_exceeded=0,displacement_exceeded=0,geometry_pareto_failed=0,eligible=0)
        generator=torch.Generator().manual_seed(18202+a.index)
        if arm.startswith("global_temperature_"):
            temperature=float(arm.rsplit("_",1)[-1]);probability=(logits/temperature).softmax(-1)
        for call in range(budget):
            if arm.startswith("global_temperature_"):
                ids=torch.multinomial(probability,1,replacement=True,generator=generator)[:,0].numpy()
            else:ids=apply_proposal(current if arm=="local_greedy" else original,stream[call])
            bb=decode(ids);metrics=summarize_backbone(bb);gate=evaluate_candidate(ids,bb,original,base,metrics,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"])
            reasons["edit_cap_exceeded"]+=int(gate["changed_fraction"]>protocol["edit_fraction_cap"])
            reasons["displacement_exceeded"]+=int(gate["aligned_ca_displacement"] is None or gate["aligned_ca_displacement"]>protocol["delta_angstrom"])
            cn,ca=metrics["backbone_cn_outliers"],metrics["upstream_ca_clashes"]
            pareto=cn<=baseline["backbone_cn_outliers"] and ca<=baseline["upstream_ca_clashes"] and (cn<baseline["backbone_cn_outliers"] or ca<baseline["upstream_ca_clashes"])
            reasons["geometry_pareto_failed"]+=int(not pareto);reasons["eligible"]+=int(gate["eligible"])
            key=(cn,ca);accepted=gate["eligible"] and key<best_key
            if accepted:current=ids.copy();best_bb=bb.copy();best_metrics=metrics;best_key=key
            trace.append(dict(call=call+1,accepted=accepted,gate=gate,cn_count=cn,ca_clashes=ca))
        elapsed=time.perf_counter()-started
        assert len(trace)==budget
        np.savez_compressed(directory/(arm+".npz"),fine=current,backbone=best_bb)
        results[arm]=dict(selected=best_metrics,returned_original=bool(np.array_equal(current,original)),gate=evaluate_candidate(current,best_bb,original,base,best_metrics,baseline,protocol["delta_angstrom"],protocol["edit_fraction_cap"]),decoder_calls=len(trace),search_seconds=elapsed,rejection_diagnostics_overlapping=reasons,trace=trace)
        print("MATCHED_BON_ARM_COMPLETE",item["target"],arm,budget,elapsed,flush=True)
    # No reference data participates in proposal generation, scoring or selection.
    corpus=Path(protocol["corpus"]);refrow=next(x for x in json.loads((corpus/"manifest.json").read_text())["targets"] if x["id"]==item["protein"])
    with np.load(corpus/refrow["backbone"]) as z:ref=z["backbone"].copy()
    for arm,result in results.items():
        with np.load(directory/(arm+".npz")) as z:bb=z["backbone"].copy()
        result["posthoc_metrics"]=summarize_backbone(bb,[ref[:,1]]);result["posthoc_quality"]=measurements(bb,ref)
    atomic_json(directory/"summary.json",dict(target=item["target"],protein=item["protein"],sample=item["sample"],decoder_call_budget=budget,baseline=summarize_backbone(base,[ref[:,1]]),baseline_quality=measurements(base,ref),decoder_load_seconds=load_seconds,baseline_decode_seconds=baseline_decode_seconds,host=socket.gethostname(),arms=results,scope=protocol["scope"]))


if __name__=="__main__":main()
