"""Equal decoder-call proposal-source ablation, frozen GPU decoder and fixed pools."""
import argparse,hashlib,json,socket,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from fine_search_constraints import evaluate_candidate
from structural_metrics import summarize_backbone
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--chunk",type=int,default=0);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("Slurm GPU required")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device("cuda")
    protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"])
    for file,digest in protocol["code_sha256"].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    path=a.run/"candidates.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["candidate_sha256"]
    with np.load(path) as z:tables={k:z[k].copy() for k in z.files}
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.to(device).eval().requires_grad_(False)
    corpus=Path(protocol["corpus"]);refs={x["id"]:x for x in json.loads((corpus/"manifest.json").read_text())["targets"]}
    def decode(ids):return decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long().to(device))).cpu().numpy()
    targets=protocol["targets"][:1] if a.preflight_only else protocol["targets"][a.chunk::protocol["chunks"]]
    for item in targets:
        path=source/item["input"];assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["input_sha256"][item["target"]]
        with np.load(path) as z:original=z["fine"].copy();logits=torch.from_numpy(z["logits"].copy())
        assert np.array_equal(logits.argmax(-1).numpy(),original)
        start=time.perf_counter();head=logits.topk(5).indices.numpy();pools=dict(head_topk=head,fine_neighbors=tables["fine_neighbors"][original],repaired_cluster=tables["repaired_cluster"][original],original_cluster=tables["original_cluster"][original]);retrieval_seconds=time.perf_counter()-start
        for pool in pools.values():
            assert pool.shape==(len(original),5)
            if not all(len(set(row))==5 and int(token) in row for row,token in zip(pool,original)):raise ValueError("Invalid fixed proposal pool")
        pool_overlap={}
        for arm,pool in pools.items():
            def overlap(other):return float(np.mean([len((set(row)-{int(token)})&(set(other_row)-{int(token)}))/4 for row,other_row,token in zip(pool,other,original)]))
            pool_overlap[arm]=dict(alternative_overlap_with_head=overlap(head),alternative_overlap_with_fine_neighbors=overlap(pools["fine_neighbors"]))
        baseline=decode(original);baseline_metrics=summarize_backbone(baseline)
        if not np.isfinite(baseline).all():raise ValueError("Nonfinite initial decode")
        results={};arrays={};budget=item["decoder_call_budget"];index=protocol["targets"].index(item);arms=protocol["arms"];offset=index%len(arms);arms=arms[offset:]+arms[:offset]
        for arm in arms:
            current=original.copy();best_bb=baseline.copy();best_metrics=baseline_metrics;key=(best_metrics["backbone_cn_outliers"],best_metrics["upstream_ca_clashes"]);proposals=[];cursor=0;trace=[];padding=0;fallback_calls=0;torch.cuda.synchronize();started=time.perf_counter()
            for call in range(budget):
                if not proposals or cursor>=len(proposals):
                    cn=np.linalg.norm(best_bb[:-1,2]-best_bb[1:,0],axis=-1);positions=set(np.where((cn<1)|(cn>1.8))[0].tolist());positions.update(i+1 for i in list(positions));ca=best_bb[:,1];x,y=np.where(np.triu(np.linalg.norm(ca[:,None]-ca[None],axis=-1)<3,1));positions.update(x.tolist());positions.update(y.tolist())
                    proposals=[(site,int(token),j) for site in sorted(positions) for j,token in enumerate(pools[arm][site]) if token!=current[site]];cursor=0
                ids=current.copy();is_padding=not proposals;used_fallback=False
                if is_padding:padding+=1
                else:
                    site,token,column=proposals[cursor];cursor+=1;ids[site]=token
                    if arm in ["repaired_cluster","original_cluster"]:
                        used_fallback=bool(tables[arm+"_fallback"][original[site],column]);fallback_calls+=int(used_fallback)
                bb=decode(ids);metrics=summarize_backbone(bb);gate=evaluate_candidate(ids,bb,original,baseline,metrics,baseline_metrics,protocol["delta_angstrom"],protocol["edit_fraction_cap"])
                candidate_key=(metrics.get("backbone_cn_outliers",10**9),metrics.get("upstream_ca_clashes",10**9));accepted=not is_padding and gate["eligible"] and candidate_key<key
                if accepted:current=ids.copy();best_bb=bb.copy();best_metrics=metrics;key=candidate_key;proposals=[]
                trace.append(dict(call=call+1,accepted=accepted,padding=is_padding,cluster_fallback=used_fallback,gate=gate,cn_count=metrics.get("backbone_cn_outliers"),ca_clashes=metrics.get("upstream_ca_clashes")))
            torch.cuda.synchronize();elapsed=time.perf_counter()-started;assert len(trace)==budget
            results[arm]=dict(selected=best_metrics,gate=evaluate_candidate(current,best_bb,original,baseline,best_metrics,baseline_metrics,protocol["delta_angstrom"],protocol["edit_fraction_cap"]),returned_original=bool(np.array_equal(current,original)),decoder_calls=len(trace),padding_calls=padding,fallback_proposal_calls=fallback_calls,search_seconds=elapsed,trace=trace)
            arrays[arm]=(current.copy(),best_bb.copy())
        # Reference data enters only after proposal and acceptance for all four arms.
        with np.load(corpus/refs[item["protein"]]["backbone"]) as z:reference=z["backbone"].copy()
        directory=a.run/("preflight" if a.preflight_only else item["target"]);directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/"baseline.npz",fine=original,backbone=baseline)
        for arm,(ids,bb) in arrays.items():
            np.savez_compressed(directory/(arm+".npz"),fine=ids,backbone=bb)
            results[arm]["posthoc_metrics"]=summarize_backbone(bb,[reference[:,1]])
            try:results[arm]["posthoc_quality"]=measurements(bb,reference);results[arm]["quality_error"]=None
            except ValueError as error:results[arm]["posthoc_quality"]=None;results[arm]["quality_error"]=str(error)
        atomic_json(directory/"summary.json",dict(status="complete",target=item["target"],protein=item["protein"],sample=item["sample"],length=len(original),budget=budget,baseline=summarize_backbone(baseline,[reference[:,1]]),baseline_quality=measurements(baseline,reference),arms=results,candidate_retrieval_seconds=retrieval_seconds,candidate_pool_overlap=pool_overlap,host=socket.gethostname(),gpu=torch.cuda.get_device_name(),precision="All arms same GPU FP32,TF32 disabled; oldCPU calls used only as budget, no cross-hardware output or speed-equivalence claim",scope=protocol["scope"]))
        print("MEDIUM_CANDIDATE_SEARCH_COMPLETE",item["target"],budget,flush=True)
    if a.preflight_only:atomic_json(a.run/"preflight.json",dict(status="passed",target=targets[0]["target"],decoder_budget=targets[0]["decoder_call_budget"],arms=protocol["arms"]))
    else:atomic_json(a.run/("chunk"+str(a.chunk)+".json"),dict(status="complete",inputs=len(targets)))


if __name__=="__main__":main()
