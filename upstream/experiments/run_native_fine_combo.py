"""Six paired outputs on the original pathway, without any new training."""
import argparse,hashlib,json,socket,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone
from backbone_quality_audit import measurements
from fixed_pool_geometry_search import search_fixed_pool
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--chunk",type=int,default=0);p.add_argument("--preflight-only",action="store_true");a=p.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("Slurm GPU required")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device("cuda");protocol=json.loads((a.run/"protocol.json").read_text());source=Path(protocol["source"])
    for file,digest in protocol["code_sha256"].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.to(device).eval().requires_grad_(False)
    def decode(ids):return decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long().to(device))).cpu().numpy()
    targets=protocol["targets"][:1] if a.preflight_only else protocol["targets"][a.chunk::protocol["chunks"]]
    for item in targets:
        path=source/item["input"];assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol["input_sha256"][item["target"]]
        with np.load(path) as z:logits=z["logits"].copy();initial=z["fine"].copy()
        assert np.array_equal(logits.argmax(-1),initial)
        parents=protocol["parents"];offset=protocol["targets"].index(item)%3;order=parents[offset:]+parents[:offset];results={};structures={}
        for parent in order:
            index=parents.index(parent);ids=initial[index];pool=torch.from_numpy(logits[index]).topk(5).indices.numpy();baseline=decode(ids);torch.cuda.synchronize();started=time.perf_counter()
            fine,bb,result=search_fixed_pool(ids,pool,baseline,protocol["decoder_budget"],decode,delta=protocol["delta_angstrom"],edit_cap=protocol["edit_fraction_cap"]);torch.cuda.synchronize();result["search_seconds"]=time.perf_counter()-started
            assert result["decoder_calls"]==protocol["decoder_budget"]
            structures[parent]=dict(original_fine=ids.copy(),original_backbone=baseline,searched_fine=fine,searched_backbone=bb);results[parent]=result
        # Reference first enters after all parent-specific searches are finalized.
        with np.load(path) as z:reference=z["reference_backbone"].copy()
        directory=a.run/("preflight" if a.preflight_only else item["target"]);directory.mkdir(exist_ok=True)
        for parent,data in structures.items():
            np.savez_compressed(directory/(parent+".npz"),**data)
            results[parent]["original_metrics"]=summarize_backbone(data["original_backbone"],[reference[:,1]]);results[parent]["searched_metrics"]=summarize_backbone(data["searched_backbone"],[reference[:,1]])
            for phase in ["original","searched"]:
                try:results[parent][phase+"_quality"]=measurements(data[phase+"_backbone"],reference);results[parent][phase+"_quality_error"]=None
                except ValueError as error:results[parent][phase+"_quality"]=None;results[parent][phase+"_quality_error"]=str(error)
        atomic_json(directory/"summary.json",dict(status="complete",target=item["target"],protein=item["protein"],sample=item["sample"],length=len(initial[0]),parents=results,host=socket.gethostname(),gpu=torch.cuda.get_device_name(),scope=protocol["scope"]))
        print("NATIVE_FINE_COMBO_COMPLETE",item["target"],flush=True)
    if a.preflight_only:atomic_json(a.run/"preflight.json",dict(status="passed",target=targets[0]["target"],parents=protocol["parents"],decoder_budget=protocol["decoder_budget"]))
    else:atomic_json(a.run/("chunk"+str(a.chunk)+".json"),dict(status="complete",inputs=len(targets)))


if __name__=="__main__":main()
