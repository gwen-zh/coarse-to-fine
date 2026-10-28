"""Decode every sampler arm with the same actual frozen fine-token decoder."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import aligned_rmsd,summarize_backbone
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--chunk",type=int,required=True);a=p.parse_args();torch.set_num_threads(16);protocol=json.loads((a.run/"protocol.json").read_text())
    f=protocol["decoder"];st=Path(f["path"]).stat();assert st.st_size==f["size"] and st.st_mtime_ns==f["mtime_ns"]
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(f["path"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.eval().requires_grad_(False)
    output=a.run/"decoded";output.mkdir(exist_ok=True)
    for item in protocol["targets"][a.chunk::protocol["decode_chunks"]]:
        name=item["id"];n=item["length"]
        with np.load(Path(protocol["corpus"])/item["backbone"]) as z:reference=z["backbone"].copy()
        cache={};methods={}
        for steps in protocol["coarse_steps"]:
            methods[str(steps)]={}
            for arm in protocol["arms"]:
                stem=f'{name}_steps{steps}_{arm["name"]}'
                meta=json.loads((a.run/"predictions"/(stem+".json")).read_text());assert len(meta["samples"])==protocol["samples"]
                with np.load(a.run/"predictions"/(stem+".npz")) as z:tokens=z["fine"].copy()
                assert tokens.shape==(protocol["samples"],n)
                rows=[];arrays=[]
                for sample,fine in enumerate(tokens):
                    expected=meta["samples"][sample];assert expected["sample"]==sample and expected["network_calls"]==dict(coarse=steps+1,fine=1)
                    key=fine.tobytes()
                    if key not in cache:cache[key]=decode_embedding(decoder,decoder.embed(torch.from_numpy(fine).long())).numpy()
                    bb=cache[key];arrays.append(bb);metrics=summarize_backbone(bb,[reference[:,1]])
                    try:quality=measurements(bb,reference);quality_error=None
                    except ValueError as error:quality=None;quality_error=str(error)
                    rows.append(dict(sample=sample,metrics=metrics,quality=quality,quality_error=quality_error,sampling_seconds=expected["seconds"],remask_events=expected["trace"].get("remask_events",0),revised_token_events=expected["trace"].get("revised_token_events",0)))
                distances=[aligned_rmsd(x[:,1],y[:,1]) for i,x in enumerate(arrays) for y in arrays[i+1:]]
                finite_spread=all(x is not None for x in distances)
                errors=[x["metrics"].get("closest_available_reference_ca_rmsd") for x in rows]
                methods[str(steps)][arm["name"]]=dict(samples=rows,mean_pairwise_ca_rmsd=float(np.mean(distances)) if finite_spread else None,pairwise_ca_rmsd_quantiles=np.quantile(distances,[.1,.5,.9]).tolist() if finite_spread else None,best_reference_ca_rmsd=min(errors) if all(x is not None for x in errors) else None)
                np.savez_compressed(output/(stem+".npz"),backbone=np.stack(arrays),fine=tokens)
        atomic_json(output/(name+".json"),dict(target=name,length=n,methods=methods,unique_decoder_calls=len(cache)))
        print("DISCRETE_FLOW_DECODE_COMPLETE",name,len(cache),flush=True)


if __name__=="__main__":main()
