"""Actual nearest-fine snap and frozen decoder, after tiny-head training."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import summarize_backbone,aligned_rmsd
from backbone_quality_audit import measurements
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--mode",choices=["fm","regression"],required=True);a=p.parse_args();torch.set_num_threads(16);protocol=json.loads((a.run/"protocol.json").read_text());directory=a.run/a.mode;training=json.loads((directory/"summary.json").read_text());assert training["status"]=="complete"
    with np.load(protocol["calibration"]) as z:codes=torch.from_numpy(z["fine_codes"].copy())
    norm=codes.square().sum(-1)
    def snap(latent):
        x=torch.from_numpy(latent.copy())
        return (x.square().sum(-1,keepdim=True)-2*x@codes.T+norm[None,:]).argmin(-1)
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol["decoder"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.eval().requires_grad_(False)
    corpus=Path(protocol["corpus"]);refs={x["id"]:x for x in json.loads((corpus/"manifest.json").read_text())["targets"]};output=directory/"decoded";output.mkdir(exist_ok=True)
    for name in protocol["geometry_targets"]:
        with np.load(directory/"predictions"/(name+".npz")) as z:predicted=z["latent"].copy();native=torch.from_numpy(z["native_fine"].copy()).long()
        with np.load(corpus/refs[name]["backbone"]) as z:reference=z["backbone"].copy()
        with np.load(Path(protocol["tokens"])/(name+".npz")) as z:true_latent=z["latent"].copy();truth=torch.from_numpy(z["fine"].copy()).long()
        if not torch.equal(snap(true_latent),truth):raise ValueError("Oracle fine snap failed to replay")
        base=decode_embedding(decoder,decoder.embed(native)).numpy();oracle=decode_embedding(decoder,decoder.embed(truth)).numpy();samples=[];arrays=[];tokens=[]
        for latent in predicted:
            fine=snap(latent);bb=decode_embedding(decoder,decoder.embed(fine)).numpy()
            if not np.isfinite(bb).all():raise RuntimeError("Nonfinite decoded latent sample")
            try:quality=measurements(bb,reference);quality_error=None
            except ValueError as error:quality=None;quality_error=str(error)
            samples.append(dict(metrics=summarize_backbone(bb,[reference[:,1]]),quality=quality,quality_error=quality_error,latent_mse_to_oracle=float(np.mean((latent-true_latent)**2)),fine_match_to_oracle=float(np.mean(fine.numpy()==truth.numpy())),ca_displacement_from_original=aligned_rmsd(bb[:,1],base[:,1])))
            arrays.append(bb);tokens.append(fine.numpy())
        np.savez_compressed(output/(name+".npz"),backbone=np.stack(arrays),fine=np.stack(tokens),native_backbone=base,oracle_backbone=oracle)
        atomic_json(output/(name+".json"),dict(target=name,length=len(native),mode=a.mode,baseline=summarize_backbone(base,[reference[:,1]]),oracle=summarize_backbone(oracle,[reference[:,1]]),samples=samples,same_coarse_conditional_pairwise_ca_rmsd=aligned_rmsd(arrays[0][:,1],arrays[1][:,1]) if len(arrays)>1 else None,scope=protocol["scope"]))
        print("LATENT_HEAD_DECODED",a.mode,name,flush=True)


if __name__=="__main__":main()
