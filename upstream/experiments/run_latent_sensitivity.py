"""Frozen tokenizer robustness: calibrated Gaussian noise and cell-face probes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from evaluate_residual_vq import ProjectedDecoder,decode_embedding
from structural_metrics import aligned_rmsd,summarize_backbone
from backbone_quality_audit import measurements
from latent_sensitivity_math import unit_rms_noise,nearest_cell_faces,aligned_ca_distances
from diagnostic_io import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--index",type=int,required=True);a=p.parse_args();torch.set_num_threads(16)
    pp=a.run/"protocol.json";protocol=json.loads(pp.read_text());item=protocol["targets"][a.index]
    for name in ["latent","reference"]:
        path=Path(item[name]);assert hashlib.sha256(path.read_bytes()).hexdigest()==item[name+"_sha256"]
    calibration=a.run/"calibration.npz";assert hashlib.sha256(calibration.read_bytes()).hexdigest()==protocol["calibration_sha256"]
    with np.load(calibration) as z:codes=z["fine_codes"].copy();root=z["covariance_root"].copy();scale=float(z["centered_rms"])
    with np.load(item["latent"]) as z:latent=z["latent"].copy();initial=z["fine"].copy()
    with np.load(item["reference"]) as z:reference=z["backbone"].copy()
    codes_t=torch.from_numpy(codes);norm=codes_t.square().sum(-1)
    def snap(z):
        x=torch.from_numpy(np.asarray(z,dtype=np.float32))
        return (x.square().sum(-1,keepdim=True)-2*x@codes_t.T+norm[None,:]).argmin(-1).numpy()
    replay=snap(latent)
    if not np.array_equal(replay,initial):raise ValueError("Original nearest-neighbor fine IDs do not replay cached tokenizer")
    margin,neighbors,directions=nearest_cell_faces(latent,codes,initial)
    quant_rms=float(np.sqrt(np.mean((latent-codes[initial])**2)))
    decoder=ProjectedDecoder(d_model=1280,n_heads=20,n_layers=30);state=torch.load(protocol["weights"],map_location="cpu",mmap=True,weights_only=False);decoder.load_state_dict(state.get("decode_proj",state),strict=True);del state;decoder.eval().requires_grad_(False)
    cache={};decoder_seconds=0.
    def decode(ids):
        nonlocal decoder_seconds
        key=ids.tobytes()
        if key in cache:return cache[key],True
        started=time.perf_counter();bb=decode_embedding(decoder,decoder.embed(torch.from_numpy(ids).long())).numpy();decoder_seconds+=time.perf_counter()-started;cache[key]=bb
        return bb,False
    base,_=decode(initial)
    if not np.isfinite(base).all():raise ValueError("Nonfinite initial reconstruction")
    rows=[];all_ids=[];all_bb=[]
    def record(value,metadata):
        value=np.asarray(value,dtype=np.float32);ids=snap(value);bb,reused=decode(ids);changed=np.where(ids!=initial)[0];noise=(value-latent).astype(np.float64);rms=float(np.sqrt(np.mean(noise*noise)))
        metrics=summarize_backbone(bb,[reference[:,1]])
        if np.isfinite(bb[:,1]).all():
            distances=aligned_ca_distances(bb[:,1],base[:,1]);displacement=float(np.sqrt(np.mean(distances**2)));maximum=float(distances.max());p95=float(np.quantile(distances,.95))
            if len(changed):
                far=np.min(np.abs(np.arange(len(ids))[:,None]-changed[None,:]),axis=-1)>8
                far_mean=float(distances[far].mean()) if far.any() else None
            else:far_mean=None
        else:displacement=maximum=p95=far_mean=None
        try:quality=measurements(bb,reference);quality_error=None
        except ValueError as error:quality=None;quality_error=str(error)
        row=dict(metadata,noise_rms=rms,noise_relative_training_rms=rms/scale,noise_relative_quantization_rms=rms/quant_rms if quant_rms>0 else None,changed_tokens=len(changed),changed_fraction=len(changed)/len(ids),ca_displacement=displacement,max_ca_displacement=maximum,p95_ca_displacement=p95,mean_far_ca_displacement=far_mean,decoder_reused=reused,metrics=metrics,ungated_quality=quality,ungated_quality_error=quality_error)
        rows.append(row);all_ids.append(ids.copy());all_bb.append(bb.copy());return row
    record(latent,dict(kind="baseline",family="baseline",replicate=0,alpha=0.))
    for replicate in range(protocol["replicates"]):
        seed=int(hashlib.sha256(("latent-sensitivity-v1:"+item["target"]+":"+str(replicate)).encode()).hexdigest()[:8],16);raw=np.random.default_rng(seed).normal(size=latent.shape)
        for family in protocol["noise_families"]:
            direction=unit_rms_noise(raw,None if family=="isotropic" else root)
            for alpha in protocol["alphas"]:
                record(latent+alpha*scale*direction,dict(kind="gaussian_noise",family=family,replicate=replicate,alpha=alpha,seed=seed))
        print("LATENT_NOISE_REPLICATE_COMPLETE",item["target"],replicate,flush=True)
    for quantile in [.1,.5,.9]:
        residue=int(np.argmin(np.abs(margin-np.quantile(margin,quantile))))
        for side in ["below","above"]:
            radius=.99*margin[residue] if side=="below" else 1.01*margin[residue]+1e-5*scale
            value=latent.astype(np.float64).copy();value[residue]+=radius*directions[residue]
            row=record(value,dict(kind="cell_face",family="nearest_cell_face",replicate=0,alpha=None,residue=residue,margin_quantile=quantile,exact_face_margin=float(margin[residue]),neighbor=int(neighbors[residue]),side=side,step_length=float(radius)))
            row["expected_crossing_observed"]=bool(row["changed_tokens"]>0) if side=="above" else bool(row["changed_tokens"]==0)
    directory=a.run/item["target"];directory.mkdir(exist_ok=True)
    np.savez_compressed(directory/"structures.npz",fine=np.stack(all_ids),backbone=np.stack(all_bb),latent=latent,cell_face_margin=margin,cell_face_neighbor=neighbors)
    assert len(rows)==1+len(protocol["noise_families"])*protocol["replicates"]*len(protocol["alphas"])+6
    atomic_json(directory/"summary.json",dict(status="complete",target=item["target"],length=len(latent),protocol_sha256=hashlib.sha256(pp.read_bytes()).hexdigest(),training_centered_rms=scale,quantization_rms=quant_rms,cell_margin_quantiles=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),requested_conditions=len(rows),unique_decoder_calls=len(cache),decoder_seconds=decoder_seconds,records=rows,scope=protocol["scope"]))
    print("LATENT_SENSITIVITY_COMPLETE",item["target"],len(rows),len(cache),flush=True)


if __name__=="__main__":main()
