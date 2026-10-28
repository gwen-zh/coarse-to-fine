"""Calibrate noise on TRAIN only and freeze an oracle latent robustness protocol."""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
import torch
from diagnostic_io import atomic_json
from latent_sensitivity_math import covariance_root


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    tokens=Path("artifacts/pdb_training_v2/tokens");manifest=json.loads((tokens/"manifest.json").read_text());train=[x for x in manifest["targets"] if x["subset"]=="train"]
    assert len(train)==899
    count=0;total=np.zeros(128);cross=np.zeros((128,128));hashes={}
    for i,row in enumerate(train):
        path=tokens/row["tokens"];raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        with np.load(io.BytesIO(raw)) as z:x=z["latent"].astype(np.float64)
        if x.shape!=(row["length"],128) or not np.isfinite(x).all():raise ValueError("Invalid TRAIN latent")
        count+=len(x);total+=x.sum(0);cross+=x.T@x
        if i%100==0:print("LATENT_CALIBRATION",i,len(train),flush=True)
    mean=total/count;covariance=cross/count-np.outer(mean,mean);root=covariance_root(covariance);scale=float(np.sqrt(np.trace(covariance)/128))
    bridge=Path("artifacts/development/residual_tokens/residual_bridge.pt");bundle=torch.load(bridge,map_location="cpu",weights_only=True);codes=bundle["fine_codes"].numpy();assert codes.shape==(4096,128)
    np.savez_compressed(a.run/"calibration.npz",mean=mean,covariance=covariance,covariance_root=root,centered_rms=np.asarray(scale),fine_codes=codes)
    original=Path("artifacts/development/objective_geometry_diagnosis/protocol.json");names=json.loads(original.read_text())["geometry_targets"];rows={x["id"]:x for x in manifest["targets"]};targets=[]
    corpus=Path("artifacts/pdb_training_v2");refs={x["id"]:x for x in json.loads((corpus/"manifest.json").read_text())["targets"]}
    for name in names:
        row=rows[name];assert row["subset"]=="validation";path=tokens/row["tokens"];ref=corpus/refs[name]["backbone"]
        targets.append(dict(target=name,length=row["length"],latent=str(path),reference=str(ref),latent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
    protocol=dict(version=1,scope="Oracle encoder-latent local robustness on24 reused validation proteins; no generation, fitting or FM efficacy claim. No58/64 confirmation cohort access.",targets=targets,alphas=[.001,.003,.01,.03,.1,.3],noise_families=["isotropic","covariance_oriented"],replicates=4,noise_scale="alpha times TRAIN pooled centered per-feature RMS; each direction normalized to unit global RMS; same base Gaussian draw across families and same direction across amplitudes",training_proteins=len(train),training_residues=count,training_centered_rms=scale,training_input_sha256=hashes,source_manifest_sha256=hashlib.sha256((tokens/"manifest.json").read_bytes()).hexdigest(),fine_code_source=str(bridge),fine_code_source_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),calibration_sha256=hashlib.sha256((a.run/"calibration.npz").read_bytes()).hexdigest(),weights="artifacts/tokenizer/decoder_checkpoint_110.pth",boundary_test="One residue at10/50/90percentile nearest-cell-face margin; move0.99*margin and1.01*margin plus1e-5*trainingRMS in face-normal direction. Quantization uses actual FP32 squared-Euclidean implementation; record crossing numerical resolution.",failure_policy="All24 required, no outcome exclusion. Identical token strings reuse exact decoded coordinates; record requested/unique decoder counts. Per-candidate quality exceptions recorded explicitly, not silently discarded.")
    for f in ["latent_sensitivity_math.py","run_latent_sensitivity.py"]:
        path=Path("upstream/experiments")/f;protocol.setdefault("code_sha256",{})[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(a.run/"protocol.json",protocol);print("LATENT_SENSITIVITY_PREPARED",len(targets),scale,flush=True)


if __name__=="__main__":main()
