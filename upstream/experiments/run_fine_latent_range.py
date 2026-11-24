"""Fixed, reference-free range enforcement on cached fine-flow outputs.

The encoder's tanh defines[-1,1]; clipping is an input-domain check, not a
learned repair or proof that bounded vectors lie on the structure manifold.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_author_codec_interface import METRIC_KEYS, GEOMETRY_KEYS, evaluate_values, load, state_digest
from run_decoder_angle_adaptation import decode_latents
from run_sequence_multiscale_ensemble import load_codec


ROOT=Path("artifacts/fine_latent_range_v1")
SOURCE=Path("artifacts/decoder_angle_adaptation_v1")
LABELS=Path("artifacts/multiframe_early_fine_v1/development_labels.pt")


def verify(p):
    for path,digest in p["source_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("Frozen range-pilot input changed: "+path)


@torch.no_grad()
def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():raise RuntimeError("Fresh range-pilot namespace required")
    old=load(SOURCE/"protocol.json")
    p={k:old[k] for k in ["tokenizer","fine_checkpoint","targets","seeds"]}
    p.update(policy="Apply coordinate-wise clamp(-1,1) to final continuous fine8 only. Use the original frozen decoder, not either adapted decoder. Do not alter coarse, sampler trajectory, sample count or seeds.",
        rationale="Post-hoc audit found4.73% fine components outside the frozen encoder's tanh range, affecting28.81% residues. This range discrepancy motivates a fixed diagnostic; it does not establish causation or structural-manifold membership.",
        evaluation="Use all fixed64 samples per reused development sequence and seed. Main outlet: own frozen generator latents. Negative control: author-generated structures encoded by our bounded fine encoder; clipping must be exactly identity there. Persist every decoded output before native-coordinate evaluation. Compare raw and clipped on accuracy, coverage, C-N, both reference-relative angles, JS and spread. No acceptance rule or tuning.",
        scope="Exploratory inference-only control on two repeatedly reused development proteins and two seeds. No training, new sampling, quantization, independent confirmation, reserved test, R3 or model promotion.",
        failure_policy="Keep all raw outputs and NaN placeholders for frame failures. Any failed sample makes full group metrics unavailable. Never filter or select alternatives.",
        optimizer_updates=0,samples=64,reference_frames=128)
    groups=torch.load(SOURCE/"evaluation_inputs.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");checks=[]
    for group in groups:
        with torch.no_grad():
            value=codec.decode(group["coarse"][0],group["fine"][0])+group["origin"][0]
        torch.testing.assert_close(value,group["baseline"][0],atol=2e-5,rtol=2e-5)
        bounded=group["fine"].clamp(-1,1)
        if group["outlet"]=="author_roundtrip":torch.testing.assert_close(bounded,group["fine"],atol=0,rtol=0)
        checks.append(dict(seed=group["seed"],protein=group["protein"],outlet=group["outlet"],
            clipped_component_fraction=float((bounded!=group["fine"]).float().mean()),
            clipped_residue_fraction=float((bounded!=group["fine"]).any(-1).float().mean())))
    atomic_json(ROOT/"integration_check.json",dict(status="passed",checks=checks,codec_sha256=state_digest(codec),
        all8_original_first_sample_backbones_replayed=True,native_coordinates_loaded=False))
    paths=[Path(__file__),Path("upstream/scripts/fine_latent_range.sbatch"),SOURCE/"protocol.json",SOURCE/"evaluation_inputs.pt",SOURCE/"latent_input_range_audit.json",
        ROOT/"integration_check.json",LABELS,Path(p["fine_checkpoint"]),*[Path(v) for v in p["tokenizer"].values()]]
    workspace=Path.cwd().resolve()
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename:
            q=Path(filename).resolve()
            if q.suffix==".py" and q.is_relative_to(workspace/"upstream"):paths.append(q)
    p["source_sha256"]={str(q):sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT/"protocol.json",p);print("FINE_LATENT_RANGE_READY",flush=True)


@torch.no_grad()
def run(p):
    if (ROOT/"complete.json").exists():raise RuntimeError("Range pilot already complete")
    codec=load_codec(p,"centers3");before=state_digest(codec)
    groups=torch.load(SOURCE/"evaluation_inputs.pt",map_location="cpu",weights_only=True);saved=[]
    for original in groups:
        group=copy.copy(original);group["fine"]=group["fine"].clamp(-1,1)
        bb,audit=decode_latents(codec,group)
        folder=ROOT/f"seed{group['seed']}"/group["protein"];folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/(group["outlet"]+".npz"),backbone=bb,valid=np.isfinite(bb).all(axis=(1,2,3)))
        if group["outlet"]=="author_roundtrip":np.testing.assert_array_equal(bb,group["baseline"].numpy())
        saved.append((original,bb,audit))
        atomic_json(ROOT/"progress.json",dict(phase="decoding",completed_groups=len(saved),total_groups=8))
    atomic_json(ROOT/"prediction_complete.json",dict(status="complete",groups=8,decoded_samples=512,
        native_coordinates_loaded=False,negative_control_bitwise_identity=True))
    dev=torch.load(LABELS,map_location="cpu",weights_only=True);rows=[]
    for group,bb,audit in saved:
        reference=np.stack([c["target"].numpy() for c in dev["validation"][group["protein"]]])
        raw=evaluate_values(group["baseline"].numpy(),reference,dict(samples=64,failure_count=0,failures=[]))
        clipped=evaluate_values(bb,reference,audit)
        rows.append(dict(seed=group["seed"],protein=group["protein"],outlet=group["outlet"],original=raw,clipped=clipped))
        atomic_json(ROOT/"progress.json",dict(phase="evaluation",completed_groups=len(rows),total_groups=8))
        print("FINE_RANGE_EVALUATED",group["seed"],group["protein"],group["outlet"],flush=True)
    if before!=state_digest(codec) or any(v.requires_grad or v.grad is not None for v in codec.parameters()):
        raise RuntimeError("Range check updated codec")
    summaries={}
    for outlet in ["own_frozen_generator","author_roundtrip"]:
        selected=[r for r in rows if r["outlet"]==outlet];summaries[outlet]={}
        for arm in ["original","clipped"]:
            good=all(r[arm]["metrics_available"] for r in selected)
            value=dict(metrics_available=good,failure_count=sum(r[arm]["failure_count"] for r in selected))
            if good:
                value.update({k:float(np.mean([r[arm][k] for r in selected])) for k in METRIC_KEYS})
                value["geometry"]={k:float(np.mean([r[arm]["geometry"][k] for r in selected])) for k in GEOMETRY_KEYS}
            summaries[outlet][arm]=value
    atomic_json(ROOT/"complete.json",dict(status="complete_awaiting_review",rows=rows,summaries=summaries,
        codec_unchanged=True,optimizer_updates=0,negative_control_bitwise_identity=True,
        test_coordinates_opened=False,R3_opened=False,scope=p["scope"]))
    print("FINE_RANGE_COMPLETE",json.dumps(summaries),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","run"],required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=load(ROOT/"protocol.json");verify(p);run(p)


if __name__=="__main__":main()
