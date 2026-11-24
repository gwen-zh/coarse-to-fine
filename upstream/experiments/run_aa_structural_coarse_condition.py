"""Frozen fine-flow sampling conditioned on AA-only predicted block means."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from aa_structural_coarse_condition import GAUGES, coarse_prompt, center_distances
from block_local_frames import FrameGeometryError
from coarse_generation_diagnostic import coarse_metrics, pool_backbones
from diagnostic_io import atomic_json
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_author_codec_interface import evaluate_values, load, state_digest
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec


ROOT=Path("artifacts/aa_structural_coarse_condition_v1")
SOURCE=Path("artifacts/decoder_angle_adaptation_v1")
AUTHOR=Path("artifacts/coarse_coverage_repair_v1")
FINE=Path("artifacts/multiframe_coarse_context_v1")
LABELS=Path("artifacts/multiframe_early_fine_v1/development_labels.pt")
BASELINES=["own","raw_author","encoded_author"]
METHODS=BASELINES+["prior_"+g for g in GAUGES]


def verify(p):
    for path,digest in p["source_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("Frozen structural-coarse input changed: "+path)


def load_fine(p,seed):
    spec=load(AUTHOR/"protocol.json")
    model=ExplicitCoarseFineFlow(**spec["normalization"],**spec["model"],coarse_dim=3,
        feature_mode="ESM3_context",feature_dim=spec["feature_dimension"],context_kind="local_geometry")
    saved=torch.load(FINE/f"local_geometry/seed{seed}/last.pt",map_location="cpu",weights_only=False)
    if saved["updates"]!=8192 or saved["arm"]!="local_geometry" or saved["protocol_sha256"]!=sha(FINE/"protocol.json"):
        raise RuntimeError("Wrong immutable fine generator")
    model.load_state_dict({k[len("fine_model."):]:v for k,v in saved["model"].items() if k.startswith("fine_model.")},strict=True)
    index=load(p["feature_index"]);bank={}
    for item in p["targets"]:
        with np.load(index[item["sequence"]]["path"]) as z:bank[item["sequence"]]=torch.from_numpy(z["h"].copy()).float()
    model.attach_feature_bank(bank)
    return model.eval().requires_grad_(False)


@torch.no_grad()
def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():raise RuntimeError("Fresh structural-coarse namespace required")
    old=load(SOURCE/"protocol.json")
    p={k:old[k] for k in ["tokenizer","fine_checkpoint","feature_index","targets","seeds"]}
    p.update(gauges=GAUGES,methods=METHODS,samples=64,fine_nfe=16,optimizer_updates=0,
        architecture="Original own decoder and original geometry-aware fine8192 flow remain frozen. Replace only coarse conditions by physical CA block means extracted from original25-step author's FINAL AA-only predicted backbone. These are not the author's coarse token IDs. No encoded source fine is supplied to the grafted fine sampler.",
        gauges_definition={"raw":"Global CA translation removed; retain author's predicted orientation.",
            "first_residue":"Center, then orient by predicted first-residue N/CA/C; no native reference.",
            "align_to_own":"Center, then proper residue-count-weighted Kabsch rotation to the SAME sample's own predicted block means. No shape change, scaling or native reference."},
        matching="For every sequence/seed/sample all three gauges reuse identical base fine noise and Euler16 calls. Report all gauges rather than selecting one. Original own coarse/fine samples and source author baselines are fixed cached outputs. This is not matched end-to-end compute: obtaining each author-derived coarse prompt requires the author's full generator AND decoder, then our fine sampler/decoder.",
        interpretation="A read-only intervention on sampled coarse, not new training, a PAR reproduction or a standalone new generator. A better coarse prompt can expose remaining fine-compatibility problems; this does not uniquely attribute all error to one component. Gauge sensitivity is expected because the current fine Transformer uses absolute normalized XYZ.",
        evaluation="Two repeatedly reused development sequences, two seeds,64 raw samples each. Cache every generated structure before native-coordinate evaluation. Compare own baseline, raw author, author encoded->decoded by own codec, and all3 grafted gauges. Evaluate accuracy, reverse coverage, C-N and angles, distribution, spread and physical coarse lower bounds. Native128-reference frames enter only after sampling.",
        scope="Exploratory frozen-generator coarse-conditioning diagnostic. No native deployment input, training, new MD data, VQ, sample selection, independent test/R3 access, thermodynamic calibration or automatic default promotion.",
        failure_policy="Retain all sample indices; undefined frames/nonfinite sampling receive raw NaN backbone/fine placeholders. Any failed sample makes that group's full metrics unavailable. Never resample/filter/select alternatives. Unexpected shape or feature errors stop the trial.")
    cache=torch.load(SOURCE/"evaluation_inputs.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");codec_hash=state_digest(codec);conditions=[];checks=[]
    for seed in p["seeds"]:
        model=load_fine(p,seed);before=state_digest(model)
        for item in p["targets"]:
            name=item["protein"];n=item["length"]
            own=next(g for g in cache if g["seed"]==seed and g["protein"]==name and g["outlet"]=="own_frozen_generator")
            source=next(g for g in cache if g["seed"]==seed and g["protein"]==name and g["outlet"]=="author_roundtrip")
            aa=encode_aa(item["sequence"])
            key=component_seed(sample_seed(seed,name,0),"fine")
            torch.testing.assert_close(model.sample_fine(aa,own["coarse"][0],key,16),own["fine"][0],atol=0,rtol=0)
            baseline=dict(own=own["baseline"],raw_author=source["original_author"],encoded_author=source["baseline"])
            for gauge in GAUGES:
                c_values=[];rotations=[];maximum=0.
                for i in range(64):
                    c,rotation=coarse_prompt(source["original_author"][i],own["coarse"][i],gauge)
                    torch.testing.assert_close(center_distances(c),center_distances(source["coarse"][i]),atol=2e-5,rtol=2e-5)
                    # With encoded source fine, changing orientation must only rotate output.
                    rotated=codec.decode(c,source["fine"][i])
                    expected=(source["baseline"][i]-source["origin"][i])@rotation
                    torch.testing.assert_close(rotated,expected,atol=2e-4,rtol=2e-5)
                    maximum=max(maximum,float((rotated-expected).abs().max()))
                    c_values.append(c);rotations.append(rotation)
                conditions.append(dict(seed=seed,protein=name,gauge=gauge,coarse=torch.stack(c_values),
                    rotations=torch.stack(rotations),baselines=baseline))
                checks.append(dict(seed=seed,protein=name,gauge=gauge,samples=64,
                    encoded_fine_rotation_control_max_abs_A=maximum))
            if state_digest(model)!=before:raise RuntimeError("Fine generator changed in preflight")
            print("STRUCTURAL_COARSE_PREFLIGHT",seed,name,flush=True)
    if state_digest(codec)!=codec_hash:raise RuntimeError("Codec changed in preflight")
    torch.save(conditions,ROOT/"conditions.pt")
    atomic_json(ROOT/"integration_check.json",dict(status="passed",checks=checks,
        all768_source_encoded_fine_rotation_controls_passed=True,
        original_fine_first_samples_bitwise_replayed=True,native_coordinates_loaded=False,optimizer_updates=0))
    paths=[Path(__file__),Path("upstream/experiments/aa_structural_coarse_condition.py"),
        Path("upstream/experiments/report_aa_structural_coarse_condition.py"),
        Path("upstream/tests/test_aa_structural_coarse_condition.py"),Path("upstream/scripts/aa_structural_coarse_condition.sbatch"),
        SOURCE/"protocol.json",SOURCE/"evaluation_inputs.pt",AUTHOR/"protocol.json",FINE/"protocol.json",
        ROOT/"conditions.pt",ROOT/"integration_check.json",LABELS,Path(p["feature_index"]),
        Path(p["fine_checkpoint"]),*[Path(v) for v in p["tokenizer"].values()]]
    paths += [FINE/f"local_geometry/seed{seed}/last.pt" for seed in p["seeds"]]
    paths += sorted(ROOT.glob("preflight_failure_job*.json"))
    index=load(p["feature_index"])
    paths += [Path(index[t["sequence"]]["path"]) for t in p["targets"]]
    workspace=Path.cwd().resolve()
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename:
            q=Path(filename).resolve()
            if q.suffix==".py" and q.is_relative_to(workspace/"upstream"):paths.append(q)
    p["source_sha256"]={str(q):sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT/"protocol.json",p);print("AA_STRUCTURAL_COARSE_READY",len(conditions),flush=True)


@torch.no_grad()
def run(p,seed):
    out=ROOT/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"started.json").exists():raise RuntimeError("Inspect partial run before a separate retry namespace")
    atomic_json(out/"started.json",dict(seed=seed,native_coordinates_loaded=False))
    model=load_fine(p,seed);codec=load_codec(p,"centers3");expected=(state_digest(model),state_digest(codec))
    conditions=torch.load(ROOT/"conditions.pt",map_location="cpu",weights_only=True)
    conditions=[g for g in conditions if g["seed"]==seed];records=[]
    for group in conditions:
        name=group["protein"];method="prior_"+group["gauge"]
        item=next(t for t in p["targets"] if t["protein"]==name);n=item["length"];aa=encode_aa(item["sequence"])
        backbones=[];fine=[];failures=[];seconds=[]
        for i,c in enumerate(group["coarse"]):
            started=time.perf_counter()
            try:
                f=model.sample_fine(aa,c,component_seed(sample_seed(seed,name,i),"fine"),16)
                if not torch.isfinite(f).all():raise FrameGeometryError("Nonfinite fine endpoint")
                value=codec.decode(c,f)
                if not torch.isfinite(value).all():raise FrameGeometryError("Nonfinite backbone")
                torch.testing.assert_close(pool_ca(value),c,atol=2e-5,rtol=2e-5)
                fine.append(f.numpy());backbones.append(value.numpy())
            except FrameGeometryError as error:
                fine.append(np.full((n,8),np.nan,np.float32));backbones.append(np.full((n,3,3),np.nan,np.float32))
                failures.append(dict(sample=i,code="fine_frame_or_nonfinite_failure",message=str(error)))
            except ValueError as error:
                if "finite" not in str(error).lower():raise
                fine.append(np.full((n,8),np.nan,np.float32));backbones.append(np.full((n,3,3),np.nan,np.float32))
                failures.append(dict(sample=i,code="nonfinite_sampling_state",message=str(error)))
            seconds.append(time.perf_counter()-started)
            if (i+1)%16==0:
                atomic_json(out/"progress.json",dict(phase="sampling",protein=name,method=method,samples=i+1,
                    total_per_group=64,completed_groups=len(records)))
        bb=np.stack(backbones);folder=out/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(method+".npz"),backbone=bb,fine=np.stack(fine),coarse=group["coarse"].numpy(),
            valid=np.isfinite(bb).all(axis=(1,2,3)))
        audit=dict(samples=64,failure_count=len(failures),failures=failures,all_samples_retained=True,
            mean_own_fine_and_decode_seconds=float(np.mean(seconds)),
            timing_scope="Own warm fine16 + decoder only. Excludes cached author's full generator/decoder, AA features, condition preparation and metric evaluation. Not full pipeline latency or a speedup claim.")
        records.append((group,method,bb,audit))
        print("STRUCTURAL_COARSE_SAMPLED",seed,name,method,len(failures),flush=True)
    atomic_json(out/"sampling_complete.json",dict(status="complete",groups=6,generated_samples=384,
        native_coordinates_loaded=False))
    # References are used only after every generated fine and backbone is saved.
    dev=torch.load(LABELS,map_location="cpu",weights_only=True);rows=[]
    for item in p["targets"]:
        name=item["protein"];n=item["length"];cases=dev["validation"][name]
        if len(cases)!=128 or cases[0]["sequence"]!=item["sequence"]:raise RuntimeError("Wrong reference identity")
        reference=np.stack([c["target"].numpy() for c in cases]);rc=pool_backbones(reference)
        selected=[r for r in records if r[0]["protein"]==name]
        entries=[(method,selected[0][0]["baselines"][method].numpy(),None,
            dict(samples=64,failure_count=0,failures=[])) for method in BASELINES]
        entries += [(method,bb,group["coarse"].numpy(),audit) for group,method,bb,audit in selected]
        for method,bb,c,audit in entries:
            measured=evaluate_values(bb,reference,audit)
            if c is None:c=pool_backbones(bb)
            cm=coarse_metrics(c,rc,n)
            if measured["metrics_available"] and cm["mean_sample_nearest_reference_center_rmsd_A"]>measured["mean_sample_nearest_reference_ca_rmsd_A"]+1e-4:
                raise RuntimeError("Fixed-center CA lower bound violated")
            rows.append(dict(seed=seed,protein=name,method=method,metrics=measured,coarse_metrics=cm))
            atomic_json(out/"progress.json",dict(phase="evaluation",completed_groups=len(rows),total_groups=12))
            print("STRUCTURAL_COARSE_EVALUATED",seed,name,method,flush=True)
    if expected!=(state_digest(model),state_digest(codec)) or any(v.requires_grad or v.grad is not None for m in [model,codec] for v in m.parameters()):
        raise RuntimeError("Frozen model or codec changed")
    atomic_json(out/"complete.json",dict(status="complete",seed=seed,rows=rows,model_and_codec_unchanged=True,
        optimizer_updates=0,all_samples_retained=True,test_coordinates_opened=False,R3_opened=False,
        host=platform.node(),cpu_threads=torch.get_num_threads(),scope=p["scope"]))
    atomic_json(out/"progress.json",dict(phase="complete",completed_groups=12,total_groups=12))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","run"],required=True)
    parser.add_argument("--seed",type=int,choices=[2026,2027]);args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        if args.seed is None:raise ValueError("Declared seed required")
        p=load(ROOT/"protocol.json");verify(p);run(p,args.seed)


if __name__=="__main__":main()
