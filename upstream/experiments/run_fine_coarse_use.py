"""Read-only 2x2 decoder/coarse crossing of existing paired fine samples."""
import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from coarse_generation_diagnostic import pool_backbones, weighted_rmsd_matrix
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from fine_coarse_use import CELLS, COMPARISONS, crossed_inputs, response
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_aa_structural_coarse_condition import load_fine
from run_author_codec_interface import evaluate_values, load, state_digest, GEOMETRY_KEYS
from run_decoder_angle_adaptation import decode_latents
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec


ROOT=Path("artifacts/fine_coarse_use_v1")
PARENT=Path("artifacts/aa_structural_coarse_condition_v1")
SOURCE=Path("artifacts/decoder_angle_adaptation_v1")
AUTHOR=Path("artifacts/coarse_coverage_repair_v1")
FINE=Path("artifacts/multiframe_coarse_context_v1")
LABELS=Path("artifacts/multiframe_early_fine_v1/development_labels.pt")


def verify(p):
    for path,digest in p["source_sha256"].items():
        if sha(path)!=digest:raise RuntimeError("Frozen fine/coarse crossing input changed: "+path)


@torch.no_grad()
def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/"protocol.json").exists():raise RuntimeError("Fresh fine/coarse crossing namespace required")
    parent=load(PARENT/"protocol.json");verify(parent)
    if load(PARENT/"review.json")["status"]!="complete_reviewed_better_predicted_coarse_improves_global_error_fine_geometry_remains_poor":
        raise RuntimeError("Review the coarse-prompt intervention first")
    p={k:parent[k] for k in ["tokenizer","fine_checkpoint","feature_index","targets","seeds","gauges"]}
    p.update(cells=CELLS,samples=64,reference_frames=128,optimizer_updates=0,
        fine_std=load(AUTHOR/"protocol.json")["normalization"]["fine_std"],
        primary_gauge="raw",gauge_rationale="Raw is primary because it applies no extra orientation change. Both previously frozen alternative gauges remain required sensitivity controls; never select the best.",
        design="2x2 crossing: decoder coarse in{own,predicted-author} and fine sample in{conditioned-on-own,conditioned-on-predicted-author}. Both fine samples already exist from the same AA/base-noise/Euler16 keys. Decode all four cells with the SAME original frozen codec. Compare columns within each coarse row to isolate fine-sample response from decoder coarse placement.",
        interpretation="The crossed fine/coarse pairs can be outside the learned joint distribution; this is an intervention probe. Nonzero fine response is sensitivity, not useful structural information by itself. Small quality changes do not prove the model ignores coarse, and strong changes do not prove correct conditioning.",
        evaluation="Two reused development proteins, two seeds,64 paired sample indices, all3 fixed gauges. Verify both diagonal cells replay cached backbones bitwise. Save all3072 crossed outputs before loading native coordinates. Report nearest-reference metrics and both reference-angle errors, plus same-coarse pair deltas and direct CA displacement. Also score both fine alternatives against the SAME nearest coarse-reference frame to avoid a reference-selection confound.",
        fixed_reference_policy="For each sample and decoder-coarse row, find the nearest among the same128 native reference frames by residue-count-weighted proper-aligned block-center RMSD. This reference is shared by both fine-source alternatives and is used only in post-hoc evaluation.",
        compute="No new scored fine sampler calls; reuse the fixed own/prior fine caches. Real-weight preflight replays first samples using the same fine noise keys. Held-fine crosses require only decoder calls. No matched end-to-end deployment speed claim.",
        scope="Exploratory frozen-model conditional-use diagnostic on two repeatedly reused development sequences. No training, new MD sampling, inference selection, VQ, reserved-test/R3 access, independent confirmation, calibrated thermodynamics or automatic model replacement.",
        failure_policy="Keep every requested sample and NaN decoder-failure placeholder. Any failed sample leaves its group/pair metrics unavailable. Never drop residues, failed samples or proteins, and never resample or fall back.")
    cache=torch.load(SOURCE/"evaluation_inputs.pt",map_location="cpu",weights_only=True)
    prompts=torch.load(PARENT/"conditions.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");initial=state_digest(codec);groups=[];checks=[]
    paths=[Path(__file__),Path("upstream/experiments/fine_coarse_use.py"),
        Path("upstream/experiments/report_fine_coarse_use.py"),Path("upstream/tests/test_fine_coarse_use.py"),
        Path("upstream/scripts/fine_coarse_use.sbatch"),PARENT/"protocol.json",PARENT/"review.json",
        PARENT/"conditions.pt",SOURCE/"evaluation_inputs.pt",LABELS]
    for seed in p["seeds"]:
        model=load_fine(p,seed);model_hash=state_digest(model)
        for item in p["targets"]:
            name=item["protein"];aa=encode_aa(item["sequence"])
            own=next(g for g in cache if g["seed"]==seed and g["protein"]==name and g["outlet"]=="own_frozen_generator")
            key=component_seed(sample_seed(seed,name,0),"fine")
            torch.testing.assert_close(model.sample_fine(aa,own["coarse"][0],key,16),own["fine"][0],atol=0,rtol=0)
            for gauge in p["gauges"]:
                path=PARENT/f"seed{seed}/{name}/prior_{gauge}.npz";paths.append(path)
                with np.load(path) as z:
                    c=torch.from_numpy(z["coarse"].copy()).float();f=torch.from_numpy(z["fine"].copy()).float()
                    baseline=torch.from_numpy(z["backbone"].copy()).float();valid=z["valid"].copy()
                if not valid.all():raise RuntimeError("Source fine sample failure cannot be omitted")
                condition=next(g for g in prompts if g["seed"]==seed and g["protein"]==name and g["gauge"]==gauge)
                torch.testing.assert_close(c,condition["coarse"],atol=0,rtol=0)
                torch.testing.assert_close(model.sample_fine(aa,c[0],key,16),f[0],atol=0,rtol=0)
                cells=crossed_inputs(own["coarse"],own["fine"],c,f)
                for diagonal,expected in [("ownC_ownF",own["baseline"]),("priorC_priorF",baseline)]:
                    cc,ff=cells[diagonal]
                    for index in [0,63]:
                        torch.testing.assert_close(codec.decode(cc[index],ff[index]),expected[index],atol=0,rtol=0)
                groups.append(dict(seed=seed,protein=name,gauge=gauge,cells=cells,
                    diagonal_backbones={"ownC_ownF":own["baseline"],"priorC_priorF":baseline},
                    fine_response=response(own["fine"],f,p["fine_std"])))
                checks.append(dict(seed=seed,protein=name,gauge=gauge,base_fine_seed=key,
                    first_fine_sampler_bitwise_replayed=True,both_diagonal_end_samples_bitwise_replayed=True))
            paths.append(PARENT/f"seed{seed}/complete.json")
        if model_hash!=state_digest(model):raise RuntimeError("Preflight changed fine model")
    if initial!=state_digest(codec):raise RuntimeError("Preflight changed codec")
    torch.save(groups,ROOT/"inputs.pt")
    atomic_json(ROOT/"integration_check.json",dict(status="passed",checks=checks,
        native_coordinates_loaded=False,scored_new_sampler_calls=0,optimizer_updates=0))
    paths += [ROOT/"inputs.pt",ROOT/"integration_check.json"]
    workspace=Path.cwd().resolve()
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename:
            q=Path(filename).resolve()
            if q.suffix==".py" and q.is_relative_to(workspace/"upstream"):paths.append(q)
    # Include all fixed parent dependencies, including the actual model weights/features.
    p["source_sha256"]=dict(parent["source_sha256"])
    p["source_sha256"].update({str(q):sha(q) for q in dict.fromkeys(paths)})
    atomic_json(ROOT/"protocol.json",p);print("FINE_COARSE_USE_READY",len(groups),flush=True)


@torch.no_grad()
def run(p,seed):
    out=ROOT/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"started.json").exists():raise RuntimeError("Inspect a partial run before any separate retry")
    atomic_json(out/"started.json",dict(seed=seed,native_coordinates_loaded=False))
    groups=torch.load(ROOT/"inputs.pt",map_location="cpu",weights_only=True)
    groups=[g for g in groups if g["seed"]==seed]
    codec=load_codec(p,"centers3");initial=state_digest(codec);decoded=[];all_diag=0.
    for group in groups:
        name=group["protein"];gauge=group["gauge"];predictions={};audits={}
        folder=out/name/gauge;folder.mkdir(parents=True,exist_ok=True)
        for cell,(c,f) in group["cells"].items():
            bb,audit=decode_latents(codec,dict(coarse=c,fine=f,origin=torch.zeros((64,3))))
            if cell in group["diagonal_backbones"]:
                expected=group["diagonal_backbones"][cell].numpy()
                np.testing.assert_array_equal(bb,expected)
                all_diag=max(all_diag,float(np.max(np.abs(bb-expected))))
            np.savez_compressed(folder/(cell+".npz"),backbone=bb,valid=np.isfinite(bb).all(axis=(1,2,3)))
            predictions[cell]=bb;audits[cell]=audit
        decoded.append((group,predictions,audits))
        atomic_json(out/"progress.json",dict(phase="decoding",completed_groups=len(decoded),total_groups=6))
        print("FINE_COARSE_CROSSED",seed,name,gauge,flush=True)
    atomic_json(out/"prediction_complete.json",dict(status="complete",groups=6,decoded_samples=1536,
        native_coordinates_loaded=False,maximum_diagonal_replay_difference_A=all_diag))
    # Only now load native development coordinates, strictly for evaluation.
    dev=torch.load(LABELS,map_location="cpu",weights_only=True);rows=[];pairs=[]
    for group,predictions,audits in decoded:
        name=group["protein"];gauge=group["gauge"];cases=dev["validation"][name]
        target=next(t for t in p["targets"] if t["protein"]==name)
        if len(cases)!=128 or cases[0]["sequence"]!=target["sequence"]:
            raise RuntimeError("Fixed128 reference identity/count differs")
        reference=np.stack([c["target"].numpy() for c in cases]);rc=pool_backbones(reference)
        fixed_refs={}
        for coarse_row,cell in [("own","ownC_ownF"),("prior","priorC_ownF")]:
            c=group["cells"][cell][0].numpy()
            fixed_refs[coarse_row]=weighted_rmsd_matrix(c,rc,len(cases[0]["sequence"])).argmin(1)
        measured={};fixed_geometry={}
        for cell in CELLS:
            bb=predictions[cell];audit=audits[cell];measured[cell]=evaluate_values(bb,reference,audit)
            coarse_row="own" if cell.startswith("ownC_") else "prior"
            fixed_geometry[cell]=None
            if measured[cell]["metrics_available"]:
                values=[metrics(value,reference[int(i)]) for value,i in zip(bb,fixed_refs[coarse_row])]
                fixed_geometry[cell]={key:float(np.mean([v[key] for v in values])) for key in GEOMETRY_KEYS}
            rows.append(dict(seed=seed,protein=name,gauge=gauge,cell=cell,
                metrics=measured[cell],fixed_coarse_reference_geometry=fixed_geometry[cell]))
        for comparison,(held,changed) in COMPARISONS.items():
            available=measured[held]["metrics_available"] and measured[changed]["metrics_available"]
            row=dict(seed=seed,protein=name,gauge=gauge,comparison=comparison,
                held_cell=held,changed_cell=changed,metrics_available=available,
                fine_response=group["fine_response"])
            if available:
                delta=predictions[changed][:,:,1]-predictions[held][:,:,1]
                row["mean_direct_CA_displacement_A"]=float(np.mean(np.sqrt(np.sum(delta**2,axis=-1).mean(1))))
                from run_author_codec_interface import METRIC_KEYS
                row["changed_minus_held"]={key:measured[changed][key]-measured[held][key] for key in METRIC_KEYS}
                row["changed_minus_held"]["geometry"]={key:measured[changed]["geometry"][key]-measured[held]["geometry"][key] for key in GEOMETRY_KEYS}
                row["changed_minus_held"]["fixed_coarse_reference_geometry"]={key:fixed_geometry[changed][key]-fixed_geometry[held][key] for key in GEOMETRY_KEYS}
            pairs.append(row)
        atomic_json(out/"progress.json",dict(phase="evaluation",completed_groups=len(rows),total_groups=24))
        print("FINE_COARSE_CROSS_EVALUATED",seed,name,gauge,flush=True)
    if initial!=state_digest(codec) or any(v.requires_grad or v.grad is not None for v in codec.parameters()):
        raise RuntimeError("Codec changed during crossing")
    atomic_json(out/"complete.json",dict(status="complete",seed=seed,rows=rows,pairs=pairs,
        codec_unchanged=True,diagonal_backbones_bitwise_replayed=True,all_samples_retained=True,
        optimizer_updates=0,new_scored_fine_sampler_calls=0,test_coordinates_opened=False,R3_opened=False,
        host=platform.node(),cpu_threads=torch.get_num_threads(),scope=p["scope"]))
    atomic_json(out/"progress.json",dict(phase="complete",completed_groups=24,total_groups=24))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","run"],required=True)
    parser.add_argument("--seed",type=int,choices=[2026,2027]);args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        if args.seed is None:raise ValueError("Declared seed required")
        p=load(ROOT/"protocol.json");verify(p);run(p,args.seed)


if __name__=="__main__":main()
