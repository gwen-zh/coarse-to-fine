"""Freeze both completed interleaved models and diagnose training examples only."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from latent_source_diagnostic import METHODS, training_cases, diagnostic_seed, decode_sources
from run_sequence_condition_probe import make_model
from run_sequence_multiscale_ensemble import load_codec
from structural_metrics import aligned_rmsd

RUN=Path("artifacts/latent_source_diagnostic_v1")
PARENT=Path("artifacts/generator_curriculum_v1")


def verify(p):
    for name,digest in p["source_sha256"].items():
        if sha(name)!=digest:raise RuntimeError("Changed diagnostic input: "+name)


def prepare():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/"protocol.json").exists():raise RuntimeError("Diagnostic already frozen")
    parent=json.loads((PARENT/"protocol.json").read_text())
    verify(parent)
    report=json.loads((PARENT/"report.json").read_text())
    if not report["exact_event_multisets_and_noise_verified"]:raise RuntimeError("Unverified parent")
    groups=torch.load(PARENT/"labels.pt",map_location="cpu",weights_only=True)
    selected=training_cases(groups,parent["pretrain_probe_ids"],4)
    # Exclude validation/reused groups from the new diagnostic artifact entirely.
    case_map={r["case"]["id"]:r["case"] for r in selected}
    torch.save(dict(cases=case_map,rows=[{k:v for k,v in row.items() if k!="case"}
        |dict(case_id=row["case"]["id"]) for row in selected]),RUN/"training_labels.pt")
    p=copy.deepcopy(parent)
    checkpoints={str(seed):str(PARENT/"interleaved"/f"seed{seed}"/"last.pt")
                 for seed in [2026,2027]}
    p.update(parent_trial=str(PARENT/"protocol.json"),seeds=[2026,2027],checkpoints=checkpoints,
        methods=METHODS,cases_per_model=len(selected),samples_per_protein=4,
        PDB_probe_ids=parent["pretrain_probe_ids"],MD_train_proteins=sorted(groups["train"]),
        training=False,model_selection=False,coarse_nfe=16,fine_nfe_controls=[16,64],
        selection="Eight PDB training IDs fixed in the preceding protocol and all five MD training proteins. Four fixed noise draws per protein; MD labels at four uniformly spaced training indices. Equal protein weights, separate PDB/MD summaries.",
        oracle_policy="True coarse/fine paths use native training structures and are not deployment. For generated coarse plus oracle fine, recompute fine from the same target in that coarse-derived frame; never transplant an unchanged fine label. Encoder removes target block translation, so this path cannot restore displaced coarse centers.",
        controls="Fine Euler16 exactly replays the existing sampler; Euler64 uses the same initial noise with four times the fine NFE and fixed coarse. Clip, snap and one encode/decode cycle are fixed diagnostics on the same true-coarse samples. No tuning, acceptance gates, reference-based candidate selection or test coordinates.",
        primary="Locate failures on training examples. Mean C-N rate, boundary/within rates and reference error; report all fixed paths and both seeds. These oracle/control results cannot be called an AA-only improvement.",
        scope="Training-only diagnostic. No validation, reserved test, reused R2 or R3 coordinate tensors are included or evaluated. Parent checkpoints previously used development data for evaluation only.",
        stop_rule="One fixed diagnostic pass; no optimizer steps, checkpoint selection or automatic replacement. Make decisions for later training only after reporting every path.")
    paths=[Path(__file__),Path("upstream/experiments/latent_source_diagnostic.py"),
        Path("upstream/tests/test_latent_source_diagnostic.py"),
        Path("upstream/scripts/latent_source_diagnostic.sbatch"),
        Path("upstream/experiments/report_latent_source_diagnostic.py"),
        PARENT/"protocol.json",PARENT/"report.json",RUN/"training_labels.pt",
        *[Path(v) for v in checkpoints.values()]]
    p["source_sha256"].update({str(v):sha(v) for v in paths})
    atomic_json(RUN/"protocol.json",p)
    print("LATENT_SOURCE_DIAGNOSTIC_READY",len(selected),flush=True)


@torch.no_grad()
def endpoint_probe(model,aa,coarse,fine,seed):
    rng=torch.Generator().manual_seed(seed)
    c=model.normalize_coarse(coarse);f=model.normalize_fine(fine)
    nc=torch.randn(c.shape,generator=rng);nf=torch.randn(f.shape,generator=rng)
    results=[]
    for t in [.1,.5,.9]:
        sc=(1-t)*nc+t*c;sf=(1-t)*nf+t*f
        vc=model.coarse_velocity(aa,sc,t);vf=model.fine_velocity(aa,sf,t,c)
        cp=model.denormalize_coarse(sc+(1-t)*vc)
        fp=model.denormalize_fine(sf+(1-t)*vf)
        results.append(dict(time=t,coarse_velocity_mse=float((vc-(c-nc)).square().mean()),
            fine_velocity_mse=float((vf-(f-nf)).square().mean()),
            coarse_endpoint_coordinate_rmse=float((cp-coarse).square().mean().sqrt()),
            fine_endpoint_latent_rmse=float((fp-fine).square().mean().sqrt())))
    return results


@torch.no_grad()
def diagnose(p,seed):
    out=RUN/f"seed{seed}";out.mkdir(exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Diagnostic already complete")
    labels=torch.load(RUN/"training_labels.pt",map_location="cpu",weights_only=True)
    if any(c["split"]!="train" for c in labels["cases"].values()):raise RuntimeError("Nontraining label")
    codec=load_codec(p,"centers3")
    model=make_model(p,"ESM3_context")
    checkpoint=torch.load(p["checkpoints"][str(seed)],map_location="cpu",weights_only=False)
    if checkpoint["updates"]!=12288 or checkpoint["protocol_sha256"]!=sha(PARENT/"protocol.json"):
        raise RuntimeError("Unexpected generator endpoint")
    model.load_state_dict(checkpoint["model"]);model.eval().requires_grad_(False)
    snapshots={name:v.clone() for name,v in model.state_dict().items()}
    records=[];start=time.perf_counter();maximum_center_error=0.
    for index,row in enumerate(labels["rows"]):
        case=labels["cases"][row["case_id"]]
        aa=encode_aa(case["sequence"]);target=case["target"]
        key=diagnostic_seed(seed,case["id"],row["draw"])
        decoded,stats=decode_sources(model,codec,aa,target,key)
        torch.testing.assert_close(stats["native_coarse"],case["coarse"],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(stats["native_fine"],case["fine_continuous"],rtol=1e-5,atol=1e-5)
        # Replay full generation against the production path, on the first case.
        if index==0:
            from sequence_multiscale_ensemble import sample_levels
            c,f=sample_levels(model,codec,aa,key,16,16)
            torch.testing.assert_close(c,stats["generated_coarse"],rtol=0,atol=0)
            torch.testing.assert_close(f,stats["generated_fine16"],rtol=0,atol=0)
        method_rows=[];arrays={"target":target.numpy()}
        for method,bb in decoded.items():
            expected=stats["generated_coarse"] if method in {
                "generated_coarse_reencoded_native_fine","full16","coarse16_fine64"} else stats["native_coarse"]
            maximum_center_error=max(maximum_center_error,float((pool_ca(bb)-expected).abs().max()))
            torch.testing.assert_close(pool_ca(bb),expected,atol=2e-5,rtol=2e-5)
            values=metrics(bb.numpy(),target.numpy())
            if method!="native_reconstruction":
                base=decoded["true_coarse_generated_fine16"] if method.startswith("true_coarse") else decoded["full16"]
                values["aligned_ca_drift_from_corresponding_fine16_A"]=aligned_rmsd(bb[:,1].numpy(),base[:,1].numpy())
            method_rows.append(dict(method=method,metrics=values));arrays[method]=bb.numpy()
        c=stats["generated_coarse"];native=stats["native_coarse"]
        record=dict(**row,seed=seed,methods=method_rows,support=stats["support"],
            fine_cycle_latent_rmse=stats["true_coarse_fine_cycle_rmse"],
            coarse_aligned_ca_center_rmsd_A=aligned_rmsd(c.numpy(),native.numpy()),
            generated_adjacent_center_distance_mean_A=float((c[1:]-c[:-1]).norm(dim=-1).mean()),
            native_adjacent_center_distance_mean_A=float((native[1:]-native[:-1]).norm(dim=-1).mean()),
            endpoint_probe=endpoint_probe(model,aa,native,stats["native_fine"],key))
        records.append(record)
        np.savez_compressed(out/f"case_{index:03d}.npz",**arrays,
            native_coarse=native.numpy(),generated_coarse=c.numpy(),
            native_fine=stats["native_fine"].numpy(),oracle_fine16=stats["oracle_fine16"].numpy(),
            generated_fine16=stats["generated_fine16"].numpy())
        atomic_json(out/"progress.json",dict(completed=index+1,total=len(labels["rows"]),
            last_case=case["id"],seconds=time.perf_counter()-start))
        print("LATENT_SOURCE",seed,index+1,len(labels["rows"]),case["id"],flush=True)
    for name,v in model.state_dict().items():torch.testing.assert_close(v,snapshots[name],atol=0,rtol=0)
    verify(p)
    atomic_json(out/"complete.json",dict(status="complete",seed=seed,records=records,
        seconds=time.perf_counter()-start,maximum_centroid_error_A=maximum_center_error,
        exact_production_replay_verified=True,generator_unchanged=True,
        test_coordinates_opened=False,validation_coordinates_evaluated=False,R3_opened=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","diagnose"],required=True)
    parser.add_argument("--seed",type=int,default=2026);args=parser.parse_args()
    torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=json.loads((RUN/"protocol.json").read_text());verify(p)
        if args.seed not in p["seeds"]:raise RuntimeError("Undeclared seed")
        diagnose(p,args.seed)
