"""Fixed-budget decoder-only trial; encoders, latent pairs and generators frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from block_local_frames import FrameGeometryError
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from decoder_angle_adaptation import ARMS, enable_decoder, frozen_digest, update
from diagnostic_io import atomic_json
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from fine_geometry_flow import sample_training_batch
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from joint_sequence_predictor import encode_aa
from run_author_codec_interface import evaluate_values, load, state_digest, METRIC_KEYS, GEOMETRY_KEYS
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec
from structural_metrics import aligned_rmsd


ROOT = Path("artifacts/decoder_angle_adaptation_v1")
PRIOR = Path("artifacts/author_codec_interface_v1")
FULL = Path("artifacts/coarse_full_cohort_v1")
AUTHOR = Path("artifacts/coarse_coverage_repair_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")
FINE = Path("artifacts/multiframe_coarse_context_v1")


def batch_rng(seed):
    return np.random.default_rng(int(seed) + 33571)


def identity(row):
    return {key: row[key] for key in ["batch", "lr"]}


def verify(p):
    for path, digest in p["source_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen decoder adaptation input changed: " + path)


@torch.no_grad()
def prepare_evaluation_inputs(p, codec):
    """Read cached AA-only outputs/features, never native development structures."""
    old = load(AUTHOR / "protocol.json"); feature_index = load(p["feature_index"])
    bank = {}
    for item in p["targets"]:
        with np.load(feature_index[item["sequence"]]["path"]) as z:
            bank[item["sequence"]] = torch.from_numpy(z["h"].copy()).float()
    groups = []
    for seed in p["seeds"]:
        model = ExplicitCoarseFineFlow(**old["normalization"], **old["model"], coarse_dim=3,
            feature_mode="ESM3_context", feature_dim=old["feature_dimension"], context_kind="local_geometry")
        saved = torch.load(FINE / f"local_geometry/seed{seed}/last.pt", map_location="cpu", weights_only=False)
        if saved["updates"] != 8192 or saved["arm"] != "local_geometry" or saved["protocol_sha256"] != sha(FINE / "protocol.json"):
            raise RuntimeError("Wrong immutable fine generator")
        model.load_state_dict({k[len("fine_model."):]: v for k, v in saved["model"].items() if k.startswith("fine_model.")}, strict=True)
        model.attach_feature_bank(bank); model.eval().requires_grad_(False)
        for item in p["targets"]:
            name = item["protein"]; n = item["length"]
            with np.load(AUTHOR / f"seed{seed}/author/{name}_steps25_backbone.npz") as z:
                original_author = torch.from_numpy(z["backbone"].copy()).float()
            with np.load(PRIOR / f"seed{seed}/{name}/projected.npz") as z:
                c = torch.from_numpy(z["coarse"].copy()).float()
                f = torch.from_numpy(z["fine"].copy()).float()
                baseline = z["backbone"].copy()
            origin = original_author[:, :, 1].mean(1)
            check = np.stack([(codec.decode(cc, ff) + oo).numpy() for cc, ff, oo in zip(c, f, origin)])
            np.testing.assert_allclose(check, baseline, atol=2e-5, rtol=2e-5)
            groups.append(dict(seed=seed, protein=name, outlet="author_roundtrip", coarse=c, fine=f,
                origin=origin, baseline=torch.from_numpy(baseline), original_author=original_author,
                replay_max_abs_A=float(np.max(np.abs(check - baseline)))))
            with np.load(FULL / f"seed{seed}/step0/ensembles/{name}/full.npz") as z:
                c = torch.from_numpy(z["coarse"].copy()).float(); previous = z["backbone"].copy()
            aa = encode_aa(item["sequence"]); fine = []
            for index, centers in enumerate(c):
                key = component_seed(sample_seed(seed, name, index), "fine")
                fine.append(model.sample_fine(aa, centers, key, 16))
            f = torch.stack(fine); baseline = torch.stack([codec.decode(cc, ff) for cc, ff in zip(c, f)])
            # Predeclared cross-host FP32 replay tolerance, far below quality gaps.
            np.testing.assert_allclose(baseline.numpy(), previous, atol=1e-3, rtol=2e-5)
            groups.append(dict(seed=seed, protein=name, outlet="own_frozen_generator", coarse=c, fine=f,
                origin=torch.zeros((64, 3)), baseline=baseline,
                replay_max_abs_A=float(np.max(np.abs(baseline.numpy() - previous)))))
            if c.shape != (64, (n + 3) // 4, 3) or f.shape != (64, n, 8):
                raise RuntimeError("Unexpected own latent cohort")
            print("DECODER_EVALUATION_INPUTS", seed, name, flush=True)
    torch.save(groups, ROOT / "evaluation_inputs.pt")
    atomic_json(ROOT / "evaluation_input_audit.json", dict(status="passed", groups=len(groups),
        generated_latent_pairs=sum(len(g["fine"]) for g in groups), native_coordinates_loaded=False,
        own_replay_absolute_tolerance_A=.001, own_replay_relative_tolerance=2e-5,
        replay=[{k:g[k] for k in ["seed", "protein", "outlet", "replay_max_abs_A"]} for g in groups]))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Fresh decoder adaptation namespace required")
    if load(PRIOR / "review.json")["status"] != "complete_reviewed_codec_preserves_prior_with_cn_gain_angle_tradeoff":
        raise RuntimeError("Review the read-only interface first")
    old = load(AUTHOR / "protocol.json"); prior = load(PRIOR / "protocol.json")
    p = dict(tokenizer=prior["tokenizer"], fine_checkpoint=prior["fine_checkpoint"],
        feature_index=old["all_feature_index"], targets=prior["targets"], seeds=[2026,2027], arms=ARMS,
        updates=2048, batch_size=4, lr=1e-4, min_lr=1e-5, weight_decay=1e-4, gradient_clip=1.,
        angle_weight=.01, angle_scale_degrees=5., checkpoint_every=512, progress_every=128,
        PDB_events_per_arm_seed=6144, MD_events_per_arm_seed=2048,
        initialization="Every arm/data-seed starts from the SAME current codec checkpoint (representation-pretraining seed2026). Seeds2026/2027 vary paired training draws, not encoder/decoder pretraining. Only fine_tokenizer.decoder parameters may change.",
        architecture="Physical centers3 and continuous fine8 remain unchanged. Frozen fine encoder, optional book, all coarse maps and both sequence generators. Residue rigid N/CA/C output and exact block CA-center contract retained. No VQ/snap, new module or generator adaptation.",
        objective="Both arms: original coordinateMSE + C-N lengthMSE + boundary C-N MSE +.25 angle-cosineMSE +.1 CA-clash hinge. Angular arm additionally adds.01 times the average of all-bond and boundary-bond squared angular errors, normalized by5degrees. Both evaluate this extra angular branch; its weight is0 in continuation. No lambda scan.",
        training="Exactly2048 AdamW updates, three uniform static PDB targets plus one uniform native MD training protein/frame each update. Cached native coarse/fine from the same structure and frozen encoder. Center target globally to match centers3 gauge. No author-generated structure or development sample enters training.",
        matching="Same decoder parameters and initial weights, batches, LR and update count in both arms, paired for each data-seed. Both run the angular branch. Report elapsed time; no matched inference-resource claim against author pretraining.",
        evaluation="Final endpoint only. Reuse fixed64x2x2 latents in each of two outlets: author-generated structure->frozen encoder, and our original parent coarse+geometry-aware fine generators. Decode all outputs before reading native development coordinates. Compare against original codec on the exact same latent pairs, plus raw author baseline. All128 references per sequence, raw samples retained. Native roundtrip is a separately labelled oracle diagnostic.",
        stop_rule="Fixed2048 endpoint; checkpoints for recovery only. No best checkpoint, budget extension, loss-weight sweep, inference selection, automatic model promotion, reserved test or R3 access.",
        scope="Exploratory decoder-only adaptation on899 static PDB structures plus2264 frames from5 seen MD training proteins. Evaluation on2 repeatedly reused development proteins; no independent validation or calibrated thermodynamic claim.",
        failure_policy="Stop on nonfinite training losses/gradients. Keep NaN placeholders for decoding frame failures; any failed sample makes its full group metrics unavailable. No filtering, fallback or selected candidate.")
    train = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    if len(train["PDB_train"]) != 899 or sum(map(len, train["train"].values())) != 2264:
        raise RuntimeError("Unexpected fixed training cohort")
    codec = load_codec(p, "centers3"); original = state_digest(codec); checks = []
    cases = train["PDB_train"][:2] + [c for frames in train["train"].values() for c in [frames[0], frames[-1]]]
    for case in cases:
        c, f = codec.encode(case["target"])
        torch.testing.assert_close(c, case["coarse"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(f, case["fine_continuous"], atol=1e-5, rtol=1e-5)
    for arm in ARMS:
        candidate = load_codec(p, "centers3"); params = enable_decoder(candidate); frozen = frozen_digest(candidate)
        optimizer = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
        example = train["PDB_train"][0]; before = candidate.encode(example["target"])
        started = time.perf_counter(); row = update(candidate, optimizer, sample_training_batch(train, batch_rng(2026)), arm, 0, p)
        elapsed = time.perf_counter() - started
        if row["gradient_norm"] <= 0 or state_digest(candidate) == original or frozen_digest(candidate) != frozen:
            raise RuntimeError("Decoder-only real update failed")
        for a,b in zip(before,candidate.encode(example["target"])):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        checks.append(dict(arm=arm, trainable_parameters=sum(v.numel() for v in params),
            frozen_components_unchanged=True, encoder_outputs_bitwise_unchanged=True,
            gradient_norm=row["gradient_norm"], one_update_seconds=elapsed))
    p["trainable_parameters"] = checks[0]["trainable_parameters"]
    prepare_evaluation_inputs(p, codec)
    atomic_json(ROOT / "integration_check.json", dict(status="passed", checks=checks,
        cached_native_labels_replayed=len(cases), discarded_preflight_updates=2, actual_trial_updates=0))
    paths = [Path(__file__), Path("upstream/experiments/decoder_angle_adaptation.py"),
        Path("upstream/experiments/report_decoder_angle_adaptation.py"),
        Path("upstream/tests/test_decoder_angle_adaptation.py"), Path("upstream/scripts/decoder_angle_adaptation.sbatch"),
        PRIOR / "protocol.json", PRIOR / "review.json", PRIOR / "complete.json", AUTHOR / "protocol.json",
        LABELS / "train_labels.pt", LABELS / "development_labels.pt", FINE / "protocol.json",
        ROOT / "evaluation_inputs.pt", ROOT / "evaluation_input_audit.json", ROOT / "integration_check.json",
        Path(p["feature_index"]), Path(p["fine_checkpoint"]), *[Path(v) for v in p["tokenizer"].values()]]
    index = load(p["feature_index"])
    for item in p["targets"]:
        paths.append(Path(index[item["sequence"]]["path"]))
        for seed in p["seeds"]:
            name = item["protein"]
            paths += [AUTHOR / f"seed{seed}/author/{name}_steps25_backbone.npz", AUTHOR / f"seed{seed}/author/complete.json",
                PRIOR / f"seed{seed}/{name}/projected.npz", FULL / f"seed{seed}/step0/ensembles/{name}/full.npz"]
    paths += [FINE / f"local_geometry/seed{seed}/last.pt" for seed in p["seeds"]]
    workspace = Path.cwd().resolve()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            q = Path(filename).resolve()
            if q.suffix == ".py" and q.is_relative_to(workspace / "upstream"):
                paths.append(q)
    p["source_sha256"] = {str(q): sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT / "protocol.json", p)
    print("DECODER_ANGLE_ADAPTATION_READY", p["trainable_parameters"], json.dumps(checks), flush=True)


@torch.no_grad()
def decode_latents(codec, group):
    backbones=[]; failures=[]; maximum=0.; seconds=[]
    for index,(c,f,origin) in enumerate(zip(group["coarse"],group["fine"],group["origin"])):
        clock=time.perf_counter()
        try:
            value=codec.decode(c,f)+origin
            if not torch.isfinite(value).all():raise FrameGeometryError("Nonfinite decoder output")
            torch.testing.assert_close(pool_ca(value),c+origin,atol=2e-5,rtol=2e-5)
            maximum=max(maximum,float((pool_ca(value)-c-origin).abs().max()))
            backbones.append(value.numpy())
        except FrameGeometryError as error:
            backbones.append(np.full((len(f),3,3),np.nan,np.float32))
            failures.append(dict(sample=index,code="decoder_frame_failure",message=str(error)))
        seconds.append(time.perf_counter()-clock)
    return np.stack(backbones),dict(samples=len(backbones),failure_count=len(failures),failures=failures,
        all_samples_retained=True,maximum_center_roundoff_A=maximum,mean_decode_seconds=float(np.mean(seconds)))


@torch.no_grad()
def evaluate(p,codec,seed,arm,out):
    groups=torch.load(ROOT/"evaluation_inputs.pt",map_location="cpu",weights_only=True)
    selected=[g for g in groups if g["seed"]==seed]; records=[]
    # Persist all outputs before loading the development reference coordinates.
    for group in selected:
        name=group["protein"];outlet=group["outlet"];folder=out/"evaluation"/name;folder.mkdir(parents=True,exist_ok=True)
        bb,audit=decode_latents(codec,group)
        np.savez_compressed(folder/(outlet+".npz"),backbone=bb,valid=np.isfinite(bb).all(axis=(1,2,3)))
        records.append((group,bb,audit))
    atomic_json(out/"prediction_complete.json",dict(status="complete",native_coordinates_loaded=False,
        groups=len(records),generated_samples=sum(len(x[1]) for x in records)))
    dev=torch.load(LABELS/"development_labels.pt",map_location="cpu",weights_only=True);rows=[]
    for group,bb,audit in records:
        name=group["protein"];outlet=group["outlet"];cases=dev["validation"][name]
        reference=np.stack([c["target"].numpy() for c in cases])
        baseline=evaluate_values(group["baseline"].numpy(),reference,dict(samples=64,failure_count=0,failures=[]))
        measured=evaluate_values(bb,reference,audit)
        row=dict(seed=seed,arm=arm,protein=name,outlet=outlet,baseline=baseline,adapted=measured)
        if measured["metrics_available"]:
            row["mean_CA_displacement_from_original_codec_A"]=float(np.mean([
                aligned_rmsd(x[:,1],y[:,1]) for x,y in zip(bb,group["baseline"].numpy())]))
        rows.append(row)
        atomic_json(out/"evaluation_progress.json",dict(phase="evaluation",completed_groups=len(rows),total_groups=4))
        print("DECODER_ADAPTATION_EVALUATED",seed,arm,name,outlet,flush=True)
    # Oracle native roundtrip: no new encoder and no quantization.
    oracle=[]
    for item in p["targets"]:
        name=item["protein"];cases=dev["validation"][name];stats=[]
        for case in cases:
            target=case["target"];c,f=codec.encode(target);value=codec.decode(c,f)
            from compare_two_level_ae_author import metrics
            stats.append(metrics(value.numpy(),target.numpy()))
        oracle.append(dict(protein=name,diagnostic_only=True,metrics={k:float(np.mean([s[k] for s in stats])) for k in stats[0]}))
    atomic_json(out/"evaluation.json",dict(status="complete",rows=rows,native_oracle=oracle,
        test_coordinates_opened=False,R3_opened=False,scope=p["scope"]))


def train(p,seed,arm,resume):
    groups=torch.load(LABELS/"train_labels.pt",map_location="cpu",weights_only=True)
    codec=load_codec(p,"centers3");params=enable_decoder(codec);expected=frozen_digest(codec)
    optimizer=torch.optim.AdamW(params,lr=p["lr"],weight_decay=p["weight_decay"])
    rng=batch_rng(seed);out=ROOT/arm/f"seed{seed}";out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():raise RuntimeError("Final decoder arm already complete")
    start=0;elapsed=0.;trace=[];digest=hashlib.sha256()
    if resume:
        saved=restore_checkpoint(out/"last.pt",codec,optimizer,rng,torch.device("cpu"))
        if saved["protocol_sha256"]!=sha(ROOT/"protocol.json") or saved["seed"]!=seed or saved["arm"]!=arm:raise RuntimeError("Wrong decoder resume")
        start=saved["updates"];elapsed=saved["training_seconds"];restore_trace_prefix(out,start)
        trace=[json.loads(line) for line in (out/"trace.jsonl").read_text().splitlines()]
        for row in trace:digest.update(json.dumps(identity(row),sort_keys=True).encode())
        if digest.hexdigest()!=saved["paired_identity_sha256"]:raise RuntimeError("Trace/checkpoint identity differs")
    elif (out/"last.pt").exists() or (out/"trace.jsonl").exists():raise RuntimeError("Explicit resume required for partial run")
    writer=CompactTrainingTrace(p["updates"],start);clock=time.perf_counter()
    for step in range(start,p["updates"]):
        row=update(codec,optimizer,sample_training_batch(groups,rng),arm,step,p)
        digest.update(json.dumps(identity(row),sort_keys=True).encode());trace.append(row)
        if (step+1)%p["progress_every"]==0 or step+1==p["updates"]:
            writer(out/"trace.json",trace)
            atomic_json(out/"progress.json",dict(phase="training",updates=step+1,total=p["updates"],
                training_seconds=elapsed+time.perf_counter()-clock,loss=row["loss"]))
        if (step+1)%p["checkpoint_every"]==0 or step+1==p["updates"]:
            if frozen_digest(codec)!=expected:raise RuntimeError("Frozen encoder/book/coarse weights changed")
            save_checkpoint(out/"last.pt",codec,optimizer,rng,updates=step+1,seed=seed,arm=arm,
                protocol_sha256=sha(ROOT/"protocol.json"),training_seconds=elapsed+time.perf_counter()-clock,
                paired_identity_sha256=digest.hexdigest())
    seconds=elapsed+time.perf_counter()-clock;codec.eval()
    evaluate(p,codec,seed,arm,out)
    if frozen_digest(codec)!=expected:raise RuntimeError("Frozen components changed during evaluation")
    atomic_json(out/"complete.json",dict(status="complete",seed=seed,arm=arm,updates=p["updates"],
        training_seconds=seconds,paired_identity_sha256=digest.hexdigest(),frozen_encoder_book_coarse_unchanged=True,
        trainable_parameters=sum(v.numel() for v in params),host=platform.node(),codec_sha256=state_digest(codec),scope=p["scope"]))
    atomic_json(out/"progress.json",dict(phase="complete",updates=p["updates"],total=p["updates"]))
    print("DECODER_ANGLE_ADAPTATION_COMPLETE",seed,arm,seconds,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["prepare","train"],required=True)
    parser.add_argument("--seed",type=int,choices=[2026,2027]);parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--resume",action="store_true");args=parser.parse_args();torch.set_num_threads(2)
    if args.stage=="prepare":prepare()
    else:
        p=load(ROOT/"protocol.json");verify(p)
        if args.seed is None or args.arm is None:raise ValueError("Declared seed and arm required")
        train(p,args.seed,args.arm,args.resume)


if __name__=="__main__":main()
