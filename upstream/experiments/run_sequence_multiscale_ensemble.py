"""Frozen, bounded pilot for AA-conditioned spatial multiscale ensembles.

One native-MD sequence, R1 training and reused R2 development. R3 is not read.
This experiment tests an executable hierarchy, not unseen-sequence performance.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from joint_sequence_predictor import encode_aa
from native_ensemble_metrics import compare_ensembles, distance_features, train_histogram_edges
from sequence_multiscale_ensemble import MultiscaleStructureCodec, paired_flow_loss, sample_levels
from structural_metrics import aligned_rmsd

RUN = Path("artifacts/sequence_multiscale_ensemble_v1")
PARENT = Path("artifacts/block_frame_full_pdb_v1")
EXPERIMENTS = Path("upstream/experiments")
ARMS = {
    "latent16_teacher": dict(coarse_kind="latent16", scheduled_probability=0.),
    "centers3_teacher": dict(coarse_kind="centers3", scheduled_probability=0.),
    "centers3_scheduled": dict(coarse_kind="centers3", scheduled_probability=.5),
}


def load_codec(p, kind):
    return MultiscaleStructureCodec.from_checkpoints(
        **p["tokenizer"], fine_checkpoint=p["fine_checkpoint"], coarse_kind=kind
    )


def frozen_protocol():
    p = json.loads((RUN / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed frozen input: " + name)
    return p


def load_labels(kind):
    manifest = json.loads((RUN / "cache_manifest.json").read_text())
    path = RUN / (kind + "_labels.pt")
    if sha(path) != manifest["cache_sha256"][kind]:
        raise RuntimeError("Changed paired MD labels")
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "protocol.json").exists() or (RUN / "preparation_protocol.json").exists():
        raise RuntimeError("Use a fresh namespace; do not overwrite a frozen preparation")
    parent = json.loads((PARENT / "protocol.json").read_text())
    parent_result = json.loads((PARENT / "report.json").read_text())
    if not parent_result["local_ready_both_seeds"]:
        raise RuntimeError("Local-frame reconstruction readiness failed")
    source_manifest = json.loads((PARENT / "cache_manifest.json").read_text())
    if sha(PARENT / "labels.pt") != source_manifest["cache_sha256"]:
        raise RuntimeError("Changed parent data cache")
    md = json.loads(Path("artifacts/atlas_shared_frame_flow_v1/protocol.json").read_text())
    official = Path("third_party/par-protein")
    revision = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
    official_paths = [official / "README.md", official / "LICENSE",
        official / "proteinfoundation/protein_autoregressive_modeling/par.py",
        official / "proteinfoundation/protein_autoregressive_modeling/protein_decomposer_by_ratio.py"]
    sources = [Path(__file__), Path("upstream/scripts/sequence_multiscale_ensemble.sbatch"),
        Path("upstream/tests/test_sequence_multiscale_ensemble.py"),
        PARENT / "protocol.json", PARENT / "cache_manifest.json", PARENT / "labels.pt",
        PARENT / "report.json", PARENT / "local/seed2026/last.pt",
        Path("artifacts/atlas_shared_frame_flow_v1/protocol.json"),
        *map(Path, parent["tokenizer"].values()), *official_paths]
    module_names = ["sequence_multiscale_ensemble.py", "hierarchical_ensemble_flow.py",
        "block_stable_frame_tokenizer.py", "block_frame_tokenizer.py",
        "block_centroid_transport_frames.py", "block_local_frames.py",
        "hierarchical_latent_contract.py", "hierarchical_rigid_vq.py", "two_level_autoencoder.py",
        "mixed_hierarchical_generator.py", "latent_flow_head.py", "joint_sequence_predictor.py",
        "hierarchical_pipeline_io.py", "native_ensemble_metrics.py", "structural_metrics.py",
        "compare_two_level_ae_author.py", "backbone_quality_audit.py", "diagnostic_io.py"]
    sources.extend(EXPERIMENTS / name for name in module_names)
    atomic_json(RUN / "preparation_protocol.json", dict(
        input_sha256={str(x): sha(x) for x in sources}, PAR_revision=revision,
        scope="Fixed one-sequence MD capacity pilot; no R3 read, filtering, or checkpoint selection.",
        change="New local fine checkpoint; physical centers vs learned coarse; same-example scheduled context; continuous vs quantized outlets.",
    ))
    p = dict(version=1, arms=ARMS, seeds=[2026, 2027], tokenizer=parent["tokenizer"],
        fine_checkpoint=str(PARENT / "local/seed2026/last.pt"), representation_seed=2026,
        sequence=md["sequence"], model=dict(width=128, layers=4), updates=2048, batch_size=4,
        lr=3e-4, min_lr=3e-5, weight_decay=1e-4, gradient_clip=1.,
        samples=64, coarse_steps=16, fine_steps=16, normalization={}, PAR_revision=revision,
        training="R1 only, fresh models, shared AA Transformer and two flow velocity fields. Fine sees the sampled coarse at inference. No CE. No new structure-model training.",
        context="Teacher: exact paired coarse, no jitter. Scheduled: probability .5, endpoint estimate from this example's noisy flow path, detached; re-encode the SAME target backbone in the predicted coarse frames. No independent coarse/fine pairing.",
        matching="Fixed backbone width/depth, seeds, frame draws, optimizer and solver budgets. Raw centers have smaller I/O heads; report actual parameter counts. Teacher/scheduled center arms have identical initial weights and random draws.",
        decoding="Both continuous and nearest-codebook outputs use the EXACT SAME generated coarse/fine vectors. All samples retained. No accept/reject geometry gate.",
        scope="One previously used 90-residue MD protein. R1 train, reused R2 development. Not a new-protein or thermodynamic claim. Existing R3 was consumed in an earlier model's confirmation and is not opened here.",
        stop_rule="Final 2048 updates only; no best checkpoint, hyperparameter sweep, or automatic extension. Evaluate all declared arms and both outlets even if worse.",
        official_code_use="Read-only pinned PAR reference. Our small shared-Transformer implementation is not a pretrained PAR adaptation, full reproduction, or claim of equivalent designability.",
    )
    data = torch.load(PARENT / "labels.pt", map_location="cpu", weights_only=True)
    counts = {}
    cache_hashes = {}
    reconstructions = {}
    for kind in ("latent16", "centers3"):
        codec = load_codec(p, kind)
        groups = {}
        for group in ("MD_train", "MD_development"):
            rows = []
            reconstruction = {"continuous": [], "quantized": []}
            for case in data[group]:
                target = case["target"]
                if len(target) != len(p["sequence"]):
                    raise RuntimeError("MD sequence mapping differs")
                c, f = codec.encode(target)
                rows.append(dict(id=case["id"], sequence=p["sequence"], target=target,
                                 coarse=c, fine_continuous=f))
                if group == "MD_development":
                    for outlet in reconstruction:
                        bb = codec.decode(c, f, quantized=outlet == "quantized")
                        reconstruction[outlet].append(metrics(bb.numpy(), target.numpy()))
            groups[group] = rows
            counts[group] = len(rows)
            if group == "MD_development":
                reconstructions[kind] = {outlet: {
                    key: float(np.mean([r[key] for r in values])) for key in values[0]
                } for outlet, values in reconstruction.items()}
            print("MULTISCALE_LABELS", kind, group, len(rows), flush=True)
        train = groups["MD_train"]
        if len({r["sequence"] for r in train}) != 1:
            raise RuntimeError("This pilot is explicitly single-sequence")
        c = torch.cat([r["coarse"] for r in train])
        f = torch.cat([r["fine_continuous"] for r in train])
        p["normalization"][kind] = {
            key: value.tolist() for key, value in [
                ("coarse_mean", c.mean(0)), ("coarse_std", c.std(0, unbiased=False).clamp_min(.05)),
                ("fine_mean", f.mean(0)), ("fine_std", f.std(0, unbiased=False).clamp_min(.05))]
        }
        path = RUN / (kind + "_labels.pt")
        torch.save(groups, path)
        cache_hashes[kind] = sha(path)
    train_ca = np.stack([r["target"][:, 1].numpy() for r in train])
    distances, rg = distance_features(train_ca)
    bins = RUN / "train_histogram_edges.npz"
    np.savez_compressed(bins, distance=train_histogram_edges(distances), rg=train_histogram_edges(rg))
    atomic_json(RUN / "cache_manifest.json", dict(
        groups=counts, cache_sha256=cache_hashes, normalizations_fit_on="MD_R1_only",
        labels="Fine frames recomputed separately for each coarse convention", R3_loaded=False,
    ))
    atomic_json(RUN / "native_reconstruction.json", reconstructions)
    p["groups"] = counts
    sources.extend([RUN / "preparation_protocol.json", RUN / "cache_manifest.json",
                    RUN / "native_reconstruction.json", bins])
    sources.extend(RUN / (kind + "_labels.pt") for kind in cache_hashes)
    p["source_sha256"] = {str(x): sha(x) for x in sources}
    atomic_json(RUN / "protocol.json", p)
    print("SEQUENCE_MULTISCALE_PROTOCOL_READY", json.dumps(dict(arms=list(ARMS), seeds=p["seeds"], updates=p["updates"], groups=counts)), flush=True)


def make_model(p, kind):
    return HierarchicalEnsembleFlow(**p["normalization"][kind], **p["model"],
        coarse_dim=16 if kind == "latent16" else 3)


def sample_seed(seed, index):
    return int(hashlib.sha256(f"sequence-multiscale-v1:{seed}:{index}".encode()).hexdigest()[:8], 16)


def write_pdb(path, backbones, sequence):
    from atlas_md_io import RESIDUE_AA
    names = {aa: name for name, aa in RESIDUE_AA.items() if name not in {"HID", "HIE", "HIP", "CYX"}}
    with Path(path).open("w") as stream:
        for model, bb in enumerate(backbones, 1):
            stream.write(f"MODEL     {model:4d}\n")
            for i, (residue, atoms) in enumerate(zip(sequence, bb), 1):
                for j, (atom, xyz) in enumerate(zip(("N", "CA", "C"), atoms)):
                    x, y, z = xyz
                    stream.write(f"ATOM  {(i-1)*3+j+1:5d} {atom:^4s} {names[residue]} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2s}\n")
            stream.write("ENDMDL\n")
        stream.write("END\n")


@torch.no_grad()
def evaluate(p, arm, seed, model, codec, groups, out):
    model.eval()
    aa = encode_aa(p["sequence"])
    reference = np.stack([r["target"].numpy() for r in groups["MD_development"]])
    train = groups["MD_train"]
    with np.load(RUN / "train_histogram_edges.npz", allow_pickle=False) as z:
        distance_edges, rg_edges = z["distance"].copy(), z["rg"].copy()
    train_ca = np.stack([r["target"][:, 1].numpy() for r in train])
    point = int(np.argmin(((train_ca - train_ca.mean(0)) ** 2).mean((1, 2))))
    empirical = np.random.default_rng(31337).integers(len(train), size=p["samples"])
    outputs = {name: [] for name in (
        "full_continuous", "full_quantized", "oracle_coarse_continuous",
        "empirical_R1_continuous", "native_R1_point")}
    latency = {"flow_seconds": [], "continuous_decode_seconds": [], "quantized_decode_seconds": []}
    coarse_values, fine_values, seeds = [], [], []
    for index in range(p["samples"]):
        key = sample_seed(seed, index)
        started = time.perf_counter()
        c, f = sample_levels(model, codec, aa, key, p["coarse_steps"], p["fine_steps"])
        latency["flow_seconds"].append(time.perf_counter() - started)
        for quantized, outlet in [(False, "continuous"), (True, "quantized")]:
            started = time.perf_counter()
            outputs["full_" + outlet].append(codec.decode(c, f, quantized).numpy())
            latency[outlet + "_decode_seconds"].append(time.perf_counter() - started)
        native = groups["MD_development"][index % len(reference)]
        oracle_f = model.sample_fine(aa, native["coarse"], key, p["fine_steps"])
        outputs["oracle_coarse_continuous"].append(codec.decode(native["coarse"], oracle_f).numpy())
        replay = train[int(empirical[index])]
        outputs["empirical_R1_continuous"].append(codec.decode(replay["coarse"], replay["fine_continuous"]).numpy())
        outputs["native_R1_point"].append(train[point]["target"].numpy())
        coarse_values.append(c.numpy()); fine_values.append(f.numpy()); seeds.append(key)
        if (index + 1) % 16 == 0:
            atomic_json(out / "evaluation_progress.json", dict(samples=index + 1, total=p["samples"]))
    np.savez_compressed(out / "generated_latents.npz", coarse=np.stack(coarse_values), fine=np.stack(fine_values), sample_seeds=np.array(seeds, dtype=np.uint32))
    rows = []
    for name, values in outputs.items():
        bb = np.stack(values)
        stats = compare_ensembles(bb[:, :, 1], reference[:, :, 1], distance_edges, rg_edges, train_ca[0])
        geometry = []
        for value in bb:
            nearest = int(np.argmin([aligned_rmsd(value[:, 1], r[:, 1]) for r in reference]))
            geometry.append(metrics(value, reference[nearest]))
        stats["geometry"] = {key: float(np.mean([r[key] for r in geometry])) for key in geometry[0]}
        rows.append(dict(outlet=name, **stats))
        np.savez_compressed(out / (name + ".npz"), backbone=bb)
        if name.startswith("full_"):
            write_pdb(out / (name + ".pdb"), bb, p["sequence"])
    result = dict(status="complete", arm=arm, seed=seed, outlets=rows,
        mean_latency_seconds={key: float(np.mean(value)) for key, value in latency.items()},
        samples=p["samples"], all_samples_retained=True, R3_loaded=False,
        oracle_scope="Oracle coarse uses R2 labels for diagnostics only; full sampling uses AA and noise only.",
        controls_scope="Empirical R1 and native point are one-protein diagnostics, not deployable unseen-protein methods.",
        scope=p["scope"])
    atomic_json(out / "evaluation.json", result)
    return result


def train(p, arm, seed):
    setting = p["arms"][arm]
    kind = setting["coarse_kind"]
    groups = load_labels(kind)
    codec = load_codec(p, kind)
    frozen = copy.deepcopy(codec.state_dict())
    torch.manual_seed(seed)
    model = make_model(p, kind)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed)
    flow_rng = torch.Generator().manual_seed(seed + 22017)
    out = RUN / arm / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Fixed pilot already complete")
    start, trace = 0, []
    digest = sha(RUN / "protocol.json")
    if (out / "last.pt").exists():
        saved = restore_checkpoint(out / "last.pt", model, opt, rng, torch.device("cpu"))
        if saved["protocol_sha256"] != digest:
            raise RuntimeError("Resume protocol differs")
        start = saved["updates"]
        flow_rng.set_state(saved["flow_rng_state"])
        trace = json.loads((out / "trace.json").read_text())[:start]
    started = time.perf_counter()
    aa = encode_aa(p["sequence"])
    for step in range(start, p["updates"]):
        model.train()
        selected = [groups["MD_train"][int(i)] for i in rng.integers(len(groups["MD_train"]), size=p["batch_size"])]
        rate = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / max(1, p["updates"] - 1)))
        for group in opt.param_groups:
            group["lr"] = rate
        opt.zero_grad(set_to_none=True)
        total, terms = 0., []
        for case in selected:
            loss, record = paired_flow_loss(model, codec, aa, case["target"], case["coarse"], case["fine_continuous"], flow_rng, setting["scheduled_probability"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multiscale flow loss")
            (loss / len(selected)).backward()
            total += float(loss.detach()) / len(selected)
            terms.append(record)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite multiscale flow gradient")
        opt.step()
        trace.append(dict(update=step + 1, batch=[c["id"] for c in selected], loss=total,
            coarse_flow=float(np.mean([r["coarse_flow"] for r in terms])),
            fine_flow=float(np.mean([r["fine_flow"] for r in terms])),
            scheduled_fraction=float(np.mean([r["scheduled"] for r in terms])),
            draws=[r["draw_sha256"] for r in terms]))
        if (step + 1) % 64 == 0 or step + 1 == p["updates"]:
            progress = dict(arm=arm, seed=seed, updates=step + 1, total=p["updates"], seconds=time.perf_counter() - started, loss=total)
            atomic_json(out / "progress.json", progress)
            atomic_json(out / "trace.json", trace)
            print("MULTISCALE_TRAIN", json.dumps(progress), flush=True)
        if (step + 1) % 512 == 0 or step + 1 == p["updates"]:
            save_checkpoint(out / "last.pt", model, opt, rng, updates=step + 1,
                model_config=model.config, arm=arm, seed=seed, protocol_sha256=digest,
                flow_rng_state=flow_rng.get_state())
    evaluate(p, arm, seed, model, codec, groups, out)
    for key, value in frozen.items():
        if not torch.equal(value, codec.state_dict()[key]):
            raise RuntimeError("Frozen structure weights changed: " + key)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed,
        updates=p["updates"], seconds=time.perf_counter() - started,
        parameters=sum(x.numel() for x in model.parameters()), frozen_structure_verified=True))


def report(p):
    rows = []
    for arm in p["arms"]:
        for seed in p["seeds"]:
            out = RUN / arm / f"seed{seed}"
            complete = json.loads((out / "complete.json").read_text())
            result = json.loads((out / "evaluation.json").read_text())
            rows.append(dict(arm=arm, seed=seed, complete=complete, evaluation=result))
    for seed in p["seeds"]:
        traces = [json.loads((RUN / arm / f"seed{seed}/trace.json").read_text()) for arm in p["arms"]]
        if any([r["batch"] for r in trace] != [r["batch"] for r in traces[0]] for trace in traces[1:]):
            raise RuntimeError("Minibatches were not matched")
        if [r["draws"] for r in traces[1]] != [r["draws"] for r in traces[2]]:
            raise RuntimeError("Center teacher/scheduled noise draws were not matched")
    atomic_json(RUN / "report.json", dict(status="complete", rows=rows,
        minibatches_matched=True, center_noise_draws_matched=True, R3_loaded=False,
        scope=p["scope"], no_automatic_model_replacement=True))
    print("MULTISCALE_REPORT_COMPLETE", len(rows), flush=True)


@torch.no_grad()
def generate(args):
    """Inference reads model metadata/weights, AA, and seeds; no label cache."""
    p = json.loads((RUN / "protocol.json").read_text())
    if args.arm not in p["arms"] or args.seed not in p["seeds"]:
        raise RuntimeError("Undeclared model checkpoint")
    if not args.sequence or set(args.sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("A standard amino-acid sequence is required")
    if len(args.sequence) < 9 or args.samples < 1:
        raise ValueError("At least three coarse blocks and one sample are required")
    kind = p["arms"][args.arm]["coarse_kind"]
    for name in [*p["tokenizer"].values(), p["fine_checkpoint"]]:
        if sha(name) != p["source_sha256"][name]:
            raise RuntimeError("Changed frozen inference checkpoint")
    checkpoint = RUN / args.arm / f"seed{args.seed}/last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["protocol_sha256"] != sha(RUN / "protocol.json") or saved["updates"] != p["updates"]:
        raise RuntimeError("Generator is not the frozen final endpoint")
    model = make_model(p, kind).eval()
    model.load_state_dict(saved["model"])
    codec = load_codec(p, kind)
    aa = encode_aa(args.sequence)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "request.json", dict(sequence=args.sequence, arm=args.arm,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        generator_training_seed=args.seed, sampling_seed=args.sample_seed,
        samples=args.samples, coarse_steps=p["coarse_steps"], fine_steps=p["fine_steps"],
        native_coordinates_read=False,
        scope="This pilot was trained on one MD protein; new sequences are unsupported extrapolations, not validated predictions."))
    backbones, centers = [], []
    for index in range(args.samples):
        c, f = sample_levels(model, codec, aa, sample_seed(args.sample_seed, index),
                             p["coarse_steps"], p["fine_steps"])
        backbones.append(codec.decode(c, f).numpy())
        centers.append(codec.centers(c, len(aa)).numpy())
    np.savez_compressed(out / "ensemble.npz", backbone=np.stack(backbones), coarse_centers=np.stack(centers))
    write_pdb(out / "ensemble.pdb", backbones, args.sequence)
    atomic_json(out / "complete.json", dict(status="complete", samples=len(backbones), all_samples_retained=True))
    print("AA_ONLY_ENSEMBLE_GENERATED", out, len(backbones), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "report", "generate"], required=True)
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sequence")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=1717)
    parser.add_argument("--output", default="artifacts/sequence_multiscale_generated_v1")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "generate":
        generate(args)
    else:
        protocol = frozen_protocol()
        if args.stage == "train":
            if args.arm not in protocol["arms"] or args.seed not in protocol["seeds"]:
                raise RuntimeError("Undeclared arm or seed")
            train(protocol, args.arm, args.seed)
        else:
            report(protocol)
