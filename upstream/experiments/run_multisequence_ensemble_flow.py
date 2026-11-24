"""Five-sequence native-MD capacity trial plus two new-sequence development cases.

Both arms start fresh from the same generator seeds. The intervention is one
PDB replay slot per four examples. Test protein coordinates remain unopened.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import mdtraj as md
import numpy as np
import torch

from atlas_md_io import backbone_indices
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import (
    align_trajectory, canonical_training_reference, balanced_batch,
    fit_train_normalization, physical_histogram_edges, slot_seed,
)
from native_ensemble_metrics import compare_ensembles
from run_sequence_multiscale_ensemble import load_codec, write_pdb, frozen_protocol as previous_protocol
from sequence_multiscale_ensemble import paired_flow_loss, sample_levels
from structural_metrics import aligned_rmsd

RUN = Path("artifacts/multisequence_ensemble_flow_v1")
DATA = Path("artifacts/multisequence_atlas_v1")
PARENT = Path("artifacts/sequence_multiscale_ensemble_v1")
ARMS = {"MD_only": False, "MD_plus_PDB": True}


def protocol():
    p = json.loads((RUN / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed multisequence input: " + name)
    return p


def label_case(codec, backbone, sequence, name, protein, split):
    c, f = codec.encode(backbone)
    return dict(id=name, protein=protein, sequence=sequence, target=backbone,
                coarse=c, fine_continuous=f, split=split)


@torch.no_grad()
def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "protocol.json").exists():
        raise RuntimeError("Multi-sequence protocol already frozen")
    parent = previous_protocol()
    selection = json.loads((DATA / "selection.json").read_text())
    download = json.loads((DATA / "report.json").read_text())
    if download["status"] != "download_complete":
        raise RuntimeError("MD downloads are not complete")
    for name, digest in selection["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed cohort-selection input: " + name)
    codec = load_codec(parent, "centers3")
    groups = {"train": {}, "validation": {}, "reused_diagnostic": {}, "PDB_train": []}
    frame_records = []
    reconstruction = []
    for entry in download["proteins"]:
        role = "train" if entry["official_split"] == "train" else "validation"
        if entry["selection_sha256"] != sha(DATA / "selection.json"):
            raise RuntimeError("MD cohort provenance differs")
        for row in entry["extracted"]:
            if sha(row["path"]) != row["sha256"]:
                raise RuntimeError("MD input changed: " + row["path"])
        topology = md.load_topology(entry["topology"])
        indices = backbone_indices(topology, entry["sequence"])
        reference = None
        cases = []
        for trajectory in sorted(entry["trajectories"]):
            if "_R3_" in trajectory:
                raise RuntimeError("R3 is not eligible for this trial")
            replicate = "R1" if "_R1_" in trajectory else "R2"
            # Decompress only N/CA/C atoms, with stride fixed before any metrics.
            traj = md.load_xtc(trajectory, top=topology, atom_indices=indices.flatten(), stride=40)
            backbones = traj.xyz.reshape(len(traj), len(entry["sequence"]), 3, 3) * 10.
            if reference is None:
                reference = canonical_training_reference(backbones[0])
            if not np.isfinite(backbones).all():
                raise RuntimeError("Nonfinite native MD coordinates")
            aligned = align_trajectory(backbones, reference)
            if role == "validation":
                take = np.linspace(0, len(aligned) - 1, 64, dtype=int)
            else:
                take = np.arange(len(aligned))
            for index in take:
                target = aligned[int(index)]
                case = label_case(codec, target, entry["sequence"],
                    f"{entry['name']}:{replicate}:{int(index)*40}", entry["name"], role)
                cases.append(case)
            frame_records.append(dict(protein=entry["name"], split=role, replicate=replicate,
                stride=40, selected_original_indices=(take*40).tolist(), frames=len(take)))
        groups[role][entry["name"]] = cases
        probe = np.linspace(0, len(cases)-1, 16, dtype=int)
        values = [metrics(codec.decode(cases[i]["coarse"], cases[i]["fine_continuous"]).numpy(),
                          cases[i]["target"].numpy()) for i in probe]
        summary = {key: float(np.mean([v[key] for v in values])) for key in values[0]}
        reconstruction.append(dict(protein=entry["name"], split=role, frames=16, metrics=summary))
        print("MULTI_MD_LABELS", entry["name"], role, len(cases), summary["ca_rmsd"], flush=True)
    # Preserve the old MD protein's existing roles: R2 is not silently added to train.
    old = torch.load(PARENT / "centers3_labels.pt", map_location="cpu", weights_only=True)
    old_train_indices = np.linspace(0, len(old["MD_train"]) - 1, 256, dtype=int)
    groups["train"]["1b2s_E"] = [dict(old["MD_train"][int(i)], protein="1b2s_E", split="train") for i in old_train_indices]
    groups["reused_diagnostic"]["1b2s_E"] = [dict(r, protein="1b2s_E", split="reused_diagnostic") for r in old["MD_development"]]
    manifest_path = Path("artifacts/pdb_training_v2/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    pdb_sources = []
    for entry in manifest["targets"]:
        if entry["subset"] != "train":
            continue
        path = manifest_path.parent / entry["backbone"]
        with np.load(path, allow_pickle=False) as z:
            target = canonical_training_reference(z["backbone"])
        case = label_case(codec, target, entry["sequence"], "PDB:" + entry["id"], entry["id"], "train")
        groups["PDB_train"].append(case)
        pdb_sources.append(path)
    if len(groups["train"]) != 5 or len(groups["validation"]) != 2:
        raise RuntimeError("Unexpected cohort size")
    for split in ["train", "validation"]:
        for cases in groups[split].values():
            if len({c["sequence"] for c in cases}) != 1:
                raise RuntimeError("Mixed sequence labels within a protein")
    # Fixed representation-readiness limits; failing cases are never dropped.
    ready = all(r["metrics"]["ca_rmsd"] <= 1. and r["metrics"]["cn_rate"] <= .10
                and r["metrics"]["boundary_cn_rate"] <= .25 for r in reconstruction)
    atomic_json(RUN / "representation_preflight.json", dict(rows=reconstruction, ready=ready,
        limits=dict(CA_max_A=1., CN_max=.10, boundary_CN_max=.25), rows_removed=0,
        note="Native reconstruction readiness on development data; not a generator result or independent test."))
    if not ready:
        raise RuntimeError("Frozen representation fails a declared native reconstruction gate")
    torch.save(groups, RUN / "labels.pt")
    sources = [Path(__file__), DATA / "selection.json", DATA / "report.json",
        PARENT / "protocol.json", PARENT / "centers3_labels.pt", manifest_path,
        RUN / "representation_preflight.json", RUN / "labels.pt",
        Path("upstream/scripts/multisequence_ensemble_flow.sbatch"),
        Path("upstream/tests/test_multisequence_flow_data.py"),
        *map(Path, parent["tokenizer"].values()), Path(parent["fine_checkpoint"]), *pdb_sources]
    names = ["multisequence_flow_data.py", "sequence_multiscale_ensemble.py",
        "run_sequence_multiscale_ensemble.py", "hierarchical_ensemble_flow.py",
        "atlas_md_io.py", "atlas_shared_frame.py", "hierarchical_rigid_vq.py",
        "hierarchical_latent_contract.py", "block_stable_frame_tokenizer.py",
        "block_centroid_transport_frames.py", "block_frame_tokenizer.py", "block_local_frames.py",
        "mixed_hierarchical_generator.py", "two_level_autoencoder.py", "joint_sequence_predictor.py",
        "hierarchical_pipeline_io.py", "native_ensemble_metrics.py", "structural_metrics.py",
        "compare_two_level_ae_author.py", "diagnostic_io.py"]
    sources.extend(Path("upstream/experiments") / name for name in names)
    counts = {split: {key: len(value) for key, value in groups[split].items()}
              for split in ["train", "validation", "reused_diagnostic"]}
    p = dict(version=1, arms=ARMS, seeds=[2026, 2027], updates=4096,
        model=dict(width=128, layers=4), tokenizer=parent["tokenizer"], fine_checkpoint=parent["fine_checkpoint"],
        normalization=fit_train_normalization(groups["train"]), groups=counts, PDB_train=len(groups["PDB_train"]),
        lr=3e-4, min_lr=3e-5, weight_decay=1e-4, gradient_clip=1., samples=64,
        old_R1_uniform_cache_indices=old_train_indices.tolist(),
        train_diagnostic_samples=16, coarse_nfe=16, fine_nfe=16, frame_records=frame_records,
        coordinate_convention="Each new protein has a fixed first-R1 reference, canonicalized once. All its R1/R2 labels align to this fixed reference. No reference is an inference input. The old protein keeps its old frame and split.",
        training="Uniform draw over five training proteins, then uniform MD frame. MD_only has four MD slots; MD_plus_PDB replaces slot four with a uniform static PDB training example. Slots 1-3 and their independently seeded noise/time draws match between arms. Both generators start fresh.",
        matching="Same structure E/D, generator architecture, training seeds and optimizer update budget. Static replay has different sequence lengths and one fewer MD draw; report measured time and label domains. Both arms share train-MD-only normalization with equal weight per protein.",
        metrics="Fixed physical distance bins 0..64 Å in 2 Å intervals plus overflow; Rg 0..32 Å in 1 Å intervals plus overflow. New-protein references do not fit bins. These JS values cannot be compared directly with earlier per-protein adaptive-bin experiments.",
        primary="Macro-average full AA-only generation on two new validation proteins; all 64 raw samples retained per model/protein. Reference consists of 64 fixed frames from each of R1 and R2. Two validation proteins are exploratory evidence, not a broad generalization claim.",
        diagnostics="Seen training sequences and old R2 are reused diagnostics. Native reconstruction/oracle coarse are not deployment results. Scrambled-AA comparison is OOD sensitivity only, not evidence that a sequence has that fold.",
        stop_rule="Final 4096 updates only; no best epoch or automatic extension, temperature selection, sample rejection or test coordinate access. Neither new test protein nor any new R3 is opened. No calibrated thermodynamic/kinetic claims.",
        source_sha256={str(path): sha(path) for path in sources})
    atomic_json(RUN / "protocol.json", p)
    print("MULTISEQUENCE_PROTOCOL_READY", json.dumps(dict(groups=counts, updates=p["updates"], PDB=p["PDB_train"])), flush=True)


def make_model(p):
    return HierarchicalEnsembleFlow(**p["normalization"], **p["model"], coarse_dim=3)


def sample_seed(seed, protein, index):
    return int(hashlib.sha256(f"multi-flow-v1:{seed}:{protein}:{index}".encode()).hexdigest()[:8], 16)


@torch.no_grad()
def evaluate(p, model, codec, groups, out, seed):
    model.eval()
    rows = []
    for split in ["validation", "reused_diagnostic", "train"]:
        for name, cases in groups[split].items():
            take = np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)
            refs = [cases[int(i)] for i in take]
            reference = np.stack([r["target"].numpy() for r in refs])
            aa = encode_aa(cases[0]["sequence"])
            shuffled = aa[torch.randperm(len(aa), generator=torch.Generator().manual_seed(719))]
            count = p["samples"] if split == "validation" else p["train_diagnostic_samples"]
            oracle_indices = np.linspace(0, len(refs)-1, count, dtype=int)
            names = ["full"] + (["oracle_coarse", "scrambled_AA"] if split == "validation" else [])
            backbones = {key: [] for key in names}
            latencies = []
            for index in range(count):
                key = sample_seed(seed, name, index)
                started = time.perf_counter()
                c, f = sample_levels(model, codec, aa, key, p["coarse_nfe"], p["fine_nfe"])
                backbones["full"].append(codec.decode(c, f).numpy())
                latencies.append(time.perf_counter() - started)
                if split == "validation":
                    native = refs[int(oracle_indices[index])]
                    of = model.sample_fine(aa, native["coarse"], key, p["fine_nfe"])
                    backbones["oracle_coarse"].append(codec.decode(native["coarse"], of).numpy())
                    sc, sf = sample_levels(model, codec, shuffled, key, p["coarse_nfe"], p["fine_nfe"])
                    backbones["scrambled_AA"].append(codec.decode(sc, sf).numpy())
            edges, rg = physical_histogram_edges(len(aa))
            for outlet, values in backbones.items():
                bb = np.stack(values)
                stats = compare_ensembles(bb[:, :, 1], reference[:, :, 1], edges, rg, reference[0, :, 1])
                geometry = []
                for value in bb:
                    nearest = int(np.argmin([aligned_rmsd(value[:, 1], r[:, 1]) for r in reference]))
                    geometry.append(metrics(value, reference[nearest]))
                stats["geometry"] = {key: float(np.mean([r[key] for r in geometry])) for key in geometry[0]}
                stats.update(protein=name, split=split, outlet=outlet,
                    mean_full_generation_seconds=float(np.mean(latencies)))
                rows.append(stats)
                folder = out / name
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / (outlet + ".npz"), backbone=bb)
                if outlet == "full" and split == "validation":
                    write_pdb(folder / "ensemble.pdb", bb, cases[0]["sequence"])
            atomic_json(out / "evaluation_progress.json", dict(protein=name, split=split, completed_groups=len(rows)))
    atomic_json(out / "evaluation.json", dict(status="complete", rows=rows,
        test_coordinates_opened=False, R3_opened=False, scope=p["primary"]))


def train(p, arm, seed):
    groups = torch.load(RUN / "labels.pt", map_location="cpu", weights_only=True)
    codec = load_codec(p, "centers3")
    torch.manual_seed(seed)
    model = make_model(p)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed)
    out = RUN / arm / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Multi-sequence arm already complete")
    start, trace = 0, []
    digest = sha(RUN / "protocol.json")
    if (out / "last.pt").exists():
        saved = restore_checkpoint(out / "last.pt", model, opt, rng, torch.device("cpu"))
        if saved["protocol_sha256"] != digest:
            raise RuntimeError("Resume protocol mismatch")
        start = saved["updates"]
        trace = json.loads((out / "trace.json").read_text())[:start]
    started = time.perf_counter()
    for step in range(start, p["updates"]):
        model.train()
        batch = balanced_batch(groups["train"], groups["PDB_train"], rng, p["arms"][arm])
        rate = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / max(1, p["updates"] - 1)))
        for group in opt.param_groups:
            group["lr"] = rate
        opt.zero_grad(set_to_none=True)
        value, draws = 0., []
        for slot, case in enumerate(batch):
            gen = torch.Generator().manual_seed(slot_seed(seed, step, slot))
            loss, terms = paired_flow_loss(model, codec, encode_aa(case["sequence"]),
                case["target"], case["coarse"], case["fine_continuous"], gen, 0.)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multisequence loss")
            (loss / 4).backward()
            value += float(loss.detach()) / 4
            draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite multisequence gradient")
        opt.step()
        trace.append(dict(update=step + 1, loss=value, batch=[c["id"] for c in batch], draws=draws))
        if (step + 1) % 64 == 0 or step + 1 == p["updates"]:
            progress = dict(arm=arm, seed=seed, updates=step + 1, total=p["updates"],
                            seconds=time.perf_counter() - started, loss=value)
            atomic_json(out / "progress.json", progress)
            atomic_json(out / "trace.json", trace)
            print("MULTI_FLOW_TRAIN", json.dumps(progress), flush=True)
        if (step + 1) % 512 == 0 or step + 1 == p["updates"]:
            save_checkpoint(out / "last.pt", model, opt, rng, updates=step + 1,
                model_config=model.config, protocol_sha256=digest, arm=arm, seed=seed)
    evaluate(p, model, codec, groups, out, seed)
    # Checkpoint hashes in protocol() also protect every frozen structure weight.
    protocol()
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed,
        updates=p["updates"], seconds=time.perf_counter() - started,
        parameters=sum(x.numel() for x in model.parameters()), frozen_representation=True))


def report(p):
    rows = []
    for arm in p["arms"]:
        for seed in p["seeds"]:
            out = RUN / arm / f"seed{seed}"
            complete = json.loads((out / "complete.json").read_text())
            data = json.loads((out / "evaluation.json").read_text())
            rows.append(dict(arm=arm, seed=seed, complete=complete, evaluation=data))
    for seed in p["seeds"]:
        a, b = [json.loads((RUN / arm / f"seed{seed}/trace.json").read_text()) for arm in p["arms"]]
        if [r["batch"][:3] for r in a] != [r["batch"][:3] for r in b]:
            raise RuntimeError("Shared MD slots differ")
        if [r["draws"][:3] for r in a] != [r["draws"][:3] for r in b]:
            raise RuntimeError("Shared MD noise/time draws differ")
    atomic_json(RUN / "report.json", dict(status="complete", rows=rows,
        common_MD_draws_verified=True, test_coordinates_opened=False, R3_opened=False,
        scope=p["primary"], automatic_adoption=False))
    print("MULTISEQUENCE_FLOW_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "report"])
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = protocol()
        if args.stage == "train":
            if args.arm not in p["arms"] or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared multisequence arm")
            train(p, args.arm, args.seed)
        else:
            report(p)
