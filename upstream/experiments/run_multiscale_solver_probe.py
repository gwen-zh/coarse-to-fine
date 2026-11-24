"""Read-only checkpoint probe: numerical error versus coarse underdispersion."""
import argparse
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
from multiscale_flow_solver import sample_with_coarse_solver
from native_ensemble_metrics import compare_ensembles, mean_pairwise_rmsd, rmsf
from run_sequence_multiscale_ensemble import (
    RUN as PARENT, frozen_protocol, load_codec, load_labels, make_model,
    sample_seed, write_pdb,
)
from structural_metrics import aligned_rmsd

RUN = Path("artifacts/multiscale_solver_probe_v1")
SOLVERS = {"euler16": ("euler", 16), "heun16": ("heun", 16),
           "euler64": ("euler", 64), "heun64": ("heun", 64)}


def freeze():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "protocol.json").exists():
        raise RuntimeError("Solver protocol already frozen")
    p = frozen_protocol()
    sources = [Path(__file__), Path("upstream/experiments/multiscale_flow_solver.py"),
        Path("upstream/tests/test_multiscale_flow_solver.py"),
        Path("upstream/scripts/multiscale_solver_probe.sbatch"), PARENT / "protocol.json",
        PARENT / "report.json", PARENT / "assessment.json", PARENT / "centers3_labels.pt",
        PARENT / "train_histogram_edges.npz"]
    for seed in p["seeds"]:
        sources.extend(PARENT / f"centers3_teacher/seed{seed}" / name
                       for name in ["last.pt", "generated_latents.npz", "full_continuous.npz"])
    q = dict(seeds=p["seeds"], solvers=SOLVERS, samples=64, fine_nfe=16,
        parent=p, weights_updated=False,
        matching="Same frozen centers3_teacher weights, exactly the previous 64 noise seeds. Fine remains Euler16. Euler16 replay is checked against saved coordinates/latents. Heun uses two velocity calls per interval; NFE16 compares 16 Euler to 8 Heun intervals.",
        scope="Posthoc diagnostic and new solver comparison on the reused single-protein R2 development set; no R3 or new-protein claim. All outputs retained, no sampling-temperature tuning.",
        checks="Record coarse and backbone spread, aligned RMSF, reference coverage, geometry and wall time. Higher spread is not automatically better. Report all solvers and both seeds without automatic adoption.",
        source_sha256={str(path): sha(path) for path in sources})
    atomic_json(RUN / "protocol.json", q)


def protocol():
    p = json.loads((RUN / "protocol.json").read_text())
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed solver input: " + name)
    return p


@torch.no_grad()
def evaluate(p, solver, seed):
    out = RUN / solver / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Solver evaluation already complete")
    q = p["parent"]
    model = make_model(q, "centers3").eval()
    checkpoint = torch.load(PARENT / f"centers3_teacher/seed{seed}/last.pt", map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != sha(PARENT / "protocol.json"):
        raise RuntimeError("Checkpoint protocol mismatch")
    model.load_state_dict(checkpoint["model"])
    codec = load_codec(q, "centers3")
    groups = load_labels("centers3")
    reference = np.stack([r["target"].numpy() for r in groups["MD_development"]])
    reference_centers = np.stack([r["coarse"].numpy() for r in groups["MD_development"]])
    train = groups["MD_train"]
    indices = np.linspace(0, len(train) - 1, p["samples"], dtype=int)
    train_centers = np.stack([train[i]["coarse"].numpy() for i in indices])
    alignment = train[0]["target"][:, 1].numpy()
    with np.load(PARENT / "train_histogram_edges.npz", allow_pickle=False) as z:
        edges, rg = z["distance"].copy(), z["rg"].copy()
    with np.load(PARENT / f"centers3_teacher/seed{seed}/generated_latents.npz", allow_pickle=False) as z:
        previous_c, previous_f = z["coarse"].copy(), z["fine"].copy()
    with np.load(PARENT / f"centers3_teacher/seed{seed}/full_continuous.npz", allow_pickle=False) as z:
        previous_bb = z["backbone"].copy()
    aa = encode_aa(q["sequence"])
    backbones, centers, coarse, fine, times = [], [], [], [], []
    method, nfe = p["solvers"][solver]
    for index in range(p["samples"]):
        start = time.perf_counter()
        c, f = sample_with_coarse_solver(model, codec, aa, sample_seed(seed, index), nfe, method, p["fine_nfe"])
        bb = codec.decode(c, f)
        times.append(time.perf_counter() - start)
        if solver == "euler16":
            for value, old in [(c, previous_c[index]), (f, previous_f[index]), (bb, previous_bb[index])]:
                np.testing.assert_allclose(value.numpy(), old, atol=3e-5, rtol=2e-5)
        backbones.append(bb.numpy()); centers.append(codec.centers(c, len(aa)).numpy())
        coarse.append(c.numpy()); fine.append(f.numpy())
        if (index + 1) % 16 == 0:
            atomic_json(out / "progress.json", dict(samples=index + 1, total=p["samples"]))
    bb, centers = np.stack(backbones), np.stack(centers)
    stats = compare_ensembles(bb[:, :, 1], reference[:, :, 1], edges, rg, alignment)
    geometry = []
    for value in bb:
        nearest = int(np.argmin([aligned_rmsd(value[:, 1], r[:, 1]) for r in reference]))
        geometry.append(metrics(value, reference[nearest]))
    stats["geometry"] = {key: float(np.mean([r[key] for r in geometry])) for key in geometry[0]}
    stats["coarse_spread"] = dict(
        generated_A=mean_pairwise_rmsd(centers), R1_A=mean_pairwise_rmsd(train_centers),
        R2_A=mean_pairwise_rmsd(reference_centers),
        generated_mean_aligned_RMSF_A=float(rmsf(centers, train_centers[0]).mean()),
        R1_mean_aligned_RMSF_A=float(rmsf(train_centers, train_centers[0]).mean()),
        R2_mean_aligned_RMSF_A=float(rmsf(reference_centers, train_centers[0]).mean()))
    stats.update(solver=solver, seed=seed, coarse_NFE=nfe, fine_NFE=p["fine_nfe"],
                 mean_sampling_seconds=float(np.mean(times)), baseline_replay_verified=solver == "euler16")
    np.savez_compressed(out / "samples.npz", backbone=bb, coarse_centers=centers,
                        coarse=np.stack(coarse), fine=np.stack(fine))
    write_pdb(out / "ensemble.pdb", bb, q["sequence"])
    atomic_json(out / "evaluation.json", stats)
    atomic_json(out / "complete.json", dict(status="complete", solver=solver, seed=seed, weights_updated=False))


def report(p):
    rows = [json.loads((RUN / solver / f"seed{seed}/evaluation.json").read_text())
            for solver in p["solvers"] for seed in p["seeds"]]
    atomic_json(RUN / "report.json", dict(status="complete", rows=rows,
        scope=p["scope"], automatic_adoption=False))
    lines = ["Frozen coarse solver probe; all values on reused single-protein development.", ""]
    for r in rows:
        lines.append(f"{r['solver']} seed {r['seed']}: CA {r['mean_sample_nearest_reference_ca_rmsd_A']:.5f}; "
            f"CN {100*r['geometry']['cn_rate']:.4f}%; spread {r['mean_pairwise_ca_rmsd_A']:.5f}; "
            f"coarse spread {r['coarse_spread']['generated_A']:.5f}, R2 {r['coarse_spread']['R2_A']:.5f}; "
            f"JS {r['js_pairwise_distance_divergence_nats']:.5f}; seconds {r['mean_sampling_seconds']:.4f}")
    (RUN / "result_summary.txt").write_text("\n".join(lines) + "\n")
    print("SOLVER_PROBE_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["freeze", "evaluate", "report"], required=True)
    parser.add_argument("--solver", choices=list(SOLVERS))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "freeze":
        freeze()
    else:
        p = protocol()
        if args.stage == "evaluate":
            if args.solver not in p["solvers"] or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared solver or seed")
            evaluate(p, args.solver, args.seed)
        else:
            report(p)
