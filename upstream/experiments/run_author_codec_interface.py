"""Read-only projection of cached AA-only author samples through our own codec.

No training, search, quantization or reference-guided selection. Projection is
saved before native development coordinates are loaded for evaluation.
"""
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
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from multisequence_flow_data import physical_histogram_edges
from native_ensemble_metrics import compare_ensembles
from run_sequence_multiscale_ensemble import load_codec
from structural_metrics import aligned_rmsd


ROOT = Path("artifacts/author_codec_interface_v1")
PREVIOUS = Path("artifacts/coarse_coverage_repair_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1/development_labels.pt")
METRIC_KEYS = ["mean_sample_nearest_reference_ca_rmsd_A",
    "mean_reference_nearest_sample_ca_rmsd_A", "js_pairwise_distance_divergence_nats",
    "mean_pairwise_ca_rmsd_A", "rmsf_mae_A"]
GEOMETRY_KEYS = ["cn_rate", "boundary_cn_rate", "ca_clashes_per_residue",
    "reference_ca_c_n_angle_mae_degrees", "reference_c_n_ca_angle_mae_degrees"]


def load(path):
    return json.loads(Path(path).read_text())


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def project_one(codec, backbone):
    x = torch.as_tensor(backbone, dtype=torch.float32)
    if x.ndim != 3 or x.shape[1:] != (3, 3) or not torch.isfinite(x).all():
        raise ValueError("Finite unpadded N/CA/C input required")
    origin = x[:, 1].mean(0)
    coarse, fine = codec.encode(x)
    if coarse.shape != ((len(x) + 3) // 4, 3) or fine.shape != (len(x), 8):
        raise ValueError("Unexpected continuous codec interface")
    # centers3 subtracts global CA translation; restore it after decoding.
    decoded = codec.decode(coarse, fine, quantized=False) + origin
    if not torch.isfinite(decoded).all():
        raise FrameGeometryError("Nonfinite codec output")
    drift = float((pool_ca(decoded) - pool_ca(x)).abs().max())
    torch.testing.assert_close(pool_ca(decoded), pool_ca(x), atol=2e-5, rtol=2e-5)
    return decoded.numpy(), coarse.numpy(), fine.numpy(), drift


@torch.no_grad()
def project_all(codec, backbone):
    values = np.asarray(backbone)
    if values.ndim != 4 or values.shape[2:] != (3, 3) or not len(values):
        raise ValueError("Nonempty sample x residue x atom x xyz array required")
    outputs = []; coarse = []; fine = []; failures = []; seconds = []; drifts = []
    for index, value in enumerate(values):
        started = time.perf_counter()
        try:
            y, c, f, drift = project_one(codec, value)
        except FrameGeometryError as error:
            y = np.full(value.shape, np.nan, np.float32)
            c = np.full(((len(value) + 3) // 4, 3), np.nan, np.float32)
            f = np.full((len(value), 8), np.nan, np.float32); drift = None
            failures.append(dict(sample=index, code="codec_frame_failure", message=str(error)))
        outputs.append(y); coarse.append(c); fine.append(f)
        seconds.append(time.perf_counter() - started)
        if drift is not None: drifts.append(drift)
    result = dict(backbone=np.stack(outputs), coarse=np.stack(coarse), fine=np.stack(fine),
        valid=np.isfinite(np.stack(outputs)).all(axis=(1, 2, 3)))
    audit = dict(samples=len(values), failure_count=len(failures), failures=failures,
        all_samples_retained=True, mean_projection_seconds=float(np.mean(seconds)),
        maximum_center_roundoff_A=max(drifts) if drifts else None)
    return result, audit


def evaluate_values(backbone, reference, audit):
    result = dict(metrics_available=audit["failure_count"] == 0, **audit)
    if not result["metrics_available"]:
        return result
    edges, rg = physical_histogram_edges(backbone.shape[1])
    result.update(compare_ensembles(backbone[:, :, 1], reference[:, :, 1], edges, rg, reference[0, :, 1]))
    rows = []
    for value in backbone:
        nearest = int(np.argmin([aligned_rmsd(value[:, 1], x[:, 1]) for x in reference]))
        rows.append(metrics(value, reference[nearest]))
    result["geometry"] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh interface namespace")
    old = load(PREVIOUS / "protocol.json")
    p = dict(seeds=[2026, 2027], samples=64, coarse_steps=25,
        targets=old["author"]["targets"], tokenizer=old["tokenizer"], fine_checkpoint=old["fine_checkpoint"],
        reference_labels=str(LABELS), threads=2, optimizer_updates=0,
        interface="Cached original25-step AA-only generated N/CA/C -> frozen own centers3 + continuous fine8 encoder -> frozen own decoder. No VQ/snap. Restore input global CA translation; preserve all block CA centers.",
        scope="Two repeatedly reused development sequences,64 cached raw author samples per sequence and seed. Different pretraining/capacity from our standalone generator. No independent test, R3, training, sample selection or model promotion.",
        evaluation="Save all256 codec-projected generated structures before loading native coordinates. Evaluate raw and projected outputs against the same128 references. Reference->codec roundtrips are separately labelled oracle reconstruction diagnostics, never generated samples.",
        decision="Assess whether the frozen representation/decoder can preserve a pretrained AA-only structural prior. This is an interface pilot, not a new generator or proof of thermodynamic ensemble calibration.",
        failure_policy="Retain raw NaN placeholders for frame failures. Any failure makes that group metrics unavailable. No success-only subset, fallback or candidate selection.")
    codec = load_codec(p, "centers3"); original_state = state_digest(codec); checks = []
    paths = [Path(__file__), Path("upstream/tests/test_author_codec_interface.py"),
        Path("upstream/scripts/author_codec_interface.sbatch"), PREVIOUS / "protocol.json", LABELS,
        Path(p["fine_checkpoint"]), *[Path(v) for v in p["tokenizer"].values()]]
    for seed in p["seeds"]:
        directory = PREVIOUS / f"seed{seed}/author"
        prior = load(directory / "complete.json")
        if prior["status"] != "complete" or not prior["all_samples_retained"]:
            raise RuntimeError("Incomplete cached author comparison")
        paths.append(directory / "complete.json")
        for item in p["targets"]:
            path = directory / f"{item['protein']}_steps25_backbone.npz"; paths.append(path)
            with np.load(path) as z:
                x = z["backbone"].copy(); valid = z["valid"].copy()
            if x.shape != (64, item["length"], 3, 3) or not valid.all() or not np.isfinite(x).all():
                raise RuntimeError("Unexpected source cohort")
            _, c, f, drift = project_one(codec, x[0])
            checks.append(dict(seed=seed, protein=item["protein"], coarse_shape=list(c.shape),
                fine_shape=list(f.shape), maximum_center_roundoff_A=drift))
    if state_digest(codec) != original_state or any(v.requires_grad or v.grad is not None for v in codec.parameters()):
        raise RuntimeError("Codec unexpectedly changed")
    atomic_json(ROOT / "integration_check.json", dict(status="passed", checks=checks,
        codec_sha256=original_state, native_coordinates_loaded=False, optimizer_updates=0))
    paths.append(ROOT / "integration_check.json")
    workspace = Path.cwd().resolve()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            q = Path(name).resolve()
            if q.suffix == ".py" and q.is_relative_to(workspace / "upstream"):
                paths.append(q)
    p["source_sha256"] = {str(q): sha(q) for q in dict.fromkeys(paths)}
    atomic_json(ROOT / "protocol.json", p)
    print("AUTHOR_CODEC_INTERFACE_READY", len(checks), flush=True)


def verify(p):
    for path, digest in p["source_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("Frozen interface input changed: " + path)


@torch.no_grad()
def run(p):
    if (ROOT / "complete.json").exists():
        raise RuntimeError("Interface evaluation already complete")
    codec = load_codec(p, "centers3"); initial = state_digest(codec); records = []
    # Deployment projection stage: deliberately no label loading here.
    for seed in p["seeds"]:
        for item in p["targets"]:
            name = item["protein"]; out = ROOT / f"seed{seed}" / name; out.mkdir(parents=True, exist_ok=True)
            with np.load(PREVIOUS / f"seed{seed}/author/{name}_steps25_backbone.npz") as z:
                values = z["backbone"].copy()
            projected, audit = project_all(codec, values)
            np.savez_compressed(out / "projected.npz", **projected)
            atomic_json(out / "projection.json", audit)
            records.append((seed, name, values, projected["backbone"], audit))
            atomic_json(ROOT / "progress.json", dict(phase="projection", completed_groups=len(records), total_groups=4))
            print("AUTHOR_CODEC_PROJECTED", seed, name, audit["failure_count"], flush=True)
    atomic_json(ROOT / "projection_complete.json", dict(status="complete", groups=len(records),
        generated_samples=sum(len(v[2]) for v in records), native_coordinates_loaded=False,
        codec_sha256=state_digest(codec)))
    # Native structures enter only after every projected sample has been saved.
    dev = torch.load(p["reference_labels"], map_location="cpu", weights_only=True)
    rows = []; oracle_rows = []
    for item in p["targets"]:
        name = item["protein"]; cases = dev["validation"][name]
        if len(cases) != 128 or cases[0]["sequence"] != item["sequence"]:
            raise RuntimeError("Native reference identity/count mismatch")
        reference = np.stack([c["target"].numpy() for c in cases])
        for seed, protein, raw, projected, audit in records:
            if protein != name:
                continue
            cached_rows = load(PREVIOUS / f"seed{seed}/author/complete.json")["rows"]
            cached = next(r for r in cached_rows if r["protein"] == name and r["coarse_steps"] == 25)
            baseline = evaluate_values(raw, reference, dict(samples=len(raw), failure_count=0, failures=[]))
            for key in METRIC_KEYS:
                np.testing.assert_allclose(baseline[key], cached[key], atol=1e-8, rtol=1e-8)
            for key in GEOMETRY_KEYS:
                np.testing.assert_allclose(baseline["geometry"][key], cached["geometry"][key], atol=1e-8, rtol=1e-8)
            measured = evaluate_values(projected, reference, audit)
            row = dict(seed=seed, protein=name, original=baseline, projected=measured)
            if measured["metrics_available"]:
                row["paired_input_displacement"] = dict(
                    mean_aligned_CA_RMSD_A=float(np.mean([aligned_rmsd(y[:, 1], x[:, 1]) for x, y in zip(raw, projected)])),
                    mean_aligned_backbone_RMSD_A=float(np.mean([aligned_rmsd(y.reshape(-1, 3), x.reshape(-1, 3)) for x, y in zip(raw, projected)])))
            rows.append(row)
            atomic_json(ROOT / f"seed{seed}/{name}/evaluation.json", row)
            atomic_json(ROOT / "progress.json", dict(phase="evaluation", completed_groups=len(rows), total_groups=4))
            print("AUTHOR_CODEC_EVALUATED", seed, name, flush=True)
        diagnostic, audit = project_all(codec, reference)
        folder = ROOT / "native_oracle"; folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / (name + ".npz"), **diagnostic)
        result = dict(protein=name, diagnostic_only=True, **audit)
        if audit["failure_count"] == 0:
            result["paired_native_reconstruction_CA_RMSD_A"] = float(np.mean([
                aligned_rmsd(y[:, 1], x[:, 1]) for x, y in zip(reference, diagnostic["backbone"])]))
        oracle_rows.append(result)
    if state_digest(codec) != initial or any(v.requires_grad or v.grad is not None for v in codec.parameters()):
        raise RuntimeError("Read-only codec state changed")
    summary = {}
    for arm in ["original", "projected"]:
        available = all(r[arm]["metrics_available"] for r in rows)
        summary[arm] = dict(metrics_available=available,
            failure_count=sum(r[arm]["failure_count"] for r in rows))
        if available:
            summary[arm].update({k: float(np.mean([r[arm][k] for r in rows])) for k in METRIC_KEYS})
            summary[arm]["geometry"] = {k: float(np.mean([r[arm]["geometry"][k] for r in rows])) for k in GEOMETRY_KEYS}
    atomic_json(ROOT / "complete.json", dict(status="complete_awaiting_review", rows=rows,
        two_seed_development_macro=summary, native_oracle_diagnostic=oracle_rows,
        codec_unchanged=True, optimizer_updates=0, all_samples_retained=True,
        test_coordinates_opened=False, R3_opened=False, host=platform.node(),
        cpu_threads=torch.get_num_threads(), scope=p["scope"]))
    print("AUTHOR_CODEC_INTERFACE_COMPLETE", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = load(ROOT / "protocol.json"); verify(p); run(p)


if __name__ == "__main__":
    main()
