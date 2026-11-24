"""Raw ensemble evaluation; invalid samples are retained, never filtered."""
from pathlib import Path
import time

import numpy as np
import torch

from block_local_frames import FrameGeometryError
from coarse_generation_diagnostic import coarse_metrics, pool_backbones, weighted_rmsd_matrix
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import physical_histogram_edges
from native_ensemble_metrics import compare_ensembles
from run_coarse_generation_diagnostic import project_centers
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import write_pdb
from structural_metrics import aligned_rmsd


SUMMARY_KEYS = ["mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A",
    "js_pairwise_distance_divergence_nats", "rmsf_mae_A", "mean_full_generation_seconds"]
GEOMETRY_KEYS = ["cn_rate", "boundary_cn_rate", "ca_clashes_per_residue",
    "reference_ca_c_n_angle_mae_degrees", "reference_c_n_ca_angle_mae_degrees"]


@torch.no_grad()
def draw_fine_backbone(model, codec, aa, coarse, seed, nfe):
    missing = np.full((len(aa), 3, 3), np.nan)
    if coarse.shape != ((len(aa) + 3) // 4, 3):
        raise ValueError("Wrong coarse shape")
    if not torch.isfinite(coarse).all():
        return missing, dict(code="coarse_nonfinite")
    try:
        fine = model.sample_fine(aa, coarse, seed, nfe)
        if fine.shape != (len(aa), model.fine_dim):
            raise ValueError("Wrong fine shape")
        if not torch.isfinite(fine).all():
            return missing, dict(code="fine_nonfinite")
        bb = codec.decode(coarse, fine)
        if not torch.isfinite(bb).all():
            return missing, dict(code="backbone_nonfinite")
        torch.testing.assert_close(pool_ca(bb), coarse, atol=2e-5, rtol=2e-5)
        return bb.numpy(), None
    except FrameGeometryError as error:
        return missing, dict(code="coarse_frame_undefined", message=str(error))
    except ValueError as error:
        if "finite" not in str(error).lower():
            raise
        return missing, dict(code="nonfinite_sampling_state", message=str(error))


@torch.no_grad()
def evaluate(p, model, codec, groups, out, seed):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); model.eval(); rows = []
    for split, cases_by_name in [("MD_train", groups["train"]), ("development", groups["validation"])]:
        for name, cases in sorted(cases_by_name.items()):
            indices = np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)
            refs = [cases[int(i)] for i in indices]
            reference = np.stack([c["target"].numpy() for c in refs])
            reference_coarse = np.stack([c["coarse"].numpy() for c in refs])
            aa = encode_aa(cases[0]["sequence"]); length = len(aa)
            shuffled = aa[torch.randperm(length, generator=torch.Generator().manual_seed(719))]
            count = p["samples"] if split == "development" else p["train_diagnostic_samples"]
            oracle_indices = np.linspace(0, len(refs) - 1, count, dtype=int)
            outlets = ["full"] + (["oracle_coarse", "scrambled_AA"] if split == "development" else [])
            backbones = {v: [] for v in outlets}; coarse_values = {v: [] for v in outlets}
            failures = {v: [] for v in outlets}; latency = []
            for index in range(count):
                key = sample_seed(seed, name, index); started = time.perf_counter()
                c = project_centers(model.sample_coarse(aa, component_seed(key, "coarse"), p["coarse_nfe"]), length)
                bb, failure = draw_fine_backbone(model, codec, aa, c, component_seed(key, "fine"), p["fine_nfe"])
                latency.append(time.perf_counter() - started)
                backbones["full"].append(bb); coarse_values["full"].append(c.numpy())
                if failure is not None: failures["full"].append(dict(sample=index, **failure))
                if split == "development":
                    nc = refs[int(oracle_indices[index])]["coarse"]
                    # Preserve the existing oracle-noise convention exactly.
                    obb, failure = draw_fine_backbone(model, codec, aa, nc, key, p["fine_nfe"])
                    backbones["oracle_coarse"].append(obb); coarse_values["oracle_coarse"].append(nc.numpy())
                    if failure is not None: failures["oracle_coarse"].append(dict(sample=index, **failure))
                    sc = project_centers(model.sample_coarse(shuffled, component_seed(key, "coarse"), p["coarse_nfe"]), length)
                    sbb, failure = draw_fine_backbone(model, codec, shuffled, sc, component_seed(key, "fine"), p["fine_nfe"])
                    backbones["scrambled_AA"].append(sbb); coarse_values["scrambled_AA"].append(sc.numpy())
                    if failure is not None: failures["scrambled_AA"].append(dict(sample=index, **failure))
            edges, rg = physical_histogram_edges(length)
            folder = out / name; folder.mkdir(exist_ok=True)
            for outlet in outlets:
                bb = np.stack(backbones[outlet]); c = np.stack(coarse_values[outlet])
                stats = dict(metrics_available=not failures[outlet], failures=failures[outlet],
                    failure_count=len(failures[outlet]), generated_samples=count, reference_frames=len(refs),
                    mean_full_generation_seconds=float(np.mean(latency)), geometry=None)
                cs = coarse_metrics(c, reference_coarse, length, p["ca_step_cap_A"]) if np.isfinite(c).all() else None
                if stats["metrics_available"]:
                    stats.update(compare_ensembles(bb[:, :, 1], reference[:, :, 1], edges, rg, reference[0, :, 1]))
                    geometry = []
                    for value in bb:
                        nearest = int(np.argmin([aligned_rmsd(value[:, 1], r[:, 1]) for r in reference]))
                        geometry.append(metrics(value, reference[nearest]))
                    stats["geometry"] = {k: float(np.mean([v[k] for v in geometry])) for k in geometry[0]}
                    decoded_centers = pool_backbones(bb)
                    actual_bound = weighted_rmsd_matrix(decoded_centers, reference_coarse, length).min(1).mean()
                    stats["maximum_decoded_center_roundoff_A"] = float(np.max(np.abs(decoded_centers - c)))
                    if actual_bound > stats["mean_sample_nearest_reference_ca_rmsd_A"] + 1e-4:
                        raise RuntimeError("Fixed-center CA lower bound violated")
                stats.update(protein=name, split=split, outlet=outlet, coarse_metrics=cs)
                rows.append(stats)
                np.savez_compressed(folder / (outlet + ".npz"), backbone=bb, coarse=c,
                    valid=np.isfinite(bb).all(axis=(1, 2, 3)))
                if outlet == "full" and split == "development" and stats["metrics_available"]:
                    write_pdb(folder / "ensemble.pdb", bb, cases[0]["sequence"])
            atomic_json(out / "evaluation_progress.json", dict(protein=name, split=split, completed_groups=len(rows)))
            print("FULL_COARSE_EVALUATE", seed, split, name, "full failures", len(failures["full"]), flush=True)
    result = dict(status="complete", rows=rows, scope=p["scope"], no_sample_selection=True,
        test_coordinates_opened=False, R3_opened=False)
    atomic_json(out / "evaluation.json", result); return result


def summarize_rows(rows):
    if not rows: raise ValueError("Nonempty declared sequence cohort required")
    available = all(r["metrics_available"] for r in rows)
    result = dict(proteins=len(rows), failed_samples=sum(r["failure_count"] for r in rows), metrics_available=available)
    for key in SUMMARY_KEYS:
        result[key] = float(np.mean([r[key] for r in rows])) if available else None
    for key in GEOMETRY_KEYS:
        result[key] = float(np.mean([r["geometry"][key] for r in rows])) if available else None
    result["spread_ratio"] = float(np.mean([r["mean_pairwise_ca_rmsd_A"] / r["reference_mean_pairwise_ca_rmsd_A"] for r in rows])) if available else None
    for key in ["mean_sample_nearest_reference_center_rmsd_A", "mean_reference_nearest_sample_center_rmsd_A",
                "generated_conditional_span_flag_rate", "js_pairwise_center_distance_nats", "spread_ratio"]:
        result["coarse_" + key] = float(np.mean([r["coarse_metrics"][key] for r in rows])) if all(r["coarse_metrics"] is not None for r in rows) else None
    return result
