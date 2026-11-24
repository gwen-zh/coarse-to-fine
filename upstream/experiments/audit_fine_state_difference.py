"""Exact mean/contrast velocity-error decomposition at the flow's noise start.

For the selected two-point training distribution and one shared noise draw,
v*_A=f_A-noise and v*_B=f_B-noise. Average two-state MSE decomposes exactly into
mean-velocity MSE plus one quarter of the predicted state-difference MSE.
This is about a fixed two-point trainability probe, not proof of ensemble
collapse: a valid conditional flow can average over unresolved fine variation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_explicit_coarse_fine import make_model as adapter_model, verify
from run_fine_single_structure import key_seed, make_model as parent_model


ROOT = Path("artifacts/fine_state_difference_v1")
TWO = Path("artifacts/fine_two_state_v1")
ADAPTER = Path("artifacts/explicit_coarse_fine_v1")
VERSIONS = ["parent", "two_state", "raw_context", "local_geometry"]


@torch.no_grad()
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "report.json").exists():
        raise RuntimeError("Use a new audit namespace")
    p = json.loads((ADAPTER / "protocol.json").read_text())
    verify(p)
    pairs = torch.load(TWO / "pairs.pt", map_location="cpu", weights_only=True)
    records = []
    sources = [Path(__file__), ADAPTER / "protocol.json", ADAPTER / "report.json", TWO / "pairs.pt"]
    for name in p["proteins"]:
        cases = [pairs[name]["a"], pairs[name]["b"]]
        aa = encode_aa(cases[0]["sequence"])
        for seed in p["seeds"]:
            for version in VERSIONS:
                if version in ["parent", "two_state"]:
                    model = parent_model(p, cases[0], seed)
                    path = Path(p["checkpoints"][str(seed)]) if version == "parent" else TWO / name / f"seed{seed}" / "last.pt"
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    model.load_state_dict(saved["model"])
                    if version == "two_state" and saved["protocol_sha256"] != sha(TWO / "protocol.json"):
                        raise RuntimeError("Wrong two-state checkpoint")
                else:
                    model, _ = adapter_model(p, cases[0], seed, version)
                    path = ADAPTER / version / name / f"seed{seed}" / "last.pt"
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    if saved["protocol_sha256"] != sha(ADAPTER / "protocol.json"):
                        raise RuntimeError("Wrong adapter checkpoint")
                    model.load_state_dict(saved["model"])
                sources.append(path)
                model.eval().requires_grad_(False)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                fs = [model.normalize_fine(c["fine_continuous"]) for c in cases]
                cs = [model.normalize_coarse(c["coarse"]) for c in cases]
                delta = fs[1] - fs[0]
                blind_floor = float(delta.square().mean() / 4)
                values = []
                for index in range(32):
                    rng = torch.Generator().manual_seed(key_seed(seed, name, "state-difference-audit", index))
                    noise = torch.randn(fs[0].shape, generator=rng)
                    va, vb = [model.fine_velocity(aa, noise, 0., c) for c in cs]
                    ta, tb = fs[0] - noise, fs[1] - noise
                    total = ((va - ta).square().mean() + (vb - tb).square().mean()) / 2
                    mean_loss = (((va + vb) - (ta + tb)) / 2).square().mean()
                    contrast_loss = ((vb - va) - delta).square().mean() / 4
                    torch.testing.assert_close(total, mean_loss + contrast_loss, atol=1e-6, rtol=1e-5)
                    response = vb - va
                    cosine = float((response * delta).sum() / (response.norm() * delta.norm()).clamp_min(1e-10))
                    values.append(dict(total_mse=float(total), mean_mse=float(mean_loss),
                        contrast_mse=float(contrast_loss), contrast_fraction=float(contrast_loss / total),
                        state_response_over_native=float(response.square().mean().sqrt() / delta.square().mean().sqrt()),
                        state_response_direction_cosine=cosine))
                record = dict(protein=name, seed=seed, version=version,
                    coarse_blind_optimal_mse=blind_floor,
                    **{k: float(np.mean([v[k] for v in values])) for k in values[0]})
                records.append(record)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                print(name, seed, version, "total", round(record["total_mse"], 5),
                    "mean", round(record["mean_mse"], 5), "contrast", round(record["contrast_mse"], 5), flush=True)
    verify(p)
    summary = []
    for name in p["proteins"]:
        for version in VERSIONS:
            group = [r for r in records if r["protein"] == name and r["version"] == version]
            keys = [k for k in group[0] if k not in ["protein", "seed", "version"]]
            summary.append(dict(protein=name, version=version, **{k: float(np.mean([r[k] for r in group])) for k in keys}))
    atomic_json(ROOT / "report.json", dict(status="complete", records=records, summary=summary,
        source_sha256={str(f): sha(f) for f in sources}, samples_per_model=32, flow_time=0.,
        exact_decomposition_checked_all_draws=True, training_performed=False, model_weights_unchanged=True,
        scope="Two selected two-state native TRAIN probes with paired fresh Gaussian draws; no full-trajectory, sampled-coarse, unseen-sequence, test or R3 evaluation.",
        interpretation="Contrast error at noise start measures whether these fixed coarse labels select the corresponding point targets. It does not imply that a real ensemble's instantaneous velocity or final samples must avoid conditional averaging.",
        test_coordinates_opened=False, R3_opened=False))
    print("STATE_DIFFERENCE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
