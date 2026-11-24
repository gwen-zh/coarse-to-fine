"""Separate coarse-state selection from noise-driven two-prototype coverage.

Only the two fixed native TRAIN fine labels are prototypes; these labels are
not independently identified physical basins or thermodynamic populations.
Thirty-two shared Gaussian draws per coarse condition; first eight replay the
registered training-probe evaluation. No filtering or new biological data.
"""
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_early_time_fine_flow import verify
from run_fine_single_structure import key_seed, make_model

ROOT = Path("artifacts/fine_state_coverage_v1")
PARENT = Path("artifacts/early_time_fine_flow_v1")


@torch.no_grad()
def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "report.json").exists():
        raise RuntimeError("Use a fresh coverage audit namespace")
    p = json.loads((PARENT / "protocol.json").read_text()); verify(p)
    prior = json.loads((PARENT / "report.json").read_text())
    pairs = torch.load(PARENT / "pairs.pt", map_location="cpu", weights_only=True)
    records = []; sources = [Path(__file__), PARENT / "protocol.json", PARENT / "report.json", PARENT / "pairs.pt"]
    for arm in p["arms"]:
        for name in p["proteins"]:
            cases = [pairs[name]["a"], pairs[name]["b"]]
            aa = encode_aa(cases[0]["sequence"])
            for seed in p["seeds"]:
                model = make_model(p, cases[0], seed)
                path = PARENT / arm / name / f"seed{seed}" / "last.pt"
                sources.append(path)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if saved["protocol_sha256"] != sha(PARENT / "protocol.json"):
                    raise RuntimeError("Wrong time-density checkpoint")
                model.load_state_dict(saved["model"]); model.eval().requires_grad_(False)
                before = {k: v.clone() for k, v in model.state_dict().items()}
                labels = [model.normalize_fine(c["fine_continuous"]) for c in cases]
                draws = []; predictions = [[], []]; accuracy = []; prefix = []
                for index in range(32):
                    key = component_seed(key_seed(seed, "two-state-evaluation:" + name, "evaluation", index), "fine")
                    for side, case in enumerate(cases):
                        f = model.normalize_fine(model.sample_fine(aa, case["coarse"], key, 16))
                        distances = [float((f - label).square().mean()) for label in labels]
                        predicted = -1 if distances[0] == distances[1] else int(np.argmin(distances))
                        score = .5 if predicted == -1 else float(predicted == side)
                        predictions[side].append(predicted); accuracy.append(score)
                        if index < 8:
                            prefix.append(score)
                        draws.append(dict(noise_index=index, coarse_state=side,
                                          nearest_fine_prototype=predicted, normalized_mse_to_prototypes=distances))
                old = next(r for r in prior["rows"] if r["arm"] == arm and r["protein"] == name and r["seed"] == seed)
                if float(np.mean(prefix)) != old["final"]["state_assignment_accuracy"]:
                    raise RuntimeError("First eight draws do not replay the registered probe")
                marginal = [sum(v.count(i) for v in predictions) for i in [0, 1]]
                record = dict(arm=arm, protein=name, seed=seed, samples_per_condition=32,
                    correct_conditional_state_fraction=float(np.mean(accuracy)),
                    first_eight_state_fraction=float(np.mean(prefix)),
                    counts_given_coarse_A={"fine_A": predictions[0].count(0), "fine_B": predictions[0].count(1), "tie": predictions[0].count(-1)},
                    counts_given_coarse_B={"fine_A": predictions[1].count(0), "fine_B": predictions[1].count(1), "tie": predictions[1].count(-1)},
                    overall_fine_A_B_counts=marginal,
                    paired_noise_state_switch_fraction=float(np.mean([a != b for a, b in zip(*predictions)])),
                    draws=draws)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)
                records.append(record)
                print("FINE_STATE_COVERAGE", arm, name, seed, record["correct_conditional_state_fraction"],
                      record["counts_given_coarse_A"], record["counts_given_coarse_B"], flush=True)
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", records=records,
        source_sha256={str(x): sha(x) for x in sources},
        first_eight_draws_replay_registered_evaluation=True,
        scope="Read-only Monte Carlo coverage of two selected TRAIN fine-label prototypes, at native coarse. Not physical-basin discovery, full-trajectory, unseen-sequence, or thermodynamic evaluation.",
        model_weights_unchanged=True, training_performed=False,
        test_coordinates_opened=False, R3_opened=False))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
