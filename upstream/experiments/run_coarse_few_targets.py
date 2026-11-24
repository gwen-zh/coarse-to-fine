"""Eight fixed static training targets: coarse-flow learnability only.

Keep the independently stored geometry-aware fine candidate and structure
maps frozen. Report the predeclared 2048/8192-update endpoints; deliberately
learning one target per sequence is not evidence for ensemble fidelity.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from block_local_frames import FrameGeometryError
from coarse_few_target_flow import (FixedFineCoarseFlow, coarse_flow_loss,
    draw_seed, event, learning_rate)
from coarse_generation_diagnostic import scalar_features, weighted_rmsd_matrix
from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case
from run_coarse_generation_diagnostic import project_centers
from run_fine_geometry_flow import state_digest
from run_multiframe_coarse_context import make_model as make_context_model
from run_sequence_multiscale_ensemble import load_codec


ROOT = Path("artifacts/coarse_few_targets_v1")
PREVIOUS = Path("artifacts/coarse_generation_diagnostic_v1")
FINE = Path("artifacts/multiframe_coarse_context_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed coarse learnability input: " + name)


def make_model(p, cases, seed):
    candidate = make_context_model(p, seed, "local_geometry")
    original_coarse_hash = state_digest(candidate.coarse_model)
    saved = torch.load(FINE / "local_geometry" / f"seed{seed}" / "last.pt", map_location="cpu", weights_only=False)
    if saved["protocol_sha256"] != sha(FINE / "protocol.json") or saved["updates"] != 8192 or saved["arm"] != "local_geometry":
        raise RuntimeError("Wrong frozen geometry-aware fine candidate")
    candidate.load_state_dict(saved["model"])
    if state_digest(candidate.coarse_model) != original_coarse_hash:
        raise RuntimeError("Frozen fine candidate changed the original coarse checkpoint")
    bank = {c["sequence"]: candidate.fine_model.feature_bank[c["sequence"]] for c in cases}
    candidate.coarse_model.feature_bank = bank; candidate.fine_model.feature_bank = bank
    return FixedFineCoarseFlow(candidate.coarse_model, candidate.fine_model)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh learnability namespace")
    p = copy.deepcopy(json.loads((PREVIOUS / "protocol.json").read_text())); verify(p)
    assessed = json.loads((PREVIOUS / "assessment.json").read_text())
    if assessed["status"] != "complete_fixed_coarse_CA_bound_and_train_development_gaps_confirmed":
        raise RuntimeError("Assess the preceding coarse diagnostic first")
    train = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    by_id = {c["id"]: c for c in train["PDB_train"]}
    cases = [by_id[name] for name in p["PDB_probe_ids"]]
    if len(cases) != 8 or len({c["sequence"] for c in cases}) != 8:
        raise RuntimeError("Eight distinct fixed training sequences required")
    for case in cases: check_case(case, "train")
    torch.save(cases, ROOT / "cases.pt")
    p.update(training_performed=True, optimizer_updates=8192, updates=8192,
        milestones=[2048, 8192], checkpoint_updates=[2048, 4096, 6144, 8192], cycle_updates=2048,
        samples=8, coarse_nfe=16, coarse_diagnostic_nfe=64, fine_nfe=16,
        batch_size=4, lr=3e-4, min_lr=3e-5, case_ids=[c["id"] for c in cases],
        hypothesis="The prior coarse audit found substantial error even on static training targets. Test whether the existing coarse architecture and uniform FM objective can fit eight fixed known targets before scaling data or changing architecture.",
        architecture="Train the original-parent coarse network in its own parameter copy. Keep the completed geometry-aware fine network, structure encoder/decoder/codebook and normalization immutable. Fine-specific modules inside the coarse copy have no role in coarse velocity and are frozen too.",
        training="Eight static PDB TRAIN structures fixed by the preceding pretrain_probe_ids. One target per minibatch, four fresh Gaussian/time draws; cyclic eight-target order. New coarse branch starts from each original 12288-update parent. It does not start from another fitted-target probe. No MD, development, reserved-test or R3 structure/feature enters optimization.",
        objective="Mean normalized coarse flow velocity MSE, uniform t in [0,1]. Original Cartesian gauge and original target normalization, no target/prediction alignment inside loss. No geometry penalty, fine loss, quantization, decoder adaptation or inference-time selection.",
        learning_rate_schedule="Budget-independent 2048-step cosine cycles 3e-4 to 3e-5 with AdamW moments retained across restarts. The 2048 checkpoint is the exact prefix of the same 8192-step algorithm, not a separately tuned short model.",
        exposure={"2048": {"minibatches_per_target": 256, "noise_time_examples_per_target": 1024},
                  "8192": {"minibatches_per_target": 1024, "noise_time_examples_per_target": 4096}},
        evaluation="Parent, update2048 and update8192 only. Same eight fresh coarse/fine Gaussian seed pairs per training target at every endpoint. Euler16 is primary, Euler64 is diagnostic only. Report every target, seed and sample, coarse coordinate/weighted-aligned RMSD, spread/span flags and separate teacher-velocity errors at fixed times. Secondary full decoding uses the frozen fine candidate and has no native structural input at inference. Native-coarse fine reference and native reconstruction are labeled oracle diagnostics.",
        primary="Training-target learnability: Euler16 mean residue-count-weighted aligned center RMSD below 1.0 A for EVERY one of the eight targets in BOTH seeds at the fixed8192 endpoint. Report all endpoints whether it passes or fails; threshold is an engineering check, not biological significance or deployment qualification.",
        interpretation="One static target per sequence intentionally trains a point conditional distribution. Coarse noise-induced spread shrinking here is expected memorization, not demonstrated ensemble collapse or fidelity. Fresh Gaussian draws are not independent biological validation. Any train-set full-decoding gains are not new-sequence gains.",
        failure_policy="Keep all coarse samples. If geometric frame construction fails for a generated coarse, retain its coarse and a NaN raw-backbone placeholder, record the failure, and leave the aggregate secondary full metric undefined when any sample fails. Never omit failures or select a replacement sample.",
        scope="Eight preselected static PDB training targets, two parent seeds, bounded coarse-only trainability check. No MD trajectory fitting, development evaluation, test/R3 coordinate access, ensemble-performance claim or author-model comparison.",
        stop_rule="Exactly8192 updates with fixed2048/8192 reported checkpoints; no best epoch, adaptive extension, weight/LR scan or automatic default-model promotion. Keep the fine candidate and structure maps unchanged.")
    codec = load_codec(p, "centers3"); codec_hash = state_digest(codec); checks = []
    for case in cases:
        torch.testing.assert_close(codec.encode_coarse(case["target"]), case["coarse"], atol=1e-5, rtol=1e-5)
    for seed in p["seeds"]:
        model = make_model(p, cases, seed).eval()
        case = cases[0]; aa = encode_aa(case["sequence"])
        before_coarse = project_centers(model.sample_coarse(aa, 514, 16), len(aa))
        before_fine = model.sample_fine(aa, case["coarse"], 615, 16)
        fine_hash = state_digest(model.fine_model)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        model.train(); opt.zero_grad()
        loss, _ = coarse_flow_loss(model, aa, case["coarse"], torch.Generator().manual_seed(714))
        loss.backward(); grad = model.coarse_model.coarse_head[-1].weight.grad
        if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
            raise RuntimeError("No finite real-checkpoint coarse gradient")
        norm = float(grad.norm()); opt.step()
        if torch.equal(before_coarse, project_centers(model.sample_coarse(aa, 514, 16), len(aa))):
            raise RuntimeError("Real-checkpoint coarse update has no sampling effect")
        torch.testing.assert_close(before_fine, model.sample_fine(aa, case["coarse"], 615, 16), atol=0, rtol=0)
        if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash:
            raise RuntimeError("Preflight changed fine or structure maps")
        if any(v.grad is not None for v in model.fine_model.parameters()):
            raise RuntimeError("Fine branch received a gradient")
        checks.append(dict(seed=seed, coarse_gradient_norm=norm, training_feature_sequences=len(model.coarse_model.feature_bank),
            trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),
            fine_sampler_bitwise_unchanged_after_coarse_update=True, actual_trial_optimizer_updates=0))
    p["trainable_parameters"] = checks[0]["trainable_parameters"]
    if len({c["trainable_parameters"] for c in checks}) != 1: raise RuntimeError("Seed parameter counts differ")
    atomic_json(ROOT / "integration_check.json", dict(status="passed", cached_coarse_labels_replayed=8, checks=checks,
        discarded_preflight_optimizer_steps=len(checks), codec_unchanged=True, actual_trial_optimizer_updates=0))
    paths = [Path(__file__), Path("upstream/experiments/coarse_few_target_flow.py"),
        Path("upstream/tests/test_coarse_few_target_flow.py"), Path("upstream/scripts/coarse_few_targets.sbatch"),
        PREVIOUS / "protocol.json", PREVIOUS / "report.json", PREVIOUS / "assessment.json",
        FINE / "protocol.json", FINE / "report.json", FINE / "assessment.json",
        ROOT / "cases.pt", ROOT / "integration_check.json"]
    for seed in p["seeds"]: paths.append(FINE / "local_geometry" / f"seed{seed}" / "last.pt")
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    atomic_json(ROOT / "protocol.json", p)
    print("COARSE_FEW_TARGETS_READY", p["case_ids"], p["trainable_parameters"], flush=True)


@torch.no_grad()
def evaluate(p, model, codec, cases, seed, folder, oracle_reference=None):
    folder.mkdir(parents=True, exist_ok=True); model.eval(); rows = []
    for case in cases:
        aa = encode_aa(case["sequence"]); c = case["coarse"]; length = len(aa)
        destination = folder / case["protein"]; destination.mkdir(exist_ok=True)
        oracle = None
        if oracle_reference is None:
            decoded = []; oracle_fine = []
            for index in range(p["samples"]):
                key = component_seed(draw_seed(seed, case["id"], "evaluation", index), "fine")
                f = model.sample_fine(aa, c, key, p["fine_nfe"])
                bb = codec.decode(c, f)
                torch.testing.assert_close(pool_ca(bb), c, atol=2e-5, rtol=2e-5)
                decoded.append(metrics(bb.numpy(), case["target"].numpy())); oracle_fine.append(f.numpy())
            oracle = dict(metrics={k: float(np.mean([v[k] for v in decoded])) for k in decoded[0]},
                native_reconstruction=metrics(codec.decode(c, case["fine_continuous"]).numpy(), case["target"].numpy()))
            np.savez_compressed(destination / "native_coarse_fine_reference.npz", fine=np.stack(oracle_fine))
        else:
            oracle = next(r["native_coarse_fine_reference"] for r in oracle_reference["rows"] if r["case_id"] == case["id"])
        endpoints = []
        for nfe in [p["coarse_nfe"], p["coarse_diagnostic_nfe"]]:
            centers = []; bb_values = []; full_metrics = []; failures = []
            for index in range(p["samples"]):
                key = draw_seed(seed, case["id"], "evaluation", index)
                generated = project_centers(model.sample_coarse(aa, component_seed(key, "coarse"), nfe), length)
                centers.append(generated.numpy())
                if nfe == p["coarse_nfe"]:
                    try:
                        fine = model.sample_fine(aa, generated, component_seed(key, "fine"), p["fine_nfe"])
                        bb = codec.decode(generated, fine)
                        torch.testing.assert_close(pool_ca(bb), generated, atol=2e-5, rtol=2e-5)
                        full_metrics.append(metrics(bb.numpy(), case["target"].numpy())); bb_values.append(bb.numpy())
                    except FrameGeometryError as error:
                        failures.append(dict(sample=index, error=str(error)))
                        bb_values.append(np.full((length, 3, 3), np.nan))
            centers = np.stack(centers)
            bound = weighted_rmsd_matrix(centers, c[None].numpy(), length)[:, 0]
            spread = weighted_rmsd_matrix(centers, centers, length)[np.triu_indices(p["samples"], 1)]
            local = scalar_features(centers, length, p["ca_step_cap_A"])
            values = dict(nfe=nfe, mean_center_rmsd_A=float(bound.mean()), center_rmsd_A_by_sample=bound.tolist(),
                coordinate_rmse_A=float(np.sqrt(((centers - c.numpy()[None]) ** 2).mean())),
                center_spread_A=float(spread.mean()), conditional_span_flag_rate=float(local["conditional_span_flag"].mean()))
            arrays = dict(centers=centers)
            if nfe == p["coarse_nfe"]:
                values.update(full_decoding_failures=failures, full_decoding_failure_count=len(failures),
                    full_training_target_metrics=({k: float(np.mean([v[k] for v in full_metrics])) for k in full_metrics[0]} if not failures else None))
                arrays["backbone"] = np.stack(bb_values)
            np.savez_compressed(destination / f"coarse{nfe}.npz", **arrays); endpoints.append(values)
        normal = model.coarse_model.normalize_coarse(c); teacher = []
        for t in [.05, .25, .5, .75, .95]:
            scores = []
            for index in range(8):
                rng = torch.Generator().manual_seed(draw_seed(seed, case["id"], "teacher_evaluation", index))
                noise = torch.randn(normal.shape, generator=rng)
                v = model.coarse_model.coarse_velocity(aa, (1 - t) * noise + t * normal, t)
                scores.append(float((v - (normal - noise)).square().mean()))
            teacher.append(dict(time=t, velocity_mse=float(np.mean(scores))))
        rows.append(dict(case_id=case["id"], protein=case["protein"], endpoints=endpoints, teacher=teacher,
            mean_teacher_velocity_mse=float(np.mean([v["velocity_mse"] for v in teacher])), native_coarse_fine_reference=oracle))
    result = dict(status="complete", rows=rows, scope=p["scope"], fresh_noise_is_not_biological_validation=True)
    atomic_json(folder / "evaluation.json", result); return result


def train(p, seed):
    cases = torch.load(ROOT / "cases.pt", map_location="cpu", weights_only=True)
    model = make_model(p, cases, seed); codec = load_codec(p, "centers3")
    fine_hash = state_digest(model.fine_model); codec_hash = state_digest(codec)
    buffers = {k: v.clone() for k, v in model.coarse_model.named_buffers()}
    out = ROOT / f"seed{seed}"; out.mkdir(exist_ok=True)
    if (out / "initial.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh fixed-budget experiment only; inspect partial state")
    atomic_json(out / "initial.json", dict(coarse_sha256=state_digest(model.coarse_model), fine_sha256=fine_hash,
        codec_sha256=codec_hash, training_feature_sequences=len(model.coarse_model.feature_bank),
        trainable_parameters=sum(v.numel() for v in model.trainable_parameters())))
    atomic_json(out / "progress.json", dict(phase="initial_evaluation", updates=0, total=p["updates"]))
    initial = evaluate(p, model, codec, cases, seed, out / "step0")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    trace = []; identity = hashlib.sha256(); endpoints = []; training_seconds = 0.
    started = time.perf_counter(); block_started = started
    for step in range(p["updates"]):
        case_index, counters = event(step); case = cases[case_index]; aa = encode_aa(case["sequence"])
        model.train(); opt.zero_grad(set_to_none=True); value = 0.; draws = []
        lr = learning_rate(step, p["cycle_updates"], p["lr"], p["min_lr"])
        for group in opt.param_groups: group["lr"] = lr
        for counter in counters:
            loss, terms = coarse_flow_loss(model, aa, case["coarse"],
                torch.Generator().manual_seed(draw_seed(seed, case["id"], "training", counter)))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite coarse loss")
            (loss / p["batch_size"]).backward(); value += float(loss.detach()) / p["batch_size"]; draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), p["gradient_clip"])
        if not torch.isfinite(grad): raise RuntimeError("Nonfinite coarse gradient")
        opt.step()
        observed = dict(case_id=case["id"], noise_counters=counters, base_draws=draws, lr=lr)
        identity.update(json.dumps(observed, sort_keys=True).encode())
        trace.append(dict(update=step + 1, loss=value, gradient_norm=float(grad), **observed))
        if (step + 1) % 256 == 0:
            progress = dict(seed=seed, phase="training", updates=step + 1, total=p["updates"],
                loss=value, training_seconds=training_seconds + time.perf_counter() - block_started)
            atomic_json(out / "progress.json", progress); print("COARSE_FEW_TRAIN", json.dumps(progress), flush=True)
        if step + 1 in p["checkpoint_updates"]:
            if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash:
                raise RuntimeError("Coarse training changed fine or codec")
            for name, buffer in model.coarse_model.named_buffers(): torch.testing.assert_close(buffer, buffers[name], atol=0, rtol=0)
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), torch_rng_state=torch.get_rng_state(),
                updates=step + 1, seed=seed, protocol_sha256=sha(ROOT / "protocol.json")), out / f"checkpoint_{step + 1}.pt")
            atomic_json(out / "trace.json", trace)
        if step + 1 in p["milestones"]:
            training_seconds += time.perf_counter() - block_started
            atomic_json(out / "progress.json", dict(seed=seed, phase="endpoint_evaluation", updates=step + 1,
                total=p["updates"], training_seconds=training_seconds))
            evaluate(p, model, codec, cases, seed, out / f"step{step + 1}", initial)
            endpoints.append(dict(step=step + 1, evaluation=str(out / f"step{step + 1}/evaluation.json"),
                realized_training_identity_sha256=identity.hexdigest(), training_seconds=training_seconds))
            block_started = time.perf_counter()
    if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash:
        raise RuntimeError("Endpoint evaluation changed a frozen component")
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, updates=p["updates"], endpoints=endpoints,
        fine_and_codec_frozen=True, normalization_frozen=True, training_feature_sequences=8,
        training_seconds=training_seconds, seconds_after_initial_evaluation=time.perf_counter() - started,
        runtime_host=platform.node(), training_only=True, no_ensemble_claim=True,
        test_coordinates_opened=False, development_evaluated=False, R3_opened=False, model_default_replaced=False))
    atomic_json(out / "progress.json", dict(seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def summarize(evaluation, nfe):
    pairs = [(r, next(v for v in r["endpoints"] if v["nfe"] == nfe)) for r in evaluation["rows"]]
    result = {k: float(np.mean([v[k] for _, v in pairs])) for k in ["mean_center_rmsd_A", "coordinate_rmse_A", "center_spread_A", "conditional_span_flag_rate"]}
    result["worst_target_mean_center_rmsd_A"] = max(v["mean_center_rmsd_A"] for _, v in pairs)
    result["mean_teacher_velocity_mse"] = float(np.mean([r["mean_teacher_velocity_mse"] for r, _ in pairs]))
    if nfe == 16:
        result["full_decoding_failure_count"] = sum(v["full_decoding_failure_count"] for _, v in pairs)
        result["full_training_target_CA_rmsd_A"] = float(np.mean([v["full_training_target_metrics"]["ca_rmsd"] for _, v in pairs])) if not result["full_decoding_failure_count"] else None
        result["full_training_target_CN_rate"] = float(np.mean([v["full_training_target_metrics"]["cn_rate"] for _, v in pairs])) if not result["full_decoding_failure_count"] else None
        result["native_coarse_fine_CA_rmsd_A"] = float(np.mean([r["native_coarse_fine_reference"]["metrics"]["ca_rmsd"] for r, _ in pairs]))
        result["native_coarse_fine_CN_rate"] = float(np.mean([r["native_coarse_fine_reference"]["metrics"]["cn_rate"] for r, _ in pairs]))
    return result


def report(p):
    rows = []; summaries = []; checks = []
    for seed in p["seeds"]:
        out = ROOT / f"seed{seed}"; done = json.loads((out / "complete.json").read_text())
        trace = json.loads((out / "trace.json").read_text())
        if len(trace) != p["updates"] or [v["step"] for v in done["endpoints"]] != p["milestones"] or not done["fine_and_codec_frozen"]:
            raise RuntimeError("Missing fixed coarse endpoint")
        for step in [0] + p["milestones"]:
            evaluation = json.loads((out / f"step{step}/evaluation.json").read_text())
            if {r["case_id"] for r in evaluation["rows"]} != set(p["case_ids"]): raise RuntimeError("Target set changed")
            rows.append(dict(seed=seed, step=step, evaluation=evaluation))
            for nfe in [16, 64]: summaries.append(dict(seed=seed, step=step, nfe=nfe, **summarize(evaluation, nfe)))
            if step:
                counts = {name: 0 for name in p["case_ids"]}
                for row in trace[:step]: counts[row["case_id"]] += len(row["noise_counters"])
                if set(counts.values()) != {p["exposure"][str(step)]["noise_time_examples_per_target"]}: raise RuntimeError("Target exposures differ")
                checks.append(dict(seed=seed, step=step, presentations_per_target=counts))
    primary = [r for r in summaries if r["step"] == 8192 and r["nfe"] == 16]
    passed = all(r["worst_target_mean_center_rmsd_A"] < 1. for r in primary)
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, summaries=summaries,
        exposure_checks=checks, all_eight_targets_both_seeds_below_1A=passed, scope=p["scope"],
        training_only=True, no_ensemble_claim=True, model_default_replaced=False,
        test_coordinates_opened=False, development_evaluated=False, R3_opened=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_result_review",
        all_eight_targets_both_seeds_below_1A=passed, scope=p["scope"],
        next="Review all fixed endpoints, target errors, teacher losses and full-train diagnostics before proposing a separate scaling study. No automatic extension or promotion.",
        training_only=True, no_ensemble_claim=True, model_default_replaced=False))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n" + p["interpretation"] + "\n\n" +
        "\n".join(json.dumps(r) for r in summaries) + "\n")
    plot(summaries)
    print("COARSE_FEW_TARGETS_COMPLETE", json.dumps(summaries), "primary_pass", passed, flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), constrained_layout=True)
    for seed, color, marker in [(2026, "#24679e", "o"), (2027, "#c27b24", "^")]:
        values = sorted([r for r in rows if r["seed"] == seed and r["nfe"] == 16], key=lambda r: r["step"])
        for axis, key, scale in [(axes[0], "mean_center_rmsd_A", 1.), (axes[1], "full_training_target_CA_rmsd_A", 1.), (axes[2], "full_training_target_CN_rate", 100.)]:
            selected = [r for r in values if r[key] is not None]
            axis.plot([r["step"] for r in selected], [r[key] * scale for r in selected], color=color, marker=marker, label=f"seed {seed}")
        axes[1].axhline(values[0]["native_coarse_fine_CA_rmsd_A"], color=color, ls="--", lw=1)
        axes[2].axhline(100 * values[0]["native_coarse_fine_CN_rate"], color=color, ls="--", lw=1)
    axes[0].axhline(1., color=".5", ls=":", label="1 Å engineering check")
    for ax, ylabel in zip(axes, ["Weighted center RMSD (Å)", "Full decoded training-target CA RMSD (Å)", "Full decoded C–N abnormalities (%)"]):
        ax.set_xticks([0, 2048, 8192], ["Parent", "2,048", "8,192"]); ax.set_xlabel("Coarse-only updates")
        ax.set_ylabel(ylabel, fontsize=9); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Coarse-only learnability: eight fixed static TRAIN targets\nFrozen geometry-aware fine / structure maps · two seeds · dashed lines: native-coarse fine reference", fontsize=11)
    folder = Path("paper/figures/coarse_few_targets_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]: fig.savefig(folder / ("Coarse_Eight_Targets." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--seed", type=int, default=2026); args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare": prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report": report(p)
        else:
            if args.seed not in p["seeds"]: raise RuntimeError("Undeclared coarse learnability seed")
            train(p, args.seed)
