"""Nested8/32 coarse training: paired target exposure and per-round LR.

The control is rerun because the old global-step cosine restarts did not
match per-target LR when target count changes. Fine and structure maps stay
fixed; biological generalization is outside this trainability experiment.
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

from coarse_few_target_flow import coarse_flow_loss, draw_seed
from coarse_target_scaling import (ROUNDS, TARGETS, initial_metric_roundoff,
    plan, round_learning_rate, select_nested_ids, training_event)
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case
from run_coarse_few_targets import evaluate, make_model, summarize
from run_fine_geometry_flow import state_digest
from run_sequence_multiscale_ensemble import load_codec


ROOT = Path("artifacts/coarse_target_scaling_v1")
PARENT = Path("artifacts/coarse_few_targets_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")
ARMS = ["eight", "thirtytwo"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        h = hashlib.sha256()
        with Path(name).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
        if h.hexdigest() != digest:
            raise RuntimeError("Changed coarse-scaling input: " + name)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh scaling namespace")
    p = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text())); verify(p)
    assessed = json.loads((PARENT / "assessment.json").read_text())
    if assessed["status"] != "complete_eight_static_coarse_targets_fit_and_composition_improves_training_only":
        raise RuntimeError("The eight-target study must be complete and assessed")
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    by_id = {c["id"]: c for c in groups["PDB_train"]}
    nested = select_nested_ids(by_id, p["case_ids"])
    all_cases = [by_id[name] for name in nested]
    if len({c["sequence"] for c in all_cases}) != 32:
        raise RuntimeError("Expected32 unique training sequences")
    for c in all_cases: check_case(c, "train")
    for arm, count in TARGETS.items():
        (ROOT / arm).mkdir(exist_ok=True)
        torch.save(all_cases[:count], ROOT / arm / "cases.pt")
    selection = dict(first_eight=nested[:8], added_twentyfour=nested[8:],
        rule="Append24 remaining TRAIN case IDs ranked by SHA256('coarse-32-training-v1:'+case_id); no geometry, fit, validation or test selection.")
    atomic_json(ROOT / "selection.json", selection)
    p.update(arms=ARMS, cohorts={arm: plan(arm) for arm in ARMS}, case_ids=nested,
        common_case_ids=nested[:8], added_case_ids=nested[8:], rounds=ROUNDS, samples=8,
        hypothesis="Does coarse training-target fitting persist from8 to32 fixed static structures when each target has the same noise/time draws, presentations and per-round LR? Examine the SAME eight targets separately from the24 added targets.",
        architecture="Same original-parent coarse copy and independently frozen geometry-aware fine candidate as the completed8-target study. Parameter count, structure encoder/decoder/codebook, train-only normalization and inference Euler16 remain fixed. Euler64 is a diagnostic only.",
        training="Fresh parent coarse for each cohort/seed, not the fitted8-target checkpoint. One target per four-noise minibatch in fixed cyclic case order. The32-target list begins with the exact old8; the remaining24 are selected by fixed ID hash. Feature banks contain only the training sequences of that cohort.",
        learning_rate_schedule="One256-round cosine, 3e-4 to3e-5; LR is CONSTANT for every update in one round. No restarts. Round=global_update_index//target_count. Thus the common8 receive exactly the same per-case LR sequence and Gaussian/time examples. This differs from the old global-step cyclic schedule, so rerun the8-target control.",
        exposure="Report after64 and256 rounds:256 and1024 Gaussian/time examples per target.8-target endpoints are512/2048 global updates;32-target endpoints2048/8192. Every checkpoint is a fixed prefix of its cohort's declared training algorithm.",
        matching="Common8: identical initial coarse/fine state, case labels, per-case Gaussian/time draw counters and hashes, LR at each exposure, optimizer settings and inference NFE.32 targets require4x as many optimizer updates and more total forward work; this is PER-TARGET exposure matching, not matched total compute. Adam histories differ because of intervening targets.",
        evaluation="Fresh paired sampling/teacher noise exactly as the completed coarse study. Report all8/32 targets at parent and both fixed exposure endpoints. Separate common8, added24 and all targets. Full decoding and native-coarse fine references reuse the frozen generic evaluator; any decoder failure is retained and invalidates its aggregate secondary full metric.",
        primary="Coarse Euler16 trainability: at256 rounds, EVERY target in the32-target cohort must have mean weighted center error<1 A in BOTH seeds for the engineering check to pass. Report paired common8 differences against the rerun8-target control at BOTH64/256 rounds regardless of this check.",
        interpretation="Static training-target fitting only. Gaussian resampling is not independent biological validation. Decreased spread is expected for one target per sequence. Shared-target comparisons reduce cohort-composition confounding but do not isolate capacity from optimizer interference or prove unseen-sequence/generalization performance.",
        scope="Nested8/32 static PDB training structures, two parent seeds. Coarse-only continuation with frozen fine/structure maps. No development, MD continuation, reserved-test or R3 evaluation; no ensemble-quality or author-baseline claim.",
        stop_rule="Exactly256 rounds per cohort, report both64/256-round endpoints and all targets. No best checkpoint, adaptive extension, geometry filtering, parameter/LR scan or automatic default replacement.")
    for stale in ["updates", "optimizer_updates", "milestones", "checkpoint_updates", "cycle_updates",
                  "MD_events", "PDB_events", "MD_training_frames", "PDB_proteins", "diagnostic_proteins"]:
        p.pop(stale, None)
    p["max_updates"] = 8192
    p["initial_metric_tolerance"] = dict(atol=2e-5, rtol=2e-5,
        note="Initial parameter hashes must match exactly; cross-host evaluation scalars allow only this floating-point tolerance.")
    codec = load_codec(p, "centers3"); codec_hash = state_digest(codec)
    for c in all_cases:
        torch.testing.assert_close(codec.encode_coarse(c["target"]), c["coarse"], atol=1e-5, rtol=1e-5)
    checks = []
    for seed in p["seeds"]:
        signatures = []; replay = None
        old = json.loads((PARENT / f"seed{seed}" / "initial.json").read_text())
        for arm in ARMS:
            cases = all_cases[:TARGETS[arm]]; model = make_model(p, cases, seed).eval()
            hashes = [state_digest(model.coarse_model), state_digest(model.fine_model)]
            if hashes != [old["coarse_sha256"], old["fine_sha256"]]:
                raise RuntimeError("Scaling initialization differs from original parents")
            signatures.append(hashes)
            first = cases[0]; aa = encode_aa(first["sequence"])
            initial_pair = (model.sample_coarse(aa, 514, 16), model.sample_fine(aa, first["coarse"], 615, 16))
            if replay is not None:
                for before, after in zip(replay, initial_pair): torch.testing.assert_close(before, after, atol=0, rtol=0)
            replay = initial_pair
            last = cases[-1]; last_aa = encode_aa(last["sequence"])
            before_fine = model.sample_fine(last_aa, last["coarse"], 819, 16)
            model.train(); opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
            loss, _ = coarse_flow_loss(model, last_aa, last["coarse"], torch.Generator().manual_seed(913))
            loss.backward(); grad = model.coarse_model.coarse_head[-1].weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
                raise RuntimeError("No finite scaling gradient on real training target")
            norm = float(grad.norm()); opt.step()
            torch.testing.assert_close(before_fine, model.sample_fine(last_aa, last["coarse"], 819, 16), atol=0, rtol=0)
            if state_digest(model.fine_model) != hashes[1] or state_digest(codec) != codec_hash:
                raise RuntimeError("Preflight changed frozen fine or structure maps")
            checks.append(dict(seed=seed, arm=arm, training_feature_sequences=len(model.coarse_model.feature_bank),
                coarse_gradient_norm=norm, trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),
                initial_coarse_sha256=hashes[0], initial_fine_sha256=hashes[1],
                fine_sampler_bitwise_unchanged_after_coarse_update=True, actual_trial_optimizer_updates=0))
        if signatures[0] != signatures[1]: raise RuntimeError("Cohort initial weights differ")
    atomic_json(ROOT / "integration_check.json", dict(status="passed", cached_coarse_labels_checked=32,
        checks=checks, actual_trial_optimizer_updates=0, fine_and_codec_frozen=True,
        common_initial_coarse_and_fine_samples_bitwise_equal=True))
    paths = [Path(__file__), Path("upstream/experiments/coarse_target_scaling.py"),
        Path("upstream/tests/test_coarse_target_scaling.py"), Path("upstream/scripts/coarse_target_scaling.sbatch"),
        PARENT / "protocol.json", PARENT / "report.json", PARENT / "assessment.json",
        PARENT / "cases.pt", ROOT / "selection.json", ROOT / "integration_check.json"]
    for seed in p["seeds"]: paths.append(PARENT / f"seed{seed}" / "initial.json")
    for arm in ARMS: paths.append(ROOT / arm / "cases.pt")
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    # Child protocols do not depend on the root manifest; the root freezes
    # their hashes afterwards. No circular hash dependency.
    for arm in ARMS:
        spec = plan(arm); trial = copy.deepcopy(p)
        trial.update(cohort=arm, targets=spec["targets"], training_rounds=ROUNDS,
            case_ids=nested[:spec["targets"]], updates=spec["updates"], optimizer_updates=spec["updates"],
            milestones=spec["milestones"], checkpoint_updates=spec["checkpoint_updates"])
        atomic_json(ROOT / arm / "protocol.json", trial)
    p["source_sha256"].update({str(ROOT / arm / "protocol.json"): sha(ROOT / arm / "protocol.json") for arm in ARMS})
    atomic_json(ROOT / "protocol.json", p)
    print("COARSE_SCALING_READY", p["cohorts"], flush=True)


def train(p, arm, seed):
    base = ROOT / arm; cases = torch.load(base / "cases.pt", map_location="cpu", weights_only=True)
    if len(cases) != p["targets"] or [c["id"] for c in cases] != p["case_ids"]:
        raise RuntimeError("Wrong cohort")
    for c in cases: check_case(c, "train")
    model = make_model(p, cases, seed); codec = load_codec(p, "centers3")
    fine_hash = state_digest(model.fine_model); codec_hash = state_digest(codec)
    buffers = {k: v.clone() for k, v in model.coarse_model.named_buffers()}
    out = base / f"seed{seed}"; out.mkdir(exist_ok=True)
    if (out / "initial.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh scaling trial only; inspect partial artifacts before resubmitting")
    initial = dict(coarse_sha256=state_digest(model.coarse_model), fine_sha256=fine_hash, codec_sha256=codec_hash,
        trainable_parameters=sum(v.numel() for v in model.trainable_parameters()), training_feature_sequences=len(model.coarse_model.feature_bank))
    atomic_json(out / "initial.json", initial)
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="initial_evaluation", updates=0, total=p["updates"]))
    initial_eval = evaluate(p, model, codec, cases, seed, out / "step0")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    trace = []; identity = hashlib.sha256(); endpoints = []; training_seconds = 0.
    started = time.perf_counter(); block_started = started
    for step in range(p["updates"]):
        index, counters = training_event(step, p["targets"]); case = cases[index]; aa = encode_aa(case["sequence"])
        model.train(); opt.zero_grad(set_to_none=True); value = 0.; draws = []
        lr = round_learning_rate(step, p["targets"], p["training_rounds"], p["lr"], p["min_lr"])
        for group in opt.param_groups: group["lr"] = lr
        for counter in counters:
            loss, terms = coarse_flow_loss(model, aa, case["coarse"],
                torch.Generator().manual_seed(draw_seed(seed, case["id"], "training", counter)))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite coarse-scaling loss")
            (loss / 4).backward(); value += float(loss.detach()) / 4; draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), p["gradient_clip"])
        if not torch.isfinite(grad): raise RuntimeError("Nonfinite coarse-scaling gradient")
        opt.step(); observation = dict(case_id=case["id"], noise_counters=counters, base_draws=draws, lr=lr)
        identity.update(json.dumps(observation, sort_keys=True).encode())
        trace.append(dict(update=step + 1, round=step // p["targets"], loss=value, gradient_norm=float(grad), **observation))
        if (step + 1) % 256 == 0:
            progress = dict(arm=arm, seed=seed, phase="training", updates=step + 1, total=p["updates"],
                rounds_completed=(step + 1) // p["targets"], loss=value, training_seconds=training_seconds + time.perf_counter() - block_started)
            atomic_json(out / "progress.json", progress); print("COARSE_SCALING_TRAIN", json.dumps(progress), flush=True)
        if step + 1 in p["checkpoint_updates"]:
            if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash:
                raise RuntimeError("Coarse training changed frozen fine or structure maps")
            for name, value_buffer in model.coarse_model.named_buffers(): torch.testing.assert_close(value_buffer, buffers[name], atol=0, rtol=0)
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), torch_rng_state=torch.get_rng_state(),
                seed=seed, arm=arm, updates=step + 1, protocol_sha256=sha(base / "protocol.json"),
                root_protocol_sha256=sha(ROOT / "protocol.json")), out / f"checkpoint_{step + 1}.pt")
            atomic_json(out / "trace.json", trace)
        if step + 1 in p["milestones"]:
            training_seconds += time.perf_counter() - block_started
            atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="endpoint_evaluation", updates=step + 1,
                total=p["updates"], training_seconds=training_seconds))
            evaluate(p, model, codec, cases, seed, out / f"step{step + 1}", initial_eval)
            endpoints.append(dict(step=step + 1, rounds=(step + 1) // p["targets"],
                noise_time_examples_per_target=4 * ((step + 1) // p["targets"]),
                realized_training_identity_sha256=identity.hexdigest(), training_seconds=training_seconds))
            block_started = time.perf_counter()
    if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash:
        raise RuntimeError("Final evaluation changed frozen components")
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed, updates=p["updates"],
        targets=p["targets"], endpoints=endpoints, fine_and_codec_frozen=True, normalization_frozen=True,
        training_feature_sequences=len(model.coarse_model.feature_bank), training_seconds=training_seconds,
        seconds_after_initial_evaluation=time.perf_counter() - started, runtime_host=platform.node(),
        training_only=True, no_ensemble_claim=True, development_evaluated=False,
        test_coordinates_opened=False, R3_opened=False, model_default_replaced=False))
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def report(p):
    rows = []; summaries = []; pairing = []; per_target = []
    for seed in p["seeds"]:
        initials = []; actual_events = {}
        for arm in ARMS:
            trial = json.loads((ROOT / arm / "protocol.json").read_text()); verify(trial)
            out = ROOT / arm / f"seed{seed}"; done = json.loads((out / "complete.json").read_text())
            trace = json.loads((out / "trace.json").read_text()); initial = json.loads((out / "initial.json").read_text())
            initials.append({k: v for k, v in initial.items() if k != "training_feature_sequences"})
            if len(trace) != trial["updates"] or [v["step"] for v in done["endpoints"]] != trial["milestones"] or not done["fine_and_codec_frozen"]:
                raise RuntimeError("Incomplete scaling trial")
            actual_events[arm] = {name: [{k: r[k] for k in ["noise_counters", "base_draws", "lr"]}
                for r in trace if r["case_id"] == name] for name in p["common_case_ids"]}
            counts = {name: sum(len(r["noise_counters"]) for r in trace if r["case_id"] == name) for name in trial["case_ids"]}
            if set(counts.values()) != {1024}: raise RuntimeError("Per-target exposure mismatch")
            for step in [0] + trial["milestones"]:
                evaluation = json.loads((out / f"step{step}/evaluation.json").read_text())
                if {r["case_id"] for r in evaluation["rows"]} != set(trial["case_ids"]): raise RuntimeError("Target set changed during evaluation")
                rounds = step // trial["targets"]; draws = 4 * rounds
                rows.append(dict(arm=arm, seed=seed, step=step, rounds=rounds, evaluation=evaluation))
                groups = {"all": trial["case_ids"], "common8": p["common_case_ids"]}
                if arm == "thirtytwo": groups["added24"] = p["added_case_ids"]
                for group_name, ids in groups.items():
                    subset = dict(rows=[r for r in evaluation["rows"] if r["case_id"] in ids])
                    for nfe in [16, 64]:
                        summaries.append(dict(arm=arm, seed=seed, step=step, rounds=rounds,
                            examples_per_target=draws, group=group_name, nfe=nfe, **summarize(subset, nfe)))
                for value in evaluation["rows"]:
                    endpoint = next(v for v in value["endpoints"] if v["nfe"] == 16)
                    per_target.append(dict(arm=arm, seed=seed, step=step, case_id=value["case_id"],
                        center_rmsd_A=endpoint["mean_center_rmsd_A"], full_decoding_failure_count=endpoint["full_decoding_failure_count"]))
        if initials[0] != initials[1] or actual_events["eight"] != actual_events["thirtytwo"]:
            raise RuntimeError("Initial weights or common-target data/noise/LR histories differ")
        for arm in ARMS:
            e = next(r["evaluation"] for r in rows if r["arm"] == arm and r["seed"] == seed and r["step"] == 0)
            subset = [r for r in e["rows"] if r["case_id"] in p["common_case_ids"]]
            if arm == "eight": initial_common = subset
            else: initial_difference = initial_metric_roundoff(initial_common, subset)
        pairing.append(dict(seed=seed, initial_weights_equal=True,
            shared_initial_metrics_match_declared_tolerance=True, maximum_initial_metric_roundoff=initial_difference,
            common_target_labels_gaussian_time_draws_LRs_and_presentations_equal=True))
    paired_deltas = []
    for seed in p["seeds"]:
        for exposure in [256, 1024]:
            a, b = [next(r for r in summaries if r["seed"] == seed and r["arm"] == arm
                         and r["group"] == "common8" and r["nfe"] == 16 and r["examples_per_target"] == exposure) for arm in ARMS]
            keys = ["mean_center_rmsd_A", "worst_target_mean_center_rmsd_A", "full_training_target_CA_rmsd_A", "full_training_target_CN_rate"]
            paired_deltas.append(dict(seed=seed, examples_per_target=exposure,
                thirtytwo_minus_eight_common_targets={k: b[k] - a[k] if b[k] is not None and a[k] is not None else None for k in keys}))
    final = [r for r in per_target if r["arm"] == "thirtytwo" and r["step"] == 8192]
    passed = len(final) == 64 and all(r["center_rmsd_A"] < 1. for r in final)
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, summaries=summaries,
        per_target_primary=per_target, pairing=pairing, paired_common_target_deltas=paired_deltas,
        all32_targets_both_seeds_below_1A=passed, scope=p["scope"], matching=p["matching"],
        training_only=True, no_ensemble_claim=True, model_default_replaced=False,
        development_evaluated=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_result_review", all32_targets_both_seeds_below_1A=passed,
        scope=p["scope"], next="Review shared8 and added24 errors under matched exposures before designing any full-cohort or MD study. No automatic extension, development evaluation or default promotion.",
        training_only=True, no_ensemble_claim=True, model_default_replaced=False))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n" + p["matching"] + "\n\n" +
        "\n".join(json.dumps(r) for r in summaries) + "\n")
    plot(summaries)
    print("COARSE_TARGET_SCALING_COMPLETE", json.dumps(paired_deltas), "primary_pass", passed, flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["mean_center_rmsd_A", "full_training_target_CA_rmsd_A", "full_training_target_CN_rate"]
    scales = [1., 1., 100.]
    labels = ["Center RMSD (Å)", "Decoded training-target CA RMSD (Å)", "C–N abnormalities (%)"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for arm, color in [("eight", "#777777"), ("thirtytwo", "#24679e")]:
        for seed, marker in [(2026, "o"), (2027, "^")]:
            selected = sorted([r for r in rows if r["arm"] == arm and r["seed"] == seed and r["group"] == "common8" and r["nfe"] == 16], key=lambda r: r["examples_per_target"])
            for column, (key, scale) in enumerate(zip(keys, scales)):
                values = [r for r in selected if r[key] is not None]
                axes[0, column].plot([r["examples_per_target"] for r in values], [scale * r[key] for r in values],
                    color=color, marker=marker, lw=1, label=f"{arm}, seed{seed}")
    categories = [("eight", "common8"), ("thirtytwo", "common8"), ("thirtytwo", "added24")]
    for index, (arm, group) in enumerate(categories):
        selected = [r for r in rows if r["arm"] == arm and r["group"] == group and r["nfe"] == 16 and r["examples_per_target"] == 1024]
        for column, (key, scale) in enumerate(zip(keys, scales)):
            values = [r for r in selected if r[key] is not None]
            if values:
                ax = axes[1, column]; ax.bar(index, scale * np.mean([r[key] for r in values]), color=["#777777", "#24679e", "#c27b24"][index], alpha=.7)
                for r in values: ax.plot(index + (-.07 if r["seed"] == 2026 else .07), scale * r[key], "o" if r["seed"] == 2026 else "^", color="black", ms=4)
                if column in [1, 2]:
                    oracle_key = "native_coarse_fine_CA_rmsd_A" if column == 1 else "native_coarse_fine_CN_rate"
                    ax.plot(index, scale * np.mean([r[oracle_key] for r in values]), "D", mfc="white", mec="black", ms=6)
    for column, label in enumerate(labels):
        axes[0, column].set_title("Same eight targets", fontsize=10)
        axes[0, column].set_xticks([0, 256, 1024]); axes[0, column].set_xlabel("Gaussian/time examples per target")
        axes[1, column].set_xticks(range(3), ["8 cohort\ncommon8", "32 cohort\ncommon8", "32 cohort\nadded24"], fontsize=8)
        for row in [0, 1]: axes[row, column].set_ylabel(label, fontsize=9); axes[row, column].grid(axis="y", alpha=.2); axes[row, column].set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].axhline(1., color=".4", ls=":", lw=1)
    fig.suptitle("Coarse learnability: matched per-target exposure, nested8/32 TRAIN targets\nSame per-target noise and round LR · two seeds · bottom:1,024 examples/target; white diamonds:native-coarse fine reference", fontsize=11)
    folder = Path("paper/figures/coarse_target_scaling_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]: fig.savefig(folder / ("Coarse_Target_Scaling." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--arm", choices=ARMS); parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare": prepare()
    else:
        root = json.loads((ROOT / "protocol.json").read_text()); verify(root)
        if args.stage == "report": report(root)
        else:
            if args.arm not in ARMS or args.seed not in root["seeds"]: raise RuntimeError("Undeclared scaling trial")
            p = json.loads((ROOT / args.arm / "protocol.json").read_text()); verify(p)
            train(p, args.arm, args.seed); verify(root)
