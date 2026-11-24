"""Capacity-matched coarse geometry conditioning on the full training cohort.

Only fine-flow conditioning changes. A separate immutable parent supplies
coarse samples. Both adapters start at exactly the parent fine function and
see the same native training pairs, base draws, optimizer and fixed budget.
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

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from early_time_fine_flow import fine_time_loss
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from fine_geometry_flow import FixedCoarseFineFlow, draw_seed, sample_training_batch
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from run_fine_geometry_flow import make_model as make_parent, state_digest
from run_multiframe_early_fine import (BASE, attach_development_features,
    conditional_probe, encode_aa, evaluate_development, load_codec, macro)
from run_multiframe_fine_convergence import learning_rate


ROOT = Path("artifacts/multiframe_coarse_context_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")
PREVIOUS = Path("artifacts/multiframe_fine_convergence_v1")
ARMS = ["raw_context", "local_geometry"]
ADAPTER_KEYS = {"context_adapter.0.weight", "context_adapter.0.bias",
                "context_adapter.2.weight", "context_adapter.2.bias"}


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed multi-frame context input: " + name)


def make_model(p, seed, arm):
    if arm not in ARMS:
        raise ValueError("Undeclared coarse-context arm")
    parent = make_parent(p, seed)
    adapter_seed = int(hashlib.sha256(f"multiframe-coarse-context-v1:{seed}".encode()).hexdigest()[:8], 16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(adapter_seed)
        fine = ExplicitCoarseFineFlow(**p["normalization"], **p["model"], coarse_dim=3,
            feature_mode="ESM3_context", feature_dim=p["feature_dimension"], context_kind=arm)
    missing, unexpected = fine.load_state_dict(parent.fine_model.state_dict(), strict=False)
    if set(missing) != ADAPTER_KEYS or unexpected:
        raise RuntimeError("Original fine weights were not loaded exactly")
    fine.feature_bank = parent.fine_model.feature_bank
    return FixedCoarseFineFlow(parent.coarse_model, fine)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh experiment namespace")
    p = copy.deepcopy(json.loads((PREVIOUS / "protocol.json").read_text()))
    verify(p)
    assessment = json.loads((PREVIOUS / "assessment.json").read_text())
    if assessment["status"] != "complete_more_training_improves_training_fit_small_geometry_change_no_accuracy_gain":
        raise RuntimeError("Previous budget study must be complete and assessed")
    p.update(arms=ARMS, updates=8192, milestones=[8192], cycle_updates=2048,
        checkpoint_updates=[2048, 4096, 6144, 8192], batch_size=4, samples=64,
        time_density={"both_arms": "uniform density 1 on [0,1]; no early-time reweighting"},
        PDB_events=24576, MD_events=8192,
        hypothesis="Explicit deterministic local geometry from the SAME supplied coarse may help fine flow learn compatible local structure. The prior selected two-state probe gave small seed-sensitive gains; this study tests transfer to multi-frame training and the two repeatedly used development sequences without presuming a gain.",
        architecture="A frozen original-parent copy supplies coarse. A trainable fine copy receives an identical zero-output 24->128->128 adapter in both arms, added to coarse prefix and broadcast to residues. raw_context inputs normalized XYZ plus 21 zeros; local_geometry inputs the frozen decoder's 24 relative-center/frame features calculated solely from supplied coarse. Original Cartesian conditioning remains in both arms.",
        training="Fresh original-parent fine weights, not the previously continued weights. Exactly 8192 additional fine updates. Same 3 static-PDB + 1 real-MD native-paired batch, uniform FM loss, optimizer, cyclic LR and base draws as the completed uniform budget study. Structure encoder/decoder/codebook, normalization and parent coarse sampler remain frozen. Fine stays continuous without VQ or snap.",
        matching="Primary raw-context vs local-geometry contrast has identical 19,712 added parameters, initial state_dict, optimizer settings, cases, Gaussian/time draws and Transformer calls. Epoch-zero fine velocities/samplers equal the original parent. Analytic geometry preprocessing costs time and is measured. Frozen no-adapter uniform8192 is a lower-capacity reference, not the capacity-matched comparator.",
        objective="Uniform-time fine flow velocity MSE only; no new geometry loss, state labels, decoder feedback, sampling-time rejection or reference-based acceptance.",
        fine_training_condition="Native coarse and native continuous fine encoded from the SAME training structure. No independently sampled coarse is paired with another structure's fine. Native-to-generated coarse exposure mismatch remains a limitation.",
        primary="One fixed 8192-update endpoint, two seeds and two reused development proteins, all 64 raw samples per protein. Report native-coarse matched-frame CA/C-N plus nearest-reference and coverage metrics, and full AA-only accuracy, geometry, distribution, spread and elapsed time. Report each protein/seed and both arms regardless of outcome. Native-coarse diagnostics use reference structure only as an explicitly labeled diagnostic; full sampling accepts AA features and noise only.",
        interpretation="Both native-coarse and AA-only outlets are evaluated regardless of the first outcome. No selection on development, no equivalence claim from small changes, and no causal speed claim across different runtime hosts. Conditional fit improvement alone is not an end-to-end or ensemble-quality improvement.",
        scope="Exploratory full-cohort conditioning ablation: 899 static PDB training structures and 2264 frames from five MD training proteins; two repeatedly used development sequences. Not an independent test, thermodynamic calibration, kinetic model, or author-model comparison.",
        stop_rule="Exactly 8192 updates per arm/seed. Intermediate checkpoints only for recovery, no metric-based checkpoint selection or adaptive extension. Keep every raw sample. No reserved test or R3 coordinates, automatic default replacement or new test access.")
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    if len(groups["PDB_train"]) != 899 or {k: len(v) for k, v in groups["train"].items()} != p["MD_training_frames"]:
        raise RuntimeError("Training cohort differs")
    codec = load_codec(p, "centers3")
    codec_hash = state_digest(codec)
    pdb = {c["id"]: c for c in groups["PDB_train"]}
    cases = [case for _, frames in sorted(groups["train"].items()) for case in [frames[0], frames[-1]]]
    cases += [pdb[name] for name in p["PDB_probe_ids"]]
    checks = []; starts = []; counts = []
    for seed in p["seeds"]:
        base = make_parent(p, seed).eval()
        hashes = []
        for arm in ARMS:
            model = make_model(p, seed, arm).eval()
            hashes.append(state_digest(model.fine_model))
            count = sum(v.numel() for v in model.trainable_parameters()); counts.append(count)
            if len(model.fine_model.feature_bank) != 904:
                raise RuntimeError("Training feature bank must contain only 904 train sequences")
            for case in cases:
                aa = encode_aa(case["sequence"]); c = case["coarse"]; f = case["fine_continuous"]
                condition = model.fine_model.normalize_coarse(c)
                if arm == "local_geometry":
                    torch.testing.assert_close(model.fine_model.context_features(condition, len(aa)),
                        codec.context(c, len(aa))["geometry"], atol=2e-5, rtol=2e-5)
                with torch.no_grad():
                    torch.testing.assert_close(model.fine_model.fine_velocity(aa, f, .3, condition),
                        base.fine_model.fine_velocity(aa, f, .3, condition), atol=0, rtol=0)
                checks.append(dict(seed=seed, arm=arm, protein=case["protein"], case_id=case["id"]))
            case = cases[0]; aa = encode_aa(case["sequence"])
            c = base.sample_coarse(aa, 347, p["coarse_nfe"])
            torch.testing.assert_close(model.sample_coarse(aa, 347, p["coarse_nfe"]), c, atol=0, rtol=0)
            torch.testing.assert_close(model.sample_fine(aa, c, 713, p["fine_nfe"]),
                base.sample_fine(aa, c, 713, p["fine_nfe"]), atol=0, rtol=0)
            before = state_digest(model.coarse_model)
            model.train(); opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
            loss, _ = fine_time_loss(model, aa, case["coarse"], case["fine_continuous"],
                torch.Generator().manual_seed(714), "uniform")
            loss.backward()
            grad = model.fine_model.context_adapter[-1].weight.grad
            if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
                raise RuntimeError("Adapter has no real-data gradient")
            grad_norm = float(grad.norm()); opt.step()
            if state_digest(model.coarse_model) != before or state_digest(codec) != codec_hash:
                raise RuntimeError("Preflight updated a frozen component")
            torch.testing.assert_close(model.sample_coarse(aa, 347, p["coarse_nfe"]), c, atol=0, rtol=0)
            starts.append(dict(seed=seed, arm=arm, adapter_gradient_norm=grad_norm,
                fine_initial_sha256=hashes[-1], training_feature_sequences=904))
        if hashes[0] != hashes[1]:
            raise RuntimeError("Initial adapter weights differ between arms")
    if set(counts) != {1107848}:
        raise RuntimeError("Matched trainable parameter count changed")
    p["trainable_parameters"] = counts[0]
    # Discard all preflight optimizer updates; actual trials construct fresh models.
    atomic_json(ROOT / "integration_check.json", dict(status="passed", checks=checks, starts=starts,
        added_parameters=19712, trainable_parameters=counts[0],
        epoch_zero_both_samplers_bitwise_parent=True, local_geometry_matches_decoder=True,
        both_arms_initial_weights_identical=True, parent_coarse_unchanged_after_fine_step=True,
        codec_unchanged=True, actual_trial_optimizer_updates=0,
        development_features_attached_during_training=False))
    paths = [Path(__file__), Path("upstream/scripts/multiframe_coarse_context.sbatch"),
        Path("upstream/experiments/explicit_coarse_fine_flow.py"),
        Path("upstream/tests/test_explicit_coarse_fine_flow.py"),
        PREVIOUS / "protocol.json", PREVIOUS / "report.json", PREVIOUS / "assessment.json",
        Path("artifacts/explicit_coarse_fine_v1/protocol.json"),
        Path("artifacts/explicit_coarse_fine_v1/report.json"),
        Path("artifacts/explicit_coarse_fine_v1/assessment.json"), ROOT / "integration_check.json"]
    for seed in p["seeds"]:
        folder = PREVIOUS / "uniform" / f"seed{seed}"
        paths += [folder / "complete.json", folder / "initial.json", folder / "trace.json",
            folder / "step8192/evaluation.json", folder / "step8192/conditional_probe.json"]
        for name in ["5w82_E", "5ydn_A"]:
            paths += [folder / "step8192" / name / "oracle_coarse.npz",
                BASE / "interleaved" / f"seed{seed}" / name / "oracle_coarse.npz",
                BASE / "interleaved" / f"seed{seed}" / name / "full.npz"]
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    atomic_json(ROOT / "protocol.json", p)
    print("MULTIFRAME_CONTEXT_READY", len(checks), "real-state checks", flush=True)


def assert_frozen(model, codec, coarse_hash, codec_hash, buffers):
    if state_digest(model.coarse_model) != coarse_hash or state_digest(codec) != codec_hash:
        raise RuntimeError("Frozen model changed")
    for name, value in model.fine_model.named_buffers():
        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)


def matched_oracle(groups, out):
    rows = []
    for name, cases in sorted(groups["validation"].items()):
        take = np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)
        refs = [cases[int(i)] for i in take]
        with np.load(out / name / "oracle_coarse.npz") as z:
            backbones = z["backbone"].copy()
        indices = np.linspace(0, len(refs) - 1, len(backbones), dtype=int)
        for index, (bb, ref_index) in enumerate(zip(backbones, indices)):
            case = refs[int(ref_index)]
            torch.testing.assert_close(pool_ca(torch.from_numpy(bb)), case["coarse"], atol=2e-5, rtol=2e-5)
            rows.append(dict(protein=name, sample=index, native_case_id=case["id"],
                metrics=metrics(bb, case["target"].numpy())))
    return dict(status="complete", rows=rows,
        scope="Supplied native coarse; errors against its SAME native frame. Diagnostic, not deployable AA-only inference or independent testing.")


def train(p, arm, seed):
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, seed, arm); codec = load_codec(p, "centers3")
    out = ROOT / arm / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "initial.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh trial only; inspect partial work before resubmission")
    coarse_hash = state_digest(model.coarse_model); codec_hash = state_digest(codec)
    buffers = {k: v.clone() for k, v in model.fine_model.named_buffers()}
    params = model.trainable_parameters()
    atomic_json(out / "initial.json", dict(coarse_sha256=coarse_hash, codec_sha256=codec_hash,
        fine_sha256=state_digest(model.fine_model), trainable_parameters=sum(v.numel() for v in params),
        training_feature_sequences=len(model.fine_model.feature_bank)))
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed + 37247); trace = []; identity = hashlib.sha256()
    started = time.perf_counter(); training_started = started
    for step in range(p["updates"]):
        model.train(); batch = sample_training_batch(groups, rng); lr = learning_rate(p, step)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True); value = 0.; draws = []
        ids = [dict(protein=c["protein"], case_id=c["id"], domain="PDB" if i < 3 else "MD") for i, c in enumerate(batch)]
        for slot, case in enumerate(batch):
            loss, terms = fine_time_loss(model, encode_aa(case["sequence"]), case["coarse"], case["fine_continuous"],
                torch.Generator().manual_seed(draw_seed(seed, step, slot)), "uniform")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite context loss")
            (loss / 4).backward(); value += float(loss.detach()) / 4; draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(params, p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite context gradient")
        opt.step()
        event = dict(batch=ids, base_draws=draws, lr=lr)
        identity.update(json.dumps(event, sort_keys=True).encode())
        trace.append(dict(update=step + 1, loss=value, gradient_norm=float(grad), **event))
        if (step + 1) % 256 == 0:
            progress = dict(arm=arm, seed=seed, phase="training", updates=step + 1, total=p["updates"],
                loss=value, training_seconds=time.perf_counter() - training_started)
            atomic_json(out / "progress.json", progress); print("MULTIFRAME_CONTEXT_TRAIN", json.dumps(progress), flush=True)
        if step + 1 in p["checkpoint_updates"]:
            assert_frozen(model, codec, coarse_hash, codec_hash, buffers)
            path = out / ("last.pt" if step + 1 == p["updates"] else f"checkpoint_{step + 1}.pt")
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), np_rng_state=rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(), arm=arm, seed=seed, updates=step + 1,
                protocol_sha256=sha(ROOT / "protocol.json")), path)
            atomic_json(out / "trace.json", trace)
    training_seconds = time.perf_counter() - training_started
    previous = json.loads((PREVIOUS / "uniform" / f"seed{seed}" / "complete.json").read_text())
    expected = previous["endpoints"][-1]["realized_training_identity_sha256"]
    if identity.hexdigest() != expected:
        raise RuntimeError("Data/noise/time/LR differ from fixed no-adapter reference")
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="training_probe", updates=p["updates"], total=p["updates"], training_seconds=training_seconds))
    conditional_probe(p, model, codec, groups, seed, out / "conditional_probe.json")
    train_bank = model.fine_model.feature_bank
    development = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="development_evaluation", updates=p["updates"], total=p["updates"], training_seconds=training_seconds))
    evaluation_started = time.perf_counter()
    try:
        attach_development_features(p, model, development)
        evaluate_development(p, model, codec, development, out, seed)
        atomic_json(out / "native_coarse_matched_frames.json", matched_oracle(development, out))
    finally:
        model.coarse_model.feature_bank = train_bank; model.fine_model.feature_bank = train_bank
    maximum_center_roundoff = 0.
    for name in development["validation"]:
        with np.load(BASE / "interleaved" / f"seed{seed}" / name / "full.npz") as z:
            before = z["backbone"].copy()
        with np.load(out / name / "full.npz") as z:
            after = z["backbone"].copy()
        if before.shape != after.shape:
            raise RuntimeError("Development sample counts differ")
        for a, b in zip(before, after):
            ac, bc = pool_ca(torch.from_numpy(a)), pool_ca(torch.from_numpy(b))
            maximum_center_roundoff = max(maximum_center_roundoff, float((ac - bc).abs().max()))
            torch.testing.assert_close(ac, bc, atol=2e-5, rtol=2e-5)
    assert_frozen(model, codec, coarse_hash, codec_hash, buffers); verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed, updates=p["updates"],
        realized_training_identity_sha256=identity.hexdigest(), matched_no_adapter_event_identity=True,
        trainable_parameters=sum(v.numel() for v in params), training_seconds=training_seconds,
        development_evaluation_seconds=time.perf_counter() - evaluation_started,
        total_seconds=time.perf_counter() - started, runtime_host=platform.node(),
        maximum_parent_coarse_center_roundoff_A=maximum_center_roundoff,
        coarse_codec_and_normalization_frozen=True, model_default_replaced=False,
        independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def probe_summary(probe):
    rows = probe["rows"]; result = []
    for domain in sorted({r.get("domain", "development_native_coarse_matched") for r in rows}):
        selected = [r for r in rows if r.get("domain", "development_native_coarse_matched") == domain]
        proteins = sorted({r["protein"] for r in selected})
        per_protein = [dict(protein=name, **{key: float(np.mean([r["metrics"][key] for r in selected if r["protein"] == name]))
            for key in ["ca_rmsd", "cn_rate", "boundary_cn_rate"]}) for name in proteins]
        result.append(dict(domain=domain, per_protein=per_protein,
            **{key: float(np.mean([r[key] for r in per_protein])) for key in ["ca_rmsd", "cn_rate", "boundary_cn_rate"]}))
    return result


def report(p):
    development = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    trials = []; macro_rows = []; train_rows = []; matched_rows = []; pairing = []
    for seed in p["seeds"]:
        signatures = []; starts = []
        references = [("parent", BASE / "interleaved" / f"seed{seed}"),
            ("no_adapter_uniform8192", PREVIOUS / "uniform" / f"seed{seed}" / "step8192")]
        for arm, out in references + [(arm, ROOT / arm / f"seed{seed}") for arm in ARMS]:
            evaluation = json.loads((out / "evaluation.json").read_text())
            trial = dict(arm=arm, seed=seed, evaluation=evaluation)
            if arm in ARMS:
                done = json.loads((out / "complete.json").read_text())
                if done["updates"] != p["updates"] or not done["coarse_codec_and_normalization_frozen"]:
                    raise RuntimeError("Incomplete context trial")
                trial["complete"] = done; signatures.append(done["realized_training_identity_sha256"])
                starts.append(json.loads((out / "initial.json").read_text()))
            trials.append(trial)
            for outlet in ["oracle_coarse", "full"]:
                macro_rows.append(dict(arm=arm, seed=seed, outlet=outlet, **macro(evaluation, outlet)))
            matched = (json.loads((out / "native_coarse_matched_frames.json").read_text()) if arm in ARMS
                       else matched_oracle(development, out))
            matched_rows += [dict(arm=arm, seed=seed, **r) for r in probe_summary(matched)]
            if arm != "parent":
                probe = json.loads((out / "conditional_probe.json").read_text())
                train_rows += [dict(arm=arm, seed=seed, **r) for r in probe_summary(probe)]
        if len(set(signatures)) != 1 or starts[0] != starts[1]:
            raise RuntimeError("Adapter initialization or actual training inputs differ")
        pairing.append(dict(seed=seed, initial_states_and_case_noise_time_LR_hashes_match=True))
    deltas = []
    for seed in p["seeds"]:
        for outlet in ["oracle_coarse", "full"]:
            a = next(r for r in macro_rows if r["arm"] == "raw_context" and r["seed"] == seed and r["outlet"] == outlet)
            b = next(r for r in macro_rows if r["arm"] == "local_geometry" and r["seed"] == seed and r["outlet"] == outlet)
            deltas.append(dict(seed=seed, outlet=outlet, local_minus_raw={key: b[key] - a[key]
                for key in a if key not in {"arm", "seed", "outlet"}}))
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", trials=trials, macro_per_seed=macro_rows,
        training_conditional_probes=train_rows, development_matched_native_frames=matched_rows,
        geometry_minus_raw_per_seed=deltas, pairing=pairing, scope=p["scope"], matching=p["matching"],
        all_fixed_endpoints_reported=True, model_default_replaced=False,
        independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_result_review",
        geometry_minus_raw_per_seed=deltas, primary_comparison="local_geometry vs capacity-matched raw_context",
        scope=p["scope"], independent_test_claim=False, model_default_replaced=False,
        next="Review both seeds, both proteins and conditional/full outlets before proposing any next experiment. No automatic extension or promotion."))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n\n" + "\n".join(json.dumps(r) for r in macro_rows) + "\n")
    plot(macro_rows, matched_rows)
    print("MULTIFRAME_CONTEXT_COMPLETE", json.dumps(deltas), flush=True)


def plot(macro_rows, matched_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = ["parent", "no_adapter_uniform8192"] + ARMS
    labels = ["Parent", "No adapter", "Raw XYZ", "Local geom."]
    colors = ["#777777", "#b8b8b8", "#c27b24", "#24679e"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for column, outlet in enumerate(["matched", "oracle_coarse", "full"]):
        source = matched_rows if outlet == "matched" else [r for r in macro_rows if r["outlet"] == outlet]
        keys = ["ca_rmsd", "cn_rate"] if outlet == "matched" else ["mean_sample_nearest_reference_ca_rmsd_A", "cn_rate"]
        for row, key in enumerate(keys):
            ax = axes[row, column]; scale = 100 if key == "cn_rate" else 1
            for index, (arm, color) in enumerate(zip(arms, colors)):
                selected = [r for r in source if r["arm"] == arm]
                ax.bar(index, scale * np.mean([r[key] for r in selected]), color=color, alpha=.75)
                for value in selected:
                    ax.plot(index + (-.09 if value["seed"] == 2026 else .09), scale * value[key],
                        "o" if value["seed"] == 2026 else "^", color="black", ms=5)
            ax.set_xticks(range(4), labels, rotation=15, fontsize=8)
            ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
            ax.set_ylabel("C–N abnormalities (%)" if row else ("Matched-frame CA RMSD (Å)" if column == 0 else "Sample→nearest-reference CA RMSD (Å)"), fontsize=9)
        axes[0, column].set_title(["Development: native coarse\nMatched native frame", "Development: native coarse\nNearest ensemble frame", "Development: full AA-only\nNearest ensemble frame"][column], fontsize=10)
    fig.suptitle("Explicit coarse geometry in fine flow: fixed 8,192-update comparison\nTwo reused development proteins · two seeds (circle/triangle) · no independent-test claim", fontsize=12)
    folder = Path("paper/figures/multiframe_coarse_context_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Multi_Frame_Coarse_Context." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--arm", choices=ARMS); parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report":
            report(p)
        else:
            if args.arm not in ARMS or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared multi-frame context trial")
            train(p, args.arm, args.seed)
