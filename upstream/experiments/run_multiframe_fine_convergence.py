"""Predeclared budget curve after the completed 2048-update negative study.

Use a budget-independent 2048-step cosine-restart schedule. The first cycle
must replay the previous study's weights; optimizer moments carry across
cycles. Both endpoints are reported; development never selects a checkpoint.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from diagnostic_io import atomic_json
from early_time_fine_flow import fine_time_loss
from fine_geometry_flow import draw_seed, sample_training_batch
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from run_fine_geometry_flow import make_model, state_digest
from run_multiframe_early_fine import (BASE, attach_development_features,
    conditional_probe, encode_aa, evaluate_development, load_codec, macro)


ROOT = Path("artifacts/multiframe_fine_convergence_v1")
PREVIOUS = Path("artifacts/multiframe_early_fine_v1")
ARMS = ["uniform", "early"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed convergence input: " + name)


def learning_rate(p, step):
    within = step % p["cycle_updates"]
    return p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * within / (p["cycle_updates"] - 1)))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Convergence plan already frozen")
    p = copy.deepcopy(json.loads((PREVIOUS / "protocol.json").read_text())); verify(p)
    assessed = json.loads((PREVIOUS / "assessment.json").read_text())
    if assessed["status"] != "complete_small_training_diagnostic_gain_no_material_full_generation_gain":
        raise RuntimeError("Previous negative result must be assessed")
    old_updates = p["updates"]
    p.update(updates=8192, cycle_updates=old_updates, milestones=[2048, 8192],
        PDB_events=24576, MD_events=8192,
        hypothesis="The completed 2048-update continuation has small training-set gains and no material full-generation gain. A separately planned, bounded training-budget curve tests whether additional exposure and optimization improve either training fit or reused-development generation. It does not presume that more training helps.",
        training="Fresh original-parent coarse/fine copies, same scoped training features and 3-PDB/1-MD batches as the completed continuation. Fixed 8192 updates, uniform/early arms and two seeds. AdamW moments are carried through all four 2048-update cycles. Only the declared cosine LR restarts at each cycle boundary.",
        learning_rate_schedule="Budget-independent cosine 3e-4 to 3e-5 within each 2048-update cycle; restart at updates 2048/4096/6144 with Adam state retained. The first 2048 events/LRs are exactly those of the frozen short study. Extra updates add data exposure and LR cycles, so do not describe this as holding LR constant.",
        matching="Uniform and early share initialization, all case/Gaussian/uniform draws, cyclic LR, optimizer updates and flow calls. The short endpoint is an exact prefix of the long training algorithm. A bitwise-identical 2048 checkpoint reuses its frozen evaluation; otherwise a <=2e-5 weight-roundoff deviation is documented and the endpoint is reevaluated. Larger deviations stop the run.",
        primary="Report fixed 2048 and 8192 endpoints against the original parent on the same two repeatedly used development proteins, all raw 64 AA-only samples per protein and seed. Include native-coarse diagnostics, accuracy, geometry, distribution/coverage, spread and elapsed time. No endpoint selection or independent-test claim.",
        scope="Separate exploratory budget-convergence study on existing five-MD-protein/static-PDB training data and two reused development sequences. The previous 2048-step result remains complete and unchanged; no reserved test or R3 coordinates are opened.",
        stop_rule="Exactly 8192 updates, checkpoints at 2048/8192 only, report both endpoints regardless of quality. No adaptive extension, new density/LR sweep, sample filtering or automatic default-model replacement.")
    for step in range(old_updates):
        prior = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / (old_updates - 1)))
        if learning_rate(p, step) != prior:
            raise RuntimeError("LR prefix does not reproduce the short study")
    paths = [Path(__file__), Path("upstream/scripts/multiframe_fine_convergence.sbatch"),
        PREVIOUS / "protocol.json", PREVIOUS / "report.json", PREVIOUS / "assessment.json",
        PREVIOUS / "continuation_audit.json"]
    for arm in ARMS:
        for seed in p["seeds"]:
            out = PREVIOUS / arm / f"seed{seed}"
            paths += [out / "last.pt", out / "trace.json", out / "evaluation.json",
                out / "initial.json", out / "initial_conditional_probe.json", out / "final_conditional_probe.json"]
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    atomic_json(ROOT / "protocol.json", p)
    atomic_json(ROOT / "preparation_check.json", dict(status="passed", existing_real_data_preflight_reused=True,
        all_2048_prefix_LRs_identical=True, cycle_boundary_LRs=[learning_rate(p, i) for i in [0, 2047, 2048, 4096, 6144]],
        actual_trial_optimizer_updates=0))
    print("MULTIFRAME_CONVERGENCE_READY", p["milestones"], flush=True)


@torch.no_grad()
def evaluate_endpoint(p, model, codec, groups, arm, seed, step, out):
    destination = out / f"step{step}"; destination.mkdir(exist_ok=True)
    if step == 2048:
        reference = PREVIOUS / arm / f"seed{seed}"
        old = torch.load(reference / "last.pt", map_location="cpu", weights_only=False)
        if old["protocol_sha256"] != sha(PREVIOUS / "protocol.json"):
            raise RuntimeError("Wrong short-study checkpoint")
        maximum = 0.; exact = True
        state = model.state_dict()
        if set(state) != set(old["model"]):
            raise RuntimeError("Prefix architecture differs")
        for name, value in state.items():
            prior = old["model"][name]
            exact = exact and torch.equal(value, prior)
            maximum = max(maximum, float((value - prior).abs().max()))
            torch.testing.assert_close(value, prior, atol=2e-5, rtol=0)
        if exact:
            return dict(step=step, evaluation=str(reference / "evaluation.json"),
                conditional_probe=str(reference / "final_conditional_probe.json"),
                metric_source="frozen_bitwise_identical_short_checkpoint", prefix_weights_bitwise_identical=True,
                prefix_maximum_weight_roundoff=maximum)
    else:
        exact, maximum = None, None
    conditional_probe(p, model, codec, groups, seed, destination / "conditional_probe.json")
    train_bank = model.fine_model.feature_bank
    development = torch.load(PREVIOUS / "development_labels.pt", map_location="cpu", weights_only=True)
    try:
        attach_development_features(p, model, development)
        evaluate_development(p, model, codec, development, destination, seed)
    finally:
        model.coarse_model.feature_bank = train_bank
        model.fine_model.feature_bank = train_bank
    maximum_center = 0.
    for name in development["validation"]:
        with np.load(BASE / "interleaved" / f"seed{seed}" / name / "full.npz") as z:
            before = z["backbone"].copy()
        with np.load(destination / name / "full.npz") as z:
            after = z["backbone"].copy()
        if before.shape != after.shape:
            raise RuntimeError("Raw development samples differ")
        for a, b in zip(before, after):
            ac, bc = pool_ca(torch.from_numpy(a)), pool_ca(torch.from_numpy(b))
            maximum_center = max(maximum_center, float((ac - bc).abs().max()))
            torch.testing.assert_close(ac, bc, atol=2e-5, rtol=2e-5)
    return dict(step=step, evaluation=str(destination / "evaluation.json"),
        conditional_probe=str(destination / "conditional_probe.json"), metric_source="evaluated_checkpoint",
        prefix_weights_bitwise_identical=exact, prefix_maximum_weight_roundoff=maximum,
        maximum_parent_coarse_center_roundoff_A=maximum_center)


def train(p, arm, seed):
    groups = torch.load(PREVIOUS / "train_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, seed); codec = load_codec(p, "centers3")
    out = ROOT / arm / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "initial.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh convergence run only; inspect partial artifacts")
    coarse_hash, codec_hash = state_digest(model.coarse_model), state_digest(codec)
    buffers = {k: v.clone() for k, v in model.fine_model.named_buffers()}
    atomic_json(out / "initial.json", dict(coarse_sha256=coarse_hash, codec_sha256=codec_hash,
        fine_sha256=state_digest(model.fine_model), trainable_parameters=p["trainable_parameters"]))
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed + 37247)
    trace = []; identity = hashlib.sha256(); endpoints = []; training_seconds = 0.
    block_started = time.perf_counter(); wall_started = block_started
    for step in range(p["updates"]):
        model.train(); batch = sample_training_batch(groups, rng)
        lr = learning_rate(p, step)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True); loss_sum = 0.; draws = []
        ids = [dict(protein=c["protein"], case_id=c["id"], domain="PDB" if i < 3 else "MD") for i, c in enumerate(batch)]
        for slot, case in enumerate(batch):
            loss, terms = fine_time_loss(model, encode_aa(case["sequence"]), case["coarse"], case["fine_continuous"],
                torch.Generator().manual_seed(draw_seed(seed, step, slot)), arm)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite convergence loss")
            (loss / 4).backward(); loss_sum += float(loss.detach()) / 4; draws.append(terms["draw_sha256"])
        grad = torch.nn.utils.clip_grad_norm_(params, p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite convergence gradient")
        opt.step()
        event = dict(batch=ids, base_draws=draws, lr=lr)
        identity.update(json.dumps(event, sort_keys=True).encode())
        trace.append(dict(update=step + 1, loss=loss_sum, gradient_norm=float(grad), **event))
        if (step + 1) % 256 == 0:
            progress = dict(arm=arm, seed=seed, phase="training", updates=step + 1, total=p["updates"],
                loss=loss_sum, training_seconds=training_seconds + time.perf_counter() - block_started)
            atomic_json(out / "progress.json", progress); print("FINE_CONVERGENCE_TRAIN", json.dumps(progress), flush=True)
        if step + 1 in p["milestones"]:
            training_seconds += time.perf_counter() - block_started
            if state_digest(model.coarse_model) != coarse_hash or state_digest(codec) != codec_hash:
                raise RuntimeError("Frozen component changed")
            for key, value in model.fine_model.named_buffers():
                torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), np_rng_state=rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(), updates=step + 1, arm=arm, seed=seed,
                protocol_sha256=sha(ROOT / "protocol.json")), out / f"checkpoint_{step + 1}.pt")
            atomic_json(out / "trace.json", trace)
            atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="endpoint_evaluation",
                updates=step + 1, total=p["updates"], training_seconds=training_seconds))
            endpoint = evaluate_endpoint(p, model, codec, groups, arm, seed, step + 1, out)
            endpoint.update(realized_training_identity_sha256=identity.hexdigest(), training_seconds=training_seconds)
            if step + 1 == 2048:
                expected = json.loads((PREVIOUS / arm / f"seed{seed}" / "complete.json").read_text())
                if identity.hexdigest() != expected["realized_training_identity_sha256"]:
                    raise RuntimeError("Short-study event/noise/LR prefix mismatch")
            endpoints.append(endpoint); atomic_json(out / "endpoints.json", endpoints)
            print("FINE_CONVERGENCE_ENDPOINT", arm, seed, step + 1, endpoint["metric_source"], flush=True)
            block_started = time.perf_counter()
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed, updates=p["updates"],
        endpoints=endpoints, training_seconds=training_seconds, total_seconds=time.perf_counter() - wall_started,
        runtime_host=platform.node(), coarse_and_codec_frozen=True, normalization_frozen=True,
        trainable_parameters=sum(v.numel() for v in params), model_default_replaced=False,
        test_coordinates_opened=False, R3_opened=False))
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def report(p):
    trials = []; macro_rows = []; train_rows = []; paired = []
    for seed in p["seeds"]:
        parent = json.loads((BASE / "interleaved" / f"seed{seed}" / "evaluation.json").read_text())
        trials.append(dict(arm="parent", seed=seed, step=0, evaluation=parent))
        signatures = []
        for arm in ARMS:
            out = ROOT / arm / f"seed{seed}"
            done = json.loads((out / "complete.json").read_text())
            if done["updates"] != p["updates"] or [r["step"] for r in done["endpoints"]] != p["milestones"]:
                raise RuntimeError("Missing fixed convergence endpoint")
            signatures.append([x["realized_training_identity_sha256"] for x in done["endpoints"]])
            for endpoint in done["endpoints"]:
                evaluation = json.loads(Path(endpoint["evaluation"]).read_text())
                trials.append(dict(arm=arm, seed=seed, step=endpoint["step"], evaluation=evaluation, metadata=endpoint))
                rows = json.loads(Path(endpoint["conditional_probe"]).read_text())["rows"]
                for domain in ["MD_train", "PDB_train"]:
                    selected = [r for r in rows if r["domain"] == domain]
                    proteins = sorted({r["protein"] for r in selected})
                    per_protein = [{key: float(np.mean([r["metrics"][key] for r in selected if r["protein"] == name]))
                                    for key in ["ca_rmsd", "cn_rate"]} for name in proteins]
                    train_rows.append(dict(arm=arm, seed=seed, step=endpoint["step"], domain=domain,
                        **{k: float(np.mean([r[k] for r in per_protein])) for k in ["ca_rmsd", "cn_rate"]}))
        if signatures[0] != signatures[1]:
            raise RuntimeError("Convergence arms do not share events/base draws/LR")
        paired.append(dict(seed=seed, all_milestone_training_identities_matched=True))
    for trial in trials:
        for outlet in ["full", "oracle_coarse"]:
            macro_rows.append(dict(arm=trial["arm"], seed=trial["seed"], step=trial["step"], outlet=outlet,
                                   **macro(trial["evaluation"], outlet)))
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", trials=trials, macro_per_seed=macro_rows,
        training_conditional_probes=train_rows, matching=paired, scope=p["scope"],
        all_fixed_endpoints_reported=True, model_default_replaced=False,
        independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n\n" + "\n".join(json.dumps(r) for r in macro_rows) + "\n")
    plot(p, macro_rows, train_rows)
    print("FINE_CONVERGENCE_COMPLETE", flush=True)


def plot(p, macro_rows, train_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    prior = json.loads((PREVIOUS / "report.json").read_text())["training_conditional_probes"]
    for arm, color in [("uniform", "#c27b24"), ("early", "#24679e")]:
        for seed, marker, style in [(2026, "o", "-"), (2027, "^", "--")]:
            for column, outlet in [(0, "MD_train"), (1, "oracle_coarse"), (2, "full")]:
                if column == 0:
                    base = next(r for r in prior if r["arm"] == arm and r["seed"] == seed and r["phase"] == "initial" and r["domain"] == outlet)
                    values = [dict(step=0, ca_rmsd=base["ca_rmsd_A"], cn_rate=base["cn_rate"])] + [r for r in train_rows if r["arm"] == arm and r["seed"] == seed and r["domain"] == outlet]
                    ca_key = "ca_rmsd"
                else:
                    values = [r for r in macro_rows if r["seed"] == seed and r["outlet"] == outlet and (r["arm"] == arm or r["arm"] == "parent")]
                    ca_key = "mean_sample_nearest_reference_ca_rmsd_A"
                values = sorted(values, key=lambda r: r["step"])
                for row, key, scale in [(0, ca_key, 1.), (1, "cn_rate", 100.)]:
                    axes[row, column].plot([r["step"] for r in values], [scale * r[key] for r in values], color=color,
                        marker=marker, ls=style, label=f"{arm}, seed {seed}")
    for column, title in enumerate(["MD training: native coarse", "Development: native coarse", "Development: full AA-only"]):
        axes[0, column].set_title(title, fontsize=10)
        for row in [0, 1]:
            axes[row, column].set_xticks([0, 2048, 8192], ["Parent", "2,048", "8,192"])
            axes[row, column].set_xlabel("Fine continuation updates"); axes[row, column].grid(axis="y", alpha=.2)
        axes[0, column].set_ylabel("CA error (Å)"); axes[1, column].set_ylabel("C–N abnormalities (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fine training budget: fixed endpoints, all results reported\nFrozen coarse/structure maps · two reused development proteins", fontsize=12)
    folder = Path("paper/figures/multiframe_fine_convergence_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Multi_Frame_Fine_Convergence." + ext), dpi=180)
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
                raise RuntimeError("Undeclared convergence trial")
            train(p, args.arm, args.seed)
