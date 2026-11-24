"""Multi-frame fine FM continuation: uniform versus fixed early-time density.

Frozen parent coarse sampler and structure maps; same paired native labels,
three static PDB replay slots plus one real MD training frame per update.
Development sequences are repeatedly used exploratory cases, not a new test.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from compare_two_level_ae_author import metrics
from diagnostic_io import atomic_json
from early_time_fine_flow import fine_time_loss
from fine_geometry_flow import draw_seed, sample_training_batch
from hierarchical_ensemble_flow import component_seed
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import AA_ALPHABET, encode_aa
from multisequence_flow_data import check_case
from run_fine_geometry_flow import make_model, state_digest
from run_multisequence_ensemble_flow import evaluate as evaluate_development
from run_sequence_multiscale_ensemble import load_codec
from sequence_multiscale_ensemble import sample_levels


ROOT = Path("artifacts/multiframe_early_fine_v1")
BASE = Path("artifacts/generator_curriculum_v1")
PILOT = Path("artifacts/early_time_fine_flow_v1")
ARMS = ["uniform", "early"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed multi-frame input: " + name)


def probe_seed(seed, protein, case_id, draw):
    return int(hashlib.sha256(f"multiframe-early-probe-v1:{seed}:{protein}:{case_id}:{draw}".encode()).hexdigest()[:8], 16)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh multi-frame namespace")
    p = copy.deepcopy(json.loads((BASE / "protocol.json").read_text())); verify(p)
    pilot = json.loads((PILOT / "assessment.json").read_text())
    if pilot["status"] != "complete_early_time_weighting_modestly_improves_selected_pair_not_ensemble_validation":
        raise RuntimeError("Missing assessed time-density pilot")
    old = torch.load(BASE / "labels.pt", map_location="cpu", weights_only=True)
    train = dict(train=old["train"], PDB_train=old["PDB_train"])
    development = dict(validation=old["validation"], reused_diagnostic={}, train={})
    if len(train["train"]) != 5 or len(train["PDB_train"]) != 899 or len(development["validation"]) != 2:
        raise RuntimeError("Unexpected fixed cohort")
    for cases in train["train"].values():
        for case in cases:
            check_case(case, "train")
    for case in train["PDB_train"]:
        check_case(case, "train")
    train_cases = train["PDB_train"] + [c for cases in train["train"].values() for c in cases]
    train_sequences = {c["sequence"] for c in train_cases}
    dev_sequences = {cases[0]["sequence"] for cases in development["validation"].values()}
    if train_sequences & dev_sequences:
        raise RuntimeError("Development sequence is in the training bank")
    all_features = p["feature_index"]
    index = json.loads(Path(all_features).read_text())
    scoped = {sequence: index[sequence] for sequence in sorted(train_sequences)}
    atomic_json(ROOT / "training_feature_index.json", scoped)
    torch.save(train, ROOT / "train_labels.pt")
    torch.save(development, ROOT / "development_labels.pt")
    p.update(arms=ARMS, updates=2048, batch_size=4, samples=64,
        lr=3e-4, min_lr=3e-5, feature_index=str(ROOT / "training_feature_index.json"),
        all_feature_index=all_features,
        checkpoints={str(s): str(BASE / "interleaved" / f"seed{s}" / "last.pt") for s in p["seeds"]},
        time_density={"uniform": "density 1 on [0,1]", "early": "density 2.5 on [0,.25], .5 on (.25,1]; fixed pilot setting, no scan"},
        PDB_events=6144, MD_events=2048,
        MD_training_frames={name: len(cases) for name, cases in train["train"].items()},
        PDB_probe_ids=p["pretrain_probe_ids"], probe_frames_per_MD_protein=16, probe_draws_per_frame=2,
        training="Continue each original 12288-update parent fine model with a fresh AdamW. Every update: three uniform static PDB training examples and one uniform choice of MD training protein/frame. Same case draws, Gaussian/uniform draws, native coarse/fine pairs, LR and updates in both arms; only sampled time differs. No extra geometry loss, quantization or structure-map update.",
        architecture="A separate immutable copy of the original parent samples coarse; the adapted copy samples continuous fine. Both use the same scoped frozen AA-only ESM3 features. Fine's coarse output head is frozen and unused. Stored generator weights are duplicated, but coarse/fine inference remains Euler16+16.",
        feature_scope="During optimization the feature bank contains only training sequences. The two development AA sequences and their fixed shuffled-AA sensitivity controls are attached after the final optimizer step; no native structure enters the full AA-only outlet.",
        matching="Identical trainable parameter count and initial weights, 2048 updates, batches, base Gaussian/uniform draws, LR and fine forward calls. Uniform and early are newly run under this multi-frame protocol. The earlier geometry study had a different LR and an extra geometry graph, so it is not the matched control here.",
        runtime_note="Record elapsed time and host for each run. Cached parent timing can come from another host and is descriptive; no training or inference speedup claim follows from cross-host timings. Inference architecture and Euler16+16 call counts stay fixed.",
        primary="64 retained raw full AA-only samples on each of two repeatedly reused development proteins, two seeds. Compare parent, uniform continuation and early continuation using nearest-reference error, reference coverage, geometry, fixed-bin JS, ensemble spread and runtime. Native-coarse outlets and training-frame probes are diagnostics.",
        diagnostics="Before and after training: native-coarse fine reconstruction on 16 fixed equally spaced frames per MD training protein, two Gaussian draws each, plus eight fixed static PDB training cases. This is a selected training diagnostic, not held-out or thermodynamic evidence.",
        scope="Exploratory multi-frame MD fine training with static PDB replay. Five MD training proteins and two repeatedly reused development sequences. No independent test or R3 opening; ESM3 pretraining exposure is unknown.",
        stop_rule="Exactly 2048 updates, final checkpoint only; no LR/time-density sweep, extension, best checkpoint or sample filtering. Keep all declared proteins, seeds and samples. Do not promote automatically or claim calibrated thermodynamics.")
    codec = load_codec(p, "centers3")
    checked = []
    for name, cases in sorted(train["train"].items()):
        for case in [cases[0], cases[-1]]:
            c, f = codec.encode(case["target"])
            torch.testing.assert_close(c, case["coarse"], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(f, case["fine_continuous"], atol=1e-5, rtol=1e-5)
            checked.append(dict(protein=name, case_id=case["id"]))
    model = make_model(p, 2026)
    if set(model.fine_model.feature_bank) != train_sequences:
        raise RuntimeError("Feature bank is not scoped to training sequences")
    coarse_before = state_digest(model.coarse_model); codec_before = state_digest(codec)
    case = train["train"]["1b2s_E"][0]; aa = encode_aa(case["sequence"])
    key = probe_seed(2026, case["protein"], case["id"], 0)
    model.eval(); before = sample_levels(model, codec, aa, key)
    native_parent_c = codec.project_coarse(model.coarse_model.sample_coarse(aa, component_seed(key, "coarse"), 16), len(aa))
    native_parent_f = model.coarse_model.sample_fine(aa, native_parent_c, component_seed(key, "fine"), 16)
    torch.testing.assert_close(before[0], native_parent_c, atol=0, rtol=0)
    torch.testing.assert_close(before[1], native_parent_f, atol=0, rtol=0)
    model.train(); optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"])
    optimizer.zero_grad(set_to_none=True)
    loss, _ = fine_time_loss(model, aa, case["coarse"], case["fine_continuous"],
                             torch.Generator().manual_seed(714), "early")
    loss.backward(); gradient = model.fine_model.fine_head[-1].weight.grad
    if not torch.isfinite(loss) or gradient is None or not torch.isfinite(gradient).all() or gradient.norm() <= 0:
        raise RuntimeError("Missing real-checkpoint fine gradient")
    optimizer.step(); model.eval()
    torch.testing.assert_close(before[0], sample_levels(model, codec, aa, key)[0], atol=0, rtol=0)
    if state_digest(model.coarse_model) != coarse_before or state_digest(codec) != codec_before:
        raise RuntimeError("Integration changed a frozen component")
    p["trainable_parameters"] = sum(v.numel() for v in model.trainable_parameters())
    atomic_json(ROOT / "integration_check.json", dict(status="passed", native_MD_endpoints_checked=checked,
        training_feature_sequences=len(train_sequences), development_features_attached_during_training=False,
        epoch_zero_both_samplers_bitwise_replayed=True, isolated_fine_optimizer_steps=1,
        coarse_sampling_bitwise_unchanged_after_update=True, codec_unchanged=True,
        actual_trial_optimizer_steps=0, trainable_parameters=p["trainable_parameters"],
        existing_early_time_and_frozen_branch_tests_passed=10))
    paths = [Path(__file__), Path("upstream/scripts/multiframe_early_fine.sbatch"),
        Path("upstream/experiments/early_time_fine_flow.py"), Path("upstream/experiments/fine_geometry_flow.py"),
        Path("upstream/experiments/run_fine_geometry_flow.py"), Path("upstream/experiments/run_multisequence_ensemble_flow.py"),
        Path("upstream/experiments/run_sequence_condition_probe.py"),
        Path("upstream/tests/test_early_time_fine_flow.py"), Path("upstream/tests/test_fine_geometry_flow.py"),
        PILOT / "protocol.json", PILOT / "report.json", PILOT / "assessment.json",
        Path("artifacts/fine_state_coverage_v1/report.json"), BASE / "protocol.json", BASE / "labels.pt",
        ROOT / "train_labels.pt", ROOT / "development_labels.pt", ROOT / "training_feature_index.json",
        ROOT / "integration_check.json"]
    for seed in p["seeds"]:
        paths += [Path(p["checkpoints"][str(seed)]), BASE / "interleaved" / f"seed{seed}" / "evaluation.json"]
    p["source_sha256"].update({str(path): sha(path) for path in paths})
    atomic_json(ROOT / "protocol.json", p)
    print("MULTIFRAME_EARLY_READY", json.dumps(dict(MD=p["MD_training_frames"], PDB=899,
        trainable=p["trainable_parameters"], updates=p["updates"])), flush=True)


@torch.no_grad()
def conditional_probe(p, model, codec, groups, seed, path):
    model.eval(); rows = []; cases = []
    for name, frames in sorted(groups["train"].items()):
        for index in np.linspace(0, len(frames) - 1, p["probe_frames_per_MD_protein"], dtype=int):
            cases.append(("MD_train", frames[int(index)]))
    pdb = {c["id"]: c for c in groups["PDB_train"]}
    cases += [("PDB_train", pdb[name]) for name in p["PDB_probe_ids"]]
    for domain, case in cases:
        aa = encode_aa(case["sequence"]); c = case["coarse"]; fine = case["fine_continuous"]
        native = metrics(codec.decode(c, fine).numpy(), case["target"].numpy())
        for draw in range(p["probe_draws_per_frame"]):
            key = component_seed(probe_seed(seed, case["protein"], case["id"], draw), "fine")
            prediction = model.sample_fine(aa, c, key, p["fine_nfe"])
            bb = codec.decode(c, prediction)
            torch.testing.assert_close(pool_ca(bb), c, atol=2e-5, rtol=2e-5)
            error = metrics(bb.numpy(), case["target"].numpy())
            rows.append(dict(domain=domain, protein=case["protein"], case_id=case["id"], draw=draw,
                metrics=error, native_reconstruction=native,
                normalized_fine_rmse=float((model.fine_model.normalize_fine(prediction)
                    - model.fine_model.normalize_fine(fine)).square().mean().sqrt())))
    atomic_json(path, dict(status="complete", rows=rows,
        scope="Fixed TRAIN frames and supplied native coarse; matched-frame errors, not nearest-ensemble accuracy or independent validation."))


def attach_development_features(p, model, groups):
    index = json.loads(Path(p["all_feature_index"]).read_text()); extra = set()
    for cases in groups["validation"].values():
        sequence = cases[0]["sequence"]; aa = encode_aa(sequence)
        shuffled = aa[torch.randperm(len(aa), generator=torch.Generator().manual_seed(719))]
        extra.update([sequence, "".join(AA_ALPHABET[i] for i in shuffled.tolist())])
    bank = dict(model.fine_model.feature_bank)
    for sequence in sorted(extra):
        entry = index[sequence]
        if sha(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Changed development AA features")
        with np.load(entry["path"], allow_pickle=False) as z:
            values = torch.from_numpy(z["h"].copy()).float()
        if values.shape != (len(sequence), p["feature_dimension"]) or not torch.isfinite(values).all():
            raise RuntimeError("Invalid AA-only feature shape")
        bank[sequence] = values.detach()
    model.coarse_model.feature_bank = bank; model.fine_model.feature_bank = bank


def train(p, arm, seed):
    groups = torch.load(ROOT / "train_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, seed); codec = load_codec(p, "centers3")
    out = ROOT / arm / f"seed{seed}"; out.mkdir(parents=True, exist_ok=True)
    if (out / "initial.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh trial only; inspect partial work before resubmission")
    coarse_hash = state_digest(model.coarse_model); codec_hash = state_digest(codec)
    normalization = {k: v.clone() for k, v in model.fine_model.named_buffers()}
    params = model.trainable_parameters()
    atomic_json(out / "initial.json", dict(coarse_sha256=coarse_hash, fine_sha256=state_digest(model.fine_model),
        codec_sha256=codec_hash, trainable_parameters=sum(v.numel() for v in params),
        training_feature_sequences=len(model.fine_model.feature_bank)))
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="initial_training_diagnostic", updates=0, total=p["updates"]))
    print("MULTIFRAME_INITIAL_DIAGNOSTIC", arm, seed, flush=True)
    conditional_probe(p, model, codec, groups, seed, out / "initial_conditional_probe.json")
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed + 37247); trace = []; realized = hashlib.sha256(); times = []
    started = time.perf_counter()
    for step in range(p["updates"]):
        model.train(); batch = sample_training_batch(groups, rng)
        lr = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / (p["updates"] - 1)))
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True); value = 0.; draws = []
        ids = [dict(protein=c["protein"], case_id=c["id"], domain="PDB" if i < 3 else "MD") for i, c in enumerate(batch)]
        for slot, case in enumerate(batch):
            loss, terms = fine_time_loss(model, encode_aa(case["sequence"]), case["coarse"],
                case["fine_continuous"], torch.Generator().manual_seed(draw_seed(seed, step, slot)), arm)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite multi-frame fine loss")
            (loss / 4).backward(); value += float(loss.detach()) / 4
            draws.append(terms["draw_sha256"]); times.append(terms["time"])
        grad = torch.nn.utils.clip_grad_norm_(params, p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite multi-frame gradient")
        opt.step()
        identity = dict(batch=ids, base_draws=draws, lr=lr)
        realized.update(json.dumps(identity, sort_keys=True).encode())
        trace.append(dict(update=step + 1, loss=value, gradient_norm=float(grad), **identity))
        if (step + 1) % 128 == 0 or step + 1 == p["updates"]:
            progress = dict(arm=arm, seed=seed, updates=step + 1, total=p["updates"],
                phase="training", loss=value, seconds=time.perf_counter() - started)
            atomic_json(out / "progress.json", progress)
            print("MULTIFRAME_EARLY_TRAIN", json.dumps(progress), flush=True)
    training_seconds = time.perf_counter() - started
    atomic_json(out / "trace.json", trace)
    torch.save(dict(model=model.state_dict(), updates=p["updates"], arm=arm, seed=seed,
        protocol_sha256=sha(ROOT / "protocol.json")), out / "last.pt")
    if state_digest(model.coarse_model) != coarse_hash or state_digest(codec) != codec_hash:
        raise RuntimeError("Training changed a frozen component")
    for key, value in model.fine_model.named_buffers():
        torch.testing.assert_close(value, normalization[key], atol=0, rtol=0)
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="final_training_diagnostic", updates=p["updates"], total=p["updates"], training_seconds=training_seconds))
    conditional_probe(p, model, codec, groups, seed, out / "final_conditional_probe.json")
    development = torch.load(ROOT / "development_labels.pt", map_location="cpu", weights_only=True)
    attach_development_features(p, model, development)
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="development_evaluation", updates=p["updates"], total=p["updates"], training_seconds=training_seconds))
    model.eval(); evaluate_development(p, model, codec, development, out, seed)
    maximum = 0.
    for name in development["validation"]:
        with np.load(BASE / "interleaved" / f"seed{seed}" / name / "full.npz") as z:
            before = z["backbone"].copy()
        with np.load(out / name / "full.npz") as z:
            after = z["backbone"].copy()
        if before.shape != after.shape:
            raise RuntimeError("Raw development sample counts changed")
        for a, b in zip(before, after):
            ac, bc = pool_ca(torch.from_numpy(a)), pool_ca(torch.from_numpy(b))
            maximum = max(maximum, float((ac - bc).abs().max()))
            torch.testing.assert_close(ac, bc, atol=2e-5, rtol=2e-5)
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, seed=seed, updates=p["updates"],
        realized_training_identity_sha256=realized.hexdigest(), training_seconds=training_seconds,
        seconds_including_final_probe_and_development=time.perf_counter() - started,
        runtime_host=platform.node(), trainable_parameters=sum(v.numel() for v in params),
        mean_training_time=float(np.mean(times)), fraction_time_le_quarter=float(np.mean(np.array(times) <= .25)),
        coarse_weights_unchanged=True, codec_weights_unchanged=True, normalization_buffers_unchanged=True,
        maximum_parent_coarse_center_roundoff_A=maximum, raw_samples_per_development_protein=64,
        model_default_replaced=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(out / "progress.json", dict(arm=arm, seed=seed, phase="complete", updates=p["updates"], total=p["updates"], training_seconds=training_seconds))
    print("MULTIFRAME_EARLY_ARM_COMPLETE", arm, seed, flush=True)


def macro(evaluation, outlet):
    values = [r for r in evaluation["rows"] if r["split"] == "validation" and r["outlet"] == outlet]
    if len(values) != 2:
        raise RuntimeError("Both development proteins must be present")
    keys = ["mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A",
        "js_pairwise_distance_divergence_nats", "rmsf_mae_A", "mean_full_generation_seconds"]
    result = {key: float(np.mean([v[key] for v in values])) for key in keys}
    for key in ["cn_rate", "boundary_cn_rate", "ca_clashes_per_residue",
                "reference_ca_c_n_angle_mae_degrees", "reference_c_n_ca_angle_mae_degrees"]:
        result[key] = float(np.mean([v["geometry"][key] for v in values]))
    result["spread_ratio"] = float(np.mean([v["mean_pairwise_ca_rmsd_A"] / v["reference_mean_pairwise_ca_rmsd_A"] for v in values]))
    return result


def report(p):
    trials = []; pairing = []; probes = []; macro_rows = []
    for seed in p["seeds"]:
        trials.append(dict(arm="parent", seed=seed,
            evaluation=json.loads((BASE / "interleaved" / f"seed{seed}" / "evaluation.json").read_text())))
        identities = []; starts = []
        for arm in ARMS:
            out = ROOT / arm / f"seed{seed}"
            done = json.loads((out / "complete.json").read_text())
            trace = json.loads((out / "trace.json").read_text())
            if done["updates"] != p["updates"] or len(trace) != p["updates"]:
                raise RuntimeError("Incomplete fine continuation")
            for flag in ["coarse_weights_unchanged", "codec_weights_unchanged", "normalization_buffers_unchanged"]:
                if not done[flag]:
                    raise RuntimeError("A frozen component changed")
            identities.append(done["realized_training_identity_sha256"])
            starts.append(json.loads((out / "initial.json").read_text()))
            counts = {"PDB": 0, "MD": 0}; proteins = {}; unique_MD = set()
            for row in trace:
                if [v["domain"] for v in row["batch"]] != ["PDB", "PDB", "PDB", "MD"]:
                    raise RuntimeError("Batch domain ratio changed")
                for case in row["batch"]:
                    counts[case["domain"]] += 1
                    if case["domain"] == "MD":
                        proteins[case["protein"]] = proteins.get(case["protein"], 0) + 1
                        unique_MD.add((case["protein"], case["case_id"]))
            if counts != dict(PDB=p["PDB_events"], MD=p["MD_events"]) or set(proteins) != set(p["MD_training_frames"]):
                raise RuntimeError("Training exposure counts differ")
            trial = dict(arm=arm, seed=seed, complete=done, presentations=counts,
                MD_draws_by_protein=proteins, unique_MD_training_frames=len(unique_MD),
                evaluation=json.loads((out / "evaluation.json").read_text()))
            trials.append(trial)
            for phase in ["initial", "final"]:
                rows = json.loads((out / f"{phase}_conditional_probe.json").read_text())["rows"]
                for domain in ["MD_train", "PDB_train"]:
                    names = sorted({r["protein"] for r in rows if r["domain"] == domain})
                    per_protein = []
                    for name in names:
                        selected = [r for r in rows if r["domain"] == domain and r["protein"] == name]
                        per_protein.append(dict(protein=name, ca_rmsd_A=float(np.mean([r["metrics"]["ca_rmsd"] for r in selected])),
                            cn_rate=float(np.mean([r["metrics"]["cn_rate"] for r in selected])),
                            normalized_fine_rmse=float(np.mean([r["normalized_fine_rmse"] for r in selected]))))
                    probes.append(dict(arm=arm, seed=seed, phase=phase, domain=domain,
                        per_protein=per_protein, ca_rmsd_A=float(np.mean([r["ca_rmsd_A"] for r in per_protein])),
                        cn_rate=float(np.mean([r["cn_rate"] for r in per_protein]))))
        if len(set(identities)) != 1 or starts[0] != starts[1]:
            raise RuntimeError("Initial weights, examples, Gaussian/uniform inputs or LR differ")
        pairing.append(dict(seed=seed, initial_weights_and_training_identities_matched=True))
    for trial in trials:
        for outlet in ["full", "oracle_coarse"]:
            macro_rows.append(dict(arm=trial["arm"], seed=trial["seed"], outlet=outlet,
                                   **macro(trial["evaluation"], outlet)))
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", trials=trials, macro_per_seed=macro_rows,
        training_conditional_probes=probes, pairing=pairing, scope=p["scope"],
        same_raw_samples_retained=True, model_default_replaced=False,
        independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    lines = [p["scope"], p["matching"], "DEVELOPMENT"] + [json.dumps(r) for r in macro_rows]
    lines += ["TRAINING NATIVE-COARSE PROBES"] + [json.dumps(r) for r in probes]
    (ROOT / "result_summary.txt").write_text("\n".join(lines) + "\n")
    plot(trials)
    print("MULTIFRAME_EARLY_REPORT_COMPLETE", flush=True)


def plot(trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [dict(r, arm=t["arm"], seed=t["seed"]) for t in trials for r in t["evaluation"]["rows"]
            if r["split"] == "validation" and r["outlet"] == "full"]
    names = sorted({r["protein"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    configs = [("parent", -.23, "#777777", "Parent"),
               ("uniform", 0., "#c27b24", "Uniform fine training"),
               ("early", .23, "#24679e", "Early-weighted fine training")]
    metrics_to_plot = [(lambda r: r["mean_sample_nearest_reference_ca_rmsd_A"], "CA error to nearest reference (Å)"),
        (lambda r: 100 * r["geometry"]["cn_rate"], "C–N abnormal bonds (%)"),
        (lambda r: r["js_pairwise_distance_divergence_nats"], "Pairwise-distance JS divergence (nats)"),
        (lambda r: r["mean_pairwise_ca_rmsd_A"] / r["reference_mean_pairwise_ca_rmsd_A"], "Generated / reference ensemble spread")]
    for ax, (value, label) in zip(axes.flat, metrics_to_plot):
        for i, name in enumerate(names):
            for seed, marker, jitter in [(2026, "o", -.035), (2027, "^", .035)]:
                xs = []; ys = []
                for arm, offset, color, _ in configs:
                    row = next(r for r in rows if r["protein"] == name and r["seed"] == seed and r["arm"] == arm)
                    x, y = i + offset + jitter, value(row)
                    ax.scatter(x, y, color=color, marker=marker, s=40, zorder=3); xs.append(x); ys.append(y)
                ax.plot(xs, ys, color=".8", lw=.7)
        ax.set_xticks(range(len(names)), names); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
        if label.startswith("Generated"):
            ax.axhline(1., color=".5", ls="--", lw=1)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=label) for _, _, c, label in configs]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Multi-frame fine flow: full AA-only generation\nFrozen coarse sampler and structure maps · two reused development proteins", fontsize=12)
    folder = Path("paper/figures/multiframe_early_fine_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Multi_Frame_Early_Fine." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report":
            report(p)
        else:
            if args.arm not in ARMS or args.seed not in p["seeds"]:
                raise RuntimeError("Undeclared multi-frame trial")
            train(p, args.arm, args.seed)
