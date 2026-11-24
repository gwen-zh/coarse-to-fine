"""Bounded full-cohort coarse-only continuation with native MD supervision."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from coarse_few_target_flow import draw_seed
from coarse_full_cohort_evaluation import evaluate, summarize_rows
from coarse_full_cohort_flow import batch_rng, identity_record, sample_training_batch, update
from coarse_generation_diagnostic import scalar_features, weighted_rmsd_matrix
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint, sha
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case
from run_coarse_few_targets import make_model
from run_coarse_generation_diagnostic import project_centers
from run_coarse_target_scaling import verify
from run_fine_geometry_flow import state_digest
from run_multiframe_early_fine import attach_development_features
from run_sequence_multiscale_ensemble import load_codec


ROOT = Path("artifacts/coarse_full_cohort_v1")
PARENT = Path("artifacts/coarse_target_scaling_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")
FINE = Path("artifacts/multiframe_coarse_context_v1")


def model_cases(groups):
    return groups["PDB_train"] + [v[0] for _, v in sorted(groups["train"].items())]


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists(): raise RuntimeError("Use a fresh full-cohort namespace")
    p = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text())); verify(p)
    assessment = json.loads((PARENT / "assessment.json").read_text())
    if assessment["status"] != "complete_nested32_coarse_trainability_survives_matched_target_exposure":
        raise RuntimeError("Assess the nested32-target study first")
    probe_ids = list(p["case_ids"])
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    md_frames = {name: len(v) for name, v in groups["train"].items()}
    if len(groups["PDB_train"]) != 899 or len(md_frames) != 5 or sum(md_frames.values()) != 2264:
        raise RuntimeError("Unexpected full training cohort")
    for c in groups["PDB_train"]:
        check_case(c, "train")
        if not torch.isfinite(c["coarse"]).all(): raise RuntimeError("Nonfinite PDB coarse label")
    for frames in groups["train"].values():
        if len({c["sequence"] for c in frames}) != 1: raise RuntimeError("MD sequence mapping changed")
        for c in frames:
            check_case(c, "train")
            if not torch.isfinite(c["coarse"]).all(): raise RuntimeError("Nonfinite MD coarse label")
    exemplars = model_cases(groups)
    if len({c["sequence"] for c in exemplars}) != 904: raise RuntimeError("Expected904 training AA sequences")
    for key in ["arms", "cohorts", "rounds", "common_case_ids", "added_case_ids", "case_ids", "max_updates", "coarse_diagnostic_nfe", "initial_metric_tolerance"]:
        p.pop(key, None)
    p.update(updates=131072, optimizer_updates=131072, milestones=[32768, 131072], cycle_updates=32768,
        checkpoint_every=4096, progress_every=512, batch_size=4, lr=3e-4, min_lr=3e-5,
        samples=64, train_diagnostic_samples=64, coarse_nfe=16, fine_nfe=16,
        PDB_probe_ids=probe_ids, PDB_probe_samples=8, teacher_times=[.1, .5, .9], teacher_draws=2,
        MD_training_frames=md_frames, PDB_training_structures=899, training_AA_sequences=904,
        PDB_events=393216, MD_events=131072,
        hypothesis="Eight/32-target coarse point fitting succeeds under sufficient per-target exposure. Test whether a bounded full-cohort coarse-only continuation improves training fit, raw sequence-conditioned development accuracy, geometry or ensemble descriptors without adapting fine.",
        architecture="Original-parent coarse network in its own trainable copy, same frozen geometry-aware fine candidate, frozen own structure encoder/decoder/codebook and original MD-train normalization. Continuous centers3 and continuous fine8, no VQ/snap. Coarse and fine inference remain Euler16+16.",
        training="Fresh original12288-update parent coarse in each seed, not the memorized8/32-target weights. Every update draws3 uniform static PDB targets and1 uniformly chosen MD-training protein and an actual cached frame from that protein. Native frame-specific centers are supervised directly; same-AA frames are NOT averaged. No new MD files, test or R3 data are added.",
        objective="Uniform-time normalized coarse flow velocity MSE, mean of four example losses. No fine objective, geometry loss, decoder update, new sequence feature extraction or sample rejection.",
        learning_rate_schedule="Budget-independent32768-update cosine3e-4 to3e-5, four cycles with AdamW moments retained. Report fixed32768/131072 endpoints, no reset of optimizer at evaluation. More updates add exposure and LR cycles; this is not a pure constant-LR budget contrast.",
        exposure="At32768 updates:98304 PDB and32768 MD examples. At131072:393216 PDB and131072 MD examples. PDB targets and MD protein/frame choices are with replacement; audit actual per-target and per-protein counts. These are not the per-target exposure-matched budgets of the earlier8/32 experiment.",
        matching="Paired uncontinued-parent versus fixed continuation endpoints, same fine weights, structure maps, original training features/normalization, AA-only sampling seeds, inference NFE and evaluation cohorts. The practical intervention combines more training with a coarse-only objective; it does not isolate data volume, capacity, ordering or joint-loss interference, and is not matched training compute to the frozen baseline.",
        feature_scope="Optimization feature banks contain only904 training AA sequences. Development AA and shuffled-AA features are attached during endpoint evaluation and removed before optimization resumes. Full sampling receives only AA features and random seeds; native structures enter training supervision and evaluation/oracle diagnostics only.",
        evaluation="Parent plus both fixed endpoints. Five seen MD-training sequences and two repeatedly used development sequences,64 raw samples per sequence and128 equally spaced native references. Retain full, development native-coarse oracle and shuffled-AA OOD controls. Report per-protein accuracy, geometry, spread/coverage and physical-bin JS. Also report8 coarse samples on each preselected32 PDB training target and teacher-velocity errors on all899 static targets and fixed16-frame-per-protein MD probes.",
        primary="Exploratory raw AA-only development accuracy/coverage, C-N and angle geometry, ensemble distributions and per-protein spread, both seeds and both endpoints. Report all results and failures. Native-coarse fine output is a frozen diagnostic, not deployment. Training-target fitting alone is not an ensemble/new-sequence gain.",
        sampling_failures="Retain every requested sample. Undefined frames or nonfinite predictions receive raw NaN backbone placeholders and explicit failure codes. Any failure makes that sequence's full metrics unavailable; never compute a success-only average or fall back to a selected alternative. Coarse metrics may still be reported if all centers are finite. Unexpected shape/feature errors stop the run.",
        baseline_check="Initial development full/oracle coordinates must replay the saved frozen-fine candidate with atol2e-4 A and rtol2e-5. Later oracle coordinates must match this run's initial oracle with atol/rtol2e-5, while fine/codec weight hashes remain exact.",
        runtime_note="Use CPU2 threads per seed for this small1.086M-parameter coarse network; record actual elapsed time and host. Historical timing from other hosts is descriptive. No inference or training speedup claim.",
        scope="Full existing899-static/5-MD-protein TRAIN cohort; two repeatedly reused development proteins. No independent test, R3, new-sequence biological confirmation, thermodynamic calibration or author-model comparison. Static structures do not supply equilibrium ensembles.",
        stop_rule="Exactly131072 coarse-only updates, fixed32768/131072 evaluations and checkpointed recovery. No adaptive extension, best endpoint selection, changed data ratio, hyperparameter scan, filtering or automatic default promotion.")
    codec = load_codec(p, "centers3"); codec_hash = state_digest(codec)
    by_id = {c["id"]: c for c in groups["PDB_train"]}
    native_cases = [by_id[name] for name in probe_ids] + [c for frames in groups["train"].values() for c in [frames[0], frames[-1]]]
    for c in native_cases: torch.testing.assert_close(codec.encode_coarse(c["target"]), c["coarse"], atol=1e-5, rtol=1e-5)
    checks = []
    for seed in p["seeds"]:
        model = make_model(p, exemplars, seed).eval()
        expected = json.loads((PARENT / "eight" / f"seed{seed}" / "initial.json").read_text())
        if state_digest(model.coarse_model) != expected["coarse_sha256"] or state_digest(model.fine_model) != expected["fine_sha256"]:
            raise RuntimeError("Wrong full-cohort initialization")
        if len(model.coarse_model.feature_bank) != 904: raise RuntimeError("Wrong training feature bank")
        sample = next(iter(groups["train"].values()))[0]; aa = encode_aa(sample["sequence"])
        before = model.sample_fine(aa, sample["coarse"], 933, 16); fine_hash = state_digest(model.fine_model)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        rng = batch_rng(seed); started = time.perf_counter()
        for step in range(8): row = update(model, opt, sample_training_batch(groups, rng), seed, step, p)
        elapsed = time.perf_counter() - started
        torch.testing.assert_close(before, model.sample_fine(aa, sample["coarse"], 933, 16), atol=0, rtol=0)
        if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash: raise RuntimeError("Full-cohort preflight changed frozen maps")
        if state_digest(model.coarse_model) == expected["coarse_sha256"]: raise RuntimeError("Coarse preflight did not update weights")
        checks.append(dict(seed=seed, discarded_preflight_updates=8, coarse_gradient_norm=row["gradient_norm"],
            seconds_per_preflight_update=elapsed / 8, training_feature_sequences=904,
            trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),
            fine_sampler_bitwise_unchanged=True, actual_trial_optimizer_updates=0))
    atomic_json(ROOT / "integration_check.json", dict(status="passed", native_coarse_labels_checked=len(native_cases),
        training_labels_checked=899 + 2264, checks=checks, codec_unchanged=True, actual_trial_optimizer_updates=0))
    paths = [Path(__file__), Path("upstream/experiments/coarse_full_cohort_flow.py"),
        Path("upstream/experiments/coarse_full_cohort_evaluation.py"),
        Path("upstream/tests/test_coarse_full_cohort_flow.py"), Path("upstream/scripts/coarse_full_cohort.sbatch"),
        Path("upstream/experiments/compact_training_trace.py"), PARENT / "protocol.json", PARENT / "report.json", PARENT / "assessment.json",
        ROOT / "integration_check.json", LABELS / "train_labels.pt", LABELS / "development_labels.pt"]
    for seed in p["seeds"]:
        paths.append(PARENT / "eight" / f"seed{seed}" / "initial.json")
        for name in ["5w82_E", "5ydn_A"]:
            for outlet in ["full", "oracle_coarse"]: paths.append(FINE / "local_geometry" / f"seed{seed}" / name / (outlet + ".npz"))
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    atomic_json(ROOT / "protocol.json", p)
    print("FULL_COARSE_READY", p["milestones"], md_frames, checks, flush=True)


@torch.no_grad()
def training_probe(p, model, groups, out, seed):
    model.eval(); by_id = {c["id"]: c for c in groups["PDB_train"]}; sampled = []
    out.mkdir(parents=True, exist_ok=True)
    for name in p["PDB_probe_ids"]:
        case = by_id[name]; aa = encode_aa(case["sequence"]); length = len(aa)
        centers = np.stack([project_centers(model.sample_coarse(aa,
            component_seed(draw_seed(seed, case["id"], "evaluation", index), "coarse"), p["coarse_nfe"]), length).numpy()
            for index in range(p["PDB_probe_samples"])])
        finite = bool(np.isfinite(centers).all())
        values = dict(case_id=name, protein=case["protein"], metrics_available=finite,
            failed_samples=int((~np.isfinite(centers).all(axis=(1, 2))).sum()))
        if finite:
            errors = weighted_rmsd_matrix(centers, case["coarse"][None].numpy(), length)[:, 0]
            values.update(mean_center_rmsd_A=float(errors.mean()),
                conditional_span_flag_rate=float(scalar_features(centers, length, p["ca_step_cap_A"])["conditional_span_flag"].mean()))
        sampled.append(values); np.savez_compressed(out / (case["protein"] + ".npz"), centers=centers)
    teacher = []; selected = [("PDB_train", c) for c in groups["PDB_train"]]
    for name, frames in sorted(groups["train"].items()):
        selected += [("MD_train", frames[int(i)]) for i in np.linspace(0, len(frames) - 1, 16, dtype=int)]
    for domain, case in selected:
        aa = encode_aa(case["sequence"]); target = model.coarse_model.normalize_coarse(case["coarse"])
        scores = []
        for t in p["teacher_times"]:
            values = []
            for index in range(p["teacher_draws"]):
                rng = torch.Generator().manual_seed(draw_seed(seed, case["id"], "full_teacher", index))
                noise = torch.randn(target.shape, generator=rng)
                v = model.coarse_model.coarse_velocity(aa, (1 - t) * noise + t * target, t)
                values.append(float((v - (target - noise)).square().mean()))
            scores.append(dict(time=t, velocity_mse=float(np.mean(values))))
        teacher.append(dict(domain=domain, case_id=case["id"], protein=case["protein"],
            mean_velocity_mse=float(np.mean([v["velocity_mse"] for v in scores])), times=scores))
    result = dict(status="complete", sampled_PDB_training=sampled, teacher=teacher,
        scope="Fixed32 static TRAIN rollout probes; teacher errors on all899 static targets and16 native TRAIN frames per MD protein. Not independent biological validation.")
    atomic_json(out / "probe.json", result); return result


@torch.no_grad()
def evaluate_endpoint(p, model, codec, groups, seed, step, out):
    folder = out / f"step{step}"; folder.mkdir(exist_ok=True)
    digest = state_digest(model.coarse_model); protocol_hash = sha(ROOT / "protocol.json")
    if (folder / "endpoint.json").exists():
        cached = json.loads((folder / "endpoint.json").read_text())
        if cached["coarse_sha256"] != digest or cached["protocol_sha256"] != protocol_hash: raise RuntimeError("Wrong cached endpoint")
        return cached
    started = time.perf_counter(); training_probe(p, model, groups, folder / "training_probe", seed)
    development = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    evaluation_groups = dict(train=groups["train"], validation=development["validation"])
    train_bank = model.coarse_model.feature_bank
    try:
        attach_development_features(p, model, development)
        evaluate(p, model, codec, evaluation_groups, folder / "ensembles", seed)
    finally:
        model.coarse_model.feature_bank = train_bank; model.fine_model.feature_bank = train_bank
    maximum = 0.
    for name in development["validation"]:
        for outlet in (["full", "oracle_coarse"] if step == 0 else ["oracle_coarse"]):
            source = (FINE / "local_geometry" / f"seed{seed}" / name / (outlet + ".npz") if step == 0
                      else out / "step0" / "ensembles" / name / (outlet + ".npz"))
            with np.load(source) as z: before = z["backbone"].copy()
            with np.load(folder / "ensembles" / name / (outlet + ".npz")) as z: after = z["backbone"].copy()
            atol = 2e-4 if step == 0 else 2e-5
            np.testing.assert_allclose(after, before, atol=atol, rtol=2e-5)
            maximum = max(maximum, float(np.max(np.abs(after - before))))
    result = dict(step=step, training_probe=str(folder / "training_probe/probe.json"),
        evaluation=str(folder / "ensembles/evaluation.json"), coarse_sha256=digest,
        protocol_sha256=protocol_hash, evaluation_seconds=time.perf_counter() - started,
        maximum_baseline_or_frozen_oracle_roundoff_A=maximum)
    atomic_json(folder / "endpoint.json", result); return result


def assert_frozen(model, codec, fine_hash, codec_hash, buffers):
    if state_digest(model.fine_model) != fine_hash or state_digest(codec) != codec_hash: raise RuntimeError("Frozen component changed")
    for key, value in model.coarse_model.named_buffers(): torch.testing.assert_close(value, buffers[key], atol=0, rtol=0)
    if any(v.grad is not None for v in model.fine_model.parameters()): raise RuntimeError("Fine received a gradient")


def train(p, seed, resume=False):
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, model_cases(groups), seed); codec = load_codec(p, "centers3")
    fine_hash = state_digest(model.fine_model); codec_hash = state_digest(codec)
    buffers = {k: v.clone() for k, v in model.coarse_model.named_buffers()}
    out = ROOT / f"seed{seed}"; out.mkdir(exist_ok=True)
    if (out / "complete.json").exists(): raise RuntimeError("Full-cohort seed already complete")
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = batch_rng(seed); start = 0; training_seconds = 0.; endpoints = []
    if resume:
        if not (out / "last.pt").exists() or not (out / "initial.json").exists(): raise RuntimeError("No resumable checkpoint")
        saved = restore_checkpoint(out / "last.pt", model, opt, rng, torch.device("cpu"))
        if saved["protocol_sha256"] != sha(ROOT / "protocol.json") or saved["seed"] != seed: raise RuntimeError("Resume protocol mismatch")
        start = saved["updates"]; training_seconds = saved["training_seconds"]
        restore_trace_prefix(out, start)
        trace = [json.loads(v) for v in (out / "trace.jsonl").read_text().splitlines()]
        if (out / "endpoints.json").exists(): endpoints = [v for v in json.loads((out / "endpoints.json").read_text()) if v["step"] <= start]
    else:
        if (out / "initial.json").exists() or (out / "progress.json").exists(): raise RuntimeError("Use explicit --resume for partial work")
        trace = []
        atomic_json(out / "initial.json", dict(coarse_sha256=state_digest(model.coarse_model), fine_sha256=fine_hash,
            codec_sha256=codec_hash, trainable_parameters=sum(v.numel() for v in model.trainable_parameters()),
            training_feature_sequences=len(model.coarse_model.feature_bank)))
        atomic_json(out / "progress.json", dict(seed=seed, phase="initial_evaluation", updates=0, total=p["updates"]))
        endpoints.append(evaluate_endpoint(p, model, codec, groups, seed, 0, out)); atomic_json(out / "endpoints.json", endpoints)
    assert_frozen(model, codec, fine_hash, codec_hash, buffers)
    identity = hashlib.sha256()
    for row in trace: identity.update(json.dumps(identity_record(row), sort_keys=True).encode())
    if resume and identity.hexdigest() != saved["realized_training_identity_sha256"]: raise RuntimeError("Resume trace identity mismatch")
    if start in p["milestones"] and not any(v["step"] == start for v in endpoints):
        snapshot = out / f"checkpoint_{start}.pt"
        if not snapshot.exists() or sha(snapshot) != sha(out / "last.pt"):
            shutil.copyfile(out / "last.pt", snapshot)
        endpoint = evaluate_endpoint(p, model, codec, groups, seed, start, out)
        endpoint.update(training_seconds=training_seconds, realized_training_identity_sha256=identity.hexdigest())
        endpoints.append(endpoint); atomic_json(out / "endpoints.json", endpoints)
    writer = CompactTrainingTrace(p["updates"], already_written=start)
    session_started = time.perf_counter(); block_started = session_started
    for step in range(start, p["updates"]):
        row = update(model, opt, sample_training_batch(groups, rng), seed, step, p)
        identity.update(json.dumps(identity_record(row), sort_keys=True).encode()); trace.append(row)
        if (step + 1) % p["progress_every"] == 0:
            writer(out / "trace.json", trace)
            progress = dict(seed=seed, phase="training", updates=step + 1, total=p["updates"],
                loss=row["loss"], PDB_loss=row["PDB_loss"], MD_loss=row["MD_loss"],
                training_seconds=training_seconds + time.perf_counter() - block_started,
                session_updates=step + 1 - start)
            atomic_json(out / "progress.json", progress); print("FULL_COARSE_TRAIN", json.dumps(progress), flush=True)
        if (step + 1) % p["checkpoint_every"] == 0:
            assert_frozen(model, codec, fine_hash, codec_hash, buffers); writer(out / "trace.json", trace)
            save_checkpoint(out / "last.pt", model, opt, rng, updates=step + 1, seed=seed,
                protocol_sha256=sha(ROOT / "protocol.json"),
                training_seconds=training_seconds + time.perf_counter() - block_started,
                realized_training_identity_sha256=identity.hexdigest())
        if step + 1 in p["milestones"]:
            training_seconds += time.perf_counter() - block_started
            shutil.copyfile(out / "last.pt", out / f"checkpoint_{step + 1}.pt")
            atomic_json(out / "progress.json", dict(seed=seed, phase="endpoint_evaluation", updates=step + 1,
                total=p["updates"], training_seconds=training_seconds))
            endpoint = evaluate_endpoint(p, model, codec, groups, seed, step + 1, out)
            endpoint.update(training_seconds=training_seconds, realized_training_identity_sha256=identity.hexdigest())
            endpoints.append(endpoint); atomic_json(out / "endpoints.json", endpoints)
            block_started = time.perf_counter()
    assert_frozen(model, codec, fine_hash, codec_hash, buffers); verify(p)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, updates=p["updates"], endpoints=endpoints,
        training_seconds=training_seconds, last_session_seconds=time.perf_counter() - session_started,
        runtime_host=platform.node(), fine_and_codec_frozen=True, normalization_frozen=True,
        training_feature_sequences=904, resumed=resume, model_default_replaced=False,
        independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(out / "progress.json", dict(seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def audit_trace(p, seed, groups):
    out = ROOT / f"seed{seed}"; pdb = Counter(); md = Counter(); frames = set(); identity = hashlib.sha256(); prefix = {}
    known_PDB = {c["id"] for c in groups["PDB_train"]}; known_MD = {(c["protein"], c["id"]) for v in groups["train"].values() for c in v}
    index = 0
    with (out / "trace.jsonl").open() as f:
        for index, line in enumerate(f, start=1):
            row = json.loads(line)
            if row["update"] != index or [c["domain"] for c in row["batch"]] != ["PDB", "PDB", "PDB", "MD"]: raise RuntimeError("Bad training trace")
            for c in row["batch"]:
                if c["domain"] == "PDB":
                    if c["case_id"] not in known_PDB: raise RuntimeError("Unknown PDB target")
                    pdb[c["case_id"]] += 1
                else:
                    if (c["protein"], c["case_id"]) not in known_MD: raise RuntimeError("Unknown MD frame")
                    md[c["protein"]] += 1; frames.add((c["protein"], c["case_id"]))
            identity.update(json.dumps(identity_record(row), sort_keys=True).encode())
            if index in p["milestones"]: prefix[str(index)] = identity.hexdigest()
    if index != p["updates"] or sum(pdb.values()) != p["PDB_events"] or sum(md.values()) != p["MD_events"]: raise RuntimeError("Wrong exposure budget")
    return dict(seed=seed, PDB_draws=sum(pdb.values()), MD_draws=sum(md.values()),
        PDB_draws_by_target=dict(pdb), MD_draws_by_protein=dict(md), unique_MD_frames=len(frames), prefix_identity_sha256=prefix)


def report(p):
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    trials = []; summaries = []; probes = []; audits = []
    for seed in p["seeds"]:
        out = ROOT / f"seed{seed}"; done = json.loads((out / "complete.json").read_text()); audit = audit_trace(p, seed, groups)
        if done["updates"] != p["updates"] or [e["step"] for e in done["endpoints"]] != [0] + p["milestones"] or not done["fine_and_codec_frozen"]:
            raise RuntimeError("Missing full-cohort endpoint")
        audits.append(audit)
        for endpoint in done["endpoints"]:
            step = endpoint["step"]
            if step and endpoint["realized_training_identity_sha256"] != audit["prefix_identity_sha256"][str(step)]: raise RuntimeError("Endpoint data identity mismatch")
            evaluation = json.loads(Path(endpoint["evaluation"]).read_text())
            trials.append(dict(seed=seed, step=step, endpoint=endpoint, evaluation=evaluation))
            for split, outlet in [("MD_train", "full"), ("development", "full"), ("development", "oracle_coarse"), ("development", "scrambled_AA")]:
                rows = [r for r in evaluation["rows"] if r["split"] == split and r["outlet"] == outlet]
                if len(rows) != (5 if split == "MD_train" else 2): raise RuntimeError("Wrong evaluation cohort")
                summaries.append(dict(seed=seed, step=step, split=split, outlet=outlet, **summarize_rows(rows)))
            probe = json.loads(Path(endpoint["training_probe"]).read_text())
            sampled = probe["sampled_PDB_training"]; teachers = probe["teacher"]
            values = dict(seed=seed, step=step, PDB_rollout_failures=sum(v["failed_samples"] for v in sampled),
                mean_PDB_probe_center_rmsd_A=float(np.mean([v["mean_center_rmsd_A"] for v in sampled])) if all(v["metrics_available"] for v in sampled) else None,
                PDB_teacher_velocity_mse=float(np.mean([v["mean_velocity_mse"] for v in teachers if v["domain"] == "PDB_train"])),
                MD_teacher_velocity_mse=float(np.mean([np.mean([v["mean_velocity_mse"] for v in teachers if v["domain"] == "MD_train" and v["protein"] == name]) for name in sorted(groups["train"])])))
            probes.append(values)
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", trials=trials, summaries=summaries,
        training_probes=probes, exposure_audits=audits, scope=p["scope"], all_fixed_endpoints_reported=True,
        model_default_replaced=False, independent_test_claim=False, test_coordinates_opened=False, R3_opened=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_result_review", scope=p["scope"],
        next="Review both seeds/endpoints, per-protein ensemble coverage and failures alongside training fit before any next intervention. No automatic extension, independent-test access or default promotion.",
        model_default_replaced=False, independent_test_claim=False))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n\n" + "\n".join(json.dumps(v) for v in summaries) + "\n")
    plot(summaries, probes, trials)
    print("FULL_COARSE_COMPLETE", json.dumps(summaries), flush=True)


def plot(rows, probes, trials):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for seed, color, marker in [(2026, "#24679e", "o"), (2027, "#c27b24", "^")]:
        dev = sorted([r for r in rows if r["seed"] == seed and r["split"] == "development" and r["outlet"] == "full"], key=lambda r: r["step"])
        for col, (key, scale) in enumerate([("mean_sample_nearest_reference_ca_rmsd_A", 1.), ("cn_rate", 100.), ("js_pairwise_distance_divergence_nats", 1.)]):
            selected = [r for r in dev if r[key] is not None]
            axes[0, col].plot([r["step"] for r in selected], [scale * r[key] for r in selected], color=color, marker=marker, label=f"seed{seed}")
        pdb = sorted([r for r in probes if r["seed"] == seed], key=lambda r: r["step"])
        selected = [r for r in pdb if r["mean_PDB_probe_center_rmsd_A"] is not None]
        axes[1, 0].plot([r["step"] for r in selected], [r["mean_PDB_probe_center_rmsd_A"] for r in selected], color=color, marker=marker)
        for col, split in [(1, "MD_train"), (2, "development")]:
            selected = sorted([r for r in rows if r["seed"] == seed and r["split"] == split and r["outlet"] == "full" and r["coarse_mean_sample_nearest_reference_center_rmsd_A"] is not None], key=lambda r: r["step"])
            axes[1, col].plot([r["step"] for r in selected], [r["coarse_mean_sample_nearest_reference_center_rmsd_A"] for r in selected], color=color, marker=marker)
    for col, (title, label) in enumerate([("Development: full AA-only", "CA RMSD (Å)"), ("Development: full AA-only", "C–N abnormalities (%)"), ("Development: full AA-only", "Pair-distance JS (nats)")]):
        axes[0, col].set_title(title, fontsize=10); axes[0, col].set_ylabel(label)
    for col, title in enumerate(["32 static TRAIN probes", "5 MD TRAIN proteins", "2 reused development proteins"]):
        axes[1, col].set_title(title, fontsize=10); axes[1, col].set_ylabel("Weighted center RMSD (Å)")
    for ax in axes.flat:
        ax.set_xticks([0, 32768, 131072], ["Parent", "32,768", "131,072"]); ax.set_xlabel("Coarse-only updates"); ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Full-cohort coarse continuation:899 static targets +5 native-MD training proteins\nFine / structure maps frozen · fixed endpoints · no independent-test claim", fontsize=12)
    folder = Path("paper/figures/coarse_full_cohort_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]: fig.savefig(folder / ("Full_Cohort_Coarse." + ext), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for col, name in enumerate(["5w82_E", "5ydn_A"]):
        ax = axes[col]
        for seed, color, marker in [(2026, "#24679e", "o"), (2027, "#c27b24", "^")]:
            values = []
            for trial in trials:
                if trial["seed"] != seed: continue
                row = next(r for r in trial["evaluation"]["rows"] if r["protein"] == name and r["outlet"] == "full")
                if row["metrics_available"]: values.append((trial["step"], row["mean_pairwise_ca_rmsd_A"] / row["reference_mean_pairwise_ca_rmsd_A"]))
            values.sort(); ax.plot([v[0] for v in values], [v[1] for v in values], color=color, marker=marker, label=f"seed{seed}")
        ax.axhline(1., color=".5", ls="--"); ax.set_title(name); ax.set_ylabel("Generated/native CA spread ratio")
        ax.set_xticks([0, 32768, 131072], ["Parent", "32,768", "131,072"]); ax.set_xlabel("Coarse-only updates"); ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8); fig.suptitle("Keep opposite development spread errors separate", fontsize=12)
    for ext in ["pdf", "png", "svg"]: fig.savefig(folder / ("Full_Cohort_Ensemble_Spread." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--seed", type=int, default=2026); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare": prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report": report(p)
        else:
            if args.seed not in p["seeds"]: raise RuntimeError("Undeclared full-cohort seed")
            train(p, args.seed, args.resume)
