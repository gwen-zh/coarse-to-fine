"""Bounded coverage repair, sampler audit and same-cohort author comparison."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from coarse_coverage_repair import SOLVERS, integrate, replay_loss, update
from coarse_full_cohort_evaluation import evaluate, summarize_rows
from coarse_full_cohort_flow import batch_rng, identity_record, sample_training_batch
from coarse_generation_diagnostic import coarse_metrics, weighted_rmsd_matrix
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint, sha
from joint_sequence_predictor import encode_aa
from run_coarse_few_targets import make_model
from run_coarse_full_cohort import assert_frozen, model_cases, training_probe
from run_coarse_generation_diagnostic import project_centers
from run_coarse_target_scaling import verify
from run_fine_geometry_flow import state_digest
from run_multiframe_early_fine import attach_development_features
from run_multisequence_ensemble_flow import sample_seed
from run_sequence_multiscale_ensemble import load_codec


ROOT = Path("artifacts/coarse_coverage_repair_v1")
PARENT = Path("artifacts/coarse_full_cohort_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")


def load_json(path):
    return json.loads(Path(path).read_text())


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists(): raise RuntimeError("Fresh namespace required")
    p = copy.deepcopy(load_json(PARENT / "protocol.json")); verify(p)
    if load_json(PARENT / "assessment.json")["status"] != "complete_reviewed_training_gain_development_ensemble_degradation":
        raise RuntimeError("Review completed full-cohort continuation first")
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    dev = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    author_old = load_json("artifacts/development/discrete_flow_probe_v1/protocol.json")
    p.update(updates=32768, optimizer_updates=32768, milestones=[32768], cycle_updates=32768,
        PDB_events=98304, MD_events=32768, diagnostic_samples=16, solver_variants=SOLVERS,
        hypothesis="Static single-structure FM rehearsal may reinforce a point conditional distribution while adapting to very few MD sequences. Rehearse the frozen parent's PDB velocity field instead, preserving its stochastic function, and continue native-frame FM on MD. Neither this attribution nor improvement is assumed.",
        objective="Three PDB parent-velocity MSE losses plus one actual MD-frame FM loss, equally averaged. Same interpolation states, batches, base draws, LR and update count as the frozen full-cohort32768 control. PDB native coordinates locate interpolation states, but PDB velocity targets come from the frozen original parent. MD targets remain frame-specific native velocities.",
        teacher="Original12288-update parent coarse copy, never the131072-update continued model. Separate frozen parameters and original normalization. Teacher sees904 TRAIN AA sequences only.",
        training="Exactly32768 updates per seed, fresh original parent coarse initialization. Own fine, encoder, decoder, codebook and normalizers frozen. No development optimization, spread matching to development, sample rejection, weight scan or best-checkpoint selection.",
        numerical_audit="Read-only parent versus fixed131072 models. Sixteen paired raw coarse samples on five TRAIN MD proteins and two reused development proteins; Euler16, midpoint8(same16 NFE), Euler64(diagnostic64 NFE). Baseline Euler16 must replay saved coarse samples within original baseline tolerances. This separates numerical sensitivity from learned-field effects, without selecting a sampler on development.",
        matching="Training cases, base draws, LR and32768 updates match the cached full-cohort prefix. Objectives differ deliberately. Replay adds three frozen teacher forward calls per update; training FLOPs/runtime are NOT matched. Inference remains identical Euler16+16. No extra inference diversity noise is introduced.",
        learning_rate_schedule="One32768-update cosine3e-4 to3e-5, identical to the first cycle of the cached full-cohort control. No LR restart or adaptive extension.",
        exposure="Exactly98304 PDB rehearsal events and32768 native MD-frame events per seed, with replacement. Case/noise/time/LR event identities must match the cached static-FM32768 control.",
        baseline_check="Initial coarse/fine/codec hashes must equal original full-cohort step0. Initial numerical-audit samples replay saved parent/final coarse at atol2e-4A rtol2e-5. At the repaired endpoint, frozen native-coarse fine arrays replay original step0 at atol/rtol2e-5.",
        author={"generator": author_old["generator"], "decoder": author_old["decoder"],
                "coarse_steps": [25, 5], "primary_steps": 25, "samples": 64,
                "policy": "Unmodified released ddpm_sample coarse-to-fine path; original top-three fixed-ratio sampling and one-pass fine argmax. No LoRA/search/retuning.25steps primary,5steps secondary; both fixed before sampling. Original README documents5-step demo and25-step larger run.",
                "targets": [dict(protein=n, sequence=v[0]["sequence"], length=len(v[0]["sequence"])) for n, v in sorted(dev["validation"].items())],
                "scope": "Two repeatedly reused development proteins only; same64samples and128 reference frames as our models. Different pretraining and architecture, no matched training compute or independent superiority claim. Raw outputs only."},
        scope="Exploratory coverage repair on existing TRAIN data, fixed32768 control and parent, both seeds; same-cohort comparison to the released Hierarchical ConfGen on two reused development sequences. No reserved test or R3 access, no thermodynamic calibration.",
        stop_rule="One read-only audit per seed, one fixed32768-update repair per seed, both author budgets and all raw samples reported. No adaptive extension, selecting favorable budgets/checkpoints/proteins, or automatic model promotion.")
    checks = []
    for seed in p["seeds"]:
        m = make_model(p, model_cases(groups), seed).eval()
        expected = load_json(PARENT / f"seed{seed}/initial.json")
        if state_digest(m.coarse_model) != expected["coarse_sha256"] or state_digest(m.fine_model) != expected["fine_sha256"]:
            raise RuntimeError("Initial weights do not equal frozen control")
        teacher = copy.deepcopy(m.coarse_model).eval().requires_grad_(False)
        teacher_hash = state_digest(teacher); fine_hash = state_digest(m.fine_model)
        batch = sample_training_batch(groups, batch_rng(seed)); aa = encode_aa(batch[0]["sequence"])
        loss, _ = replay_loss(m.coarse_model, teacher, aa, batch[0]["coarse"], torch.Generator().manual_seed(61), "PDB")
        if float(loss.detach()) != 0.: raise RuntimeError("Parent rehearsal not initially exact")
        fine_before = m.sample_fine(aa, batch[0]["coarse"], 816, 16)
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        row = update(m, teacher, opt, batch, seed, 0, p)
        torch.testing.assert_close(fine_before, m.sample_fine(aa, batch[0]["coarse"], 816, 16), atol=0, rtol=0)
        if state_digest(teacher) != teacher_hash or state_digest(m.fine_model) != fine_hash:
            raise RuntimeError("Preflight altered frozen teacher/fine")
        checks.append(dict(seed=seed, initial_PDB_replay_loss=0., MD_flow_loss=row["MD_flow_loss"],
            trainable_parameters=sum(v.numel() for v in m.trainable_parameters()), fine_sampler_exact=True, actual_trial_updates=0))
    atomic_json(ROOT / "integration_check.json", dict(status="passed", discarded_optimizer_steps=2, checks=checks))
    paths = [Path(__file__), Path("upstream/experiments/coarse_coverage_repair.py"),
        Path("upstream/experiments/compare_hierarchical_ensemble.py"), Path("upstream/tests/test_coarse_coverage_repair.py"),
        Path("upstream/scripts/coarse_coverage_repair.sbatch"), Path("upstream/scripts/hierarchical_ensemble_comparison.sbatch"),
        ROOT / "integration_check.json", PARENT / "protocol.json", PARENT / "assessment.json", PARENT / "review.json", PARENT / "report.json",
        Path("upstream/hierarchical_ConfGen/slm/models/model.py"), Path("upstream/hierarchical_ConfGen/slm/utils/checkpoint_utils.py"),
        Path("upstream/experiments/evaluate_residual_vq.py"), Path("upstream/experiments/train_pretrained_coarse.py")]
    for seed in p["seeds"]:
        paths += [PARENT / f"seed{seed}/last.pt", PARENT / f"seed{seed}/initial.json", PARENT / f"seed{seed}/trace.jsonl"]
        for n in [0, 131072]:
            for name in list(groups["train"]) + list(dev["validation"]):
                paths.append(PARENT / f"seed{seed}/step{n}/ensembles/{name}/full.npz")
        for name in dev["validation"]:
            paths.append(PARENT / f"seed{seed}/step0/ensembles/{name}/oracle_coarse.npz")
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    for f in [p["author"]["generator"], p["author"]["decoder"]]:
        stat = Path(f["path"]).stat()
        if stat.st_size != f["size"] or stat.st_mtime_ns != f["mtime_ns"]:
            raise RuntimeError("Original author weights changed")
        p["source_sha256"][f["path"]] = f["sha256"]
    atomic_json(ROOT / "protocol.json", p)
    print("COVERAGE_REPAIR_READY", json.dumps(checks), flush=True)


@torch.no_grad()
def diagnose(p, seed):
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    dev = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, model_cases(groups), seed).eval()
    attach_development_features(p, model, dev)
    out = ROOT / f"seed{seed}/solver_audit"; out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists(): raise RuntimeError("Audit already complete")
    rows = []; started = time.perf_counter(); max_replay = 0.
    for checkpoint in ["parent", "continued131072"]:
        if checkpoint == "continued131072":
            saved = torch.load(PARENT / f"seed{seed}/last.pt", map_location="cpu", weights_only=False)
            if saved["updates"] != 131072: raise RuntimeError("Wrong audit endpoint")
            model.load_state_dict(saved["model"]); del saved
        for split, by_name in [("MD_train", groups["train"]), ("development", dev["validation"])]:
            for name, cases in sorted(by_name.items()):
                refs = [cases[int(i)] for i in np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)]
                reference = np.stack([c["coarse"].numpy() for c in refs]); aa = encode_aa(cases[0]["sequence"])
                cached_step = 0 if checkpoint == "parent" else 131072
                with np.load(PARENT / f"seed{seed}/step{cached_step}/ensembles/{name}/full.npz") as z:
                    cached = z["coarse"].copy()[:p["diagnostic_samples"]]
                values = {}
                for solver in p["solver_variants"]:
                    clock = time.perf_counter()
                    generated = np.stack([project_centers(integrate(model.coarse_model, aa,
                        component_seed(sample_seed(seed, name, i), "coarse"), solver["steps"], solver["method"]), len(aa)).numpy()
                        for i in range(p["diagnostic_samples"])])
                    elapsed = time.perf_counter() - clock
                    if not np.isfinite(generated).all(): raise RuntimeError("Nonfinite solver diagnostic sample")
                    if solver["name"] == "euler16":
                        np.testing.assert_allclose(generated, cached, atol=2e-4, rtol=2e-5)
                        max_replay = max(max_replay, float(np.max(np.abs(generated - cached))))
                    stats = coarse_metrics(generated, reference, len(aa), p["ca_step_cap_A"])
                    stats.update(solver=solver["name"], nfe=solver["nfe"], seconds=elapsed)
                    values[solver["name"]] = stats
                    np.savez_compressed(out / f"{checkpoint}_{name}_{solver['name']}.npz", coarse=generated)
                rows.append(dict(checkpoint=checkpoint, split=split, protein=name, solvers=values))
                atomic_json(out / "progress.json", dict(completed=len(rows), total=14, seconds=time.perf_counter() - started))
                print("COVERAGE_SOLVER_AUDIT", seed, checkpoint, name, flush=True)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, rows=rows,
        maximum_original_sampler_roundoff_A=max_replay, seconds=time.perf_counter() - started,
        training_performed=False, sample_selection=False))


def train(p, seed, resume):
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    model = make_model(p, model_cases(groups), seed)
    teacher = copy.deepcopy(model.coarse_model).eval().requires_grad_(False)
    codec = load_codec(p, "centers3")
    expected = load_json(PARENT / f"seed{seed}/initial.json")
    fine_hash = state_digest(model.fine_model); codec_hash = state_digest(codec); teacher_hash = state_digest(teacher)
    if teacher_hash != expected["coarse_sha256"] or fine_hash != expected["fine_sha256"] or codec_hash != expected["codec_sha256"]:
        raise RuntimeError("Wrong repair initialization or frozen maps")
    buffers = {k: v.clone() for k, v in model.coarse_model.named_buffers()}
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = batch_rng(seed); out = ROOT / f"seed{seed}/repair"; out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists(): raise RuntimeError("Repair already complete")
    trace = []; start = 0; training_seconds = 0.
    if resume:
        saved = restore_checkpoint(out / "last.pt", model, opt, rng, torch.device("cpu"))
        if saved["protocol_sha256"] != sha(ROOT / "protocol.json") or saved["seed"] != seed:
            raise RuntimeError("Wrong resume metadata")
        start = saved["updates"]; training_seconds = saved["training_seconds"]
        restore_trace_prefix(out, start)
        trace = [json.loads(s) for s in (out / "trace.jsonl").read_text().splitlines()]
    else:
        if (out / "initial.json").exists(): raise RuntimeError("Explicit resume required")
        atomic_json(out / "initial.json", dict(coarse_sha256=teacher_hash, fine_sha256=fine_hash,
            codec_sha256=codec_hash, control_initialization_exact=True, training_feature_sequences=904))
    writer = CompactTrainingTrace(p["updates"], already_written=len(trace)); identity = hashlib.sha256()
    for row in trace: identity.update(json.dumps(identity_record(row), sort_keys=True).encode())
    if resume and identity.hexdigest() != saved["realized_training_identity_sha256"]:
        raise RuntimeError("Resume trace prefix changed")
    clock = time.perf_counter(); session_start = start
    for step in range(start, p["updates"]):
        row = update(model, teacher, opt, sample_training_batch(groups, rng), seed, step, p)
        trace.append(row); identity.update(json.dumps(identity_record(row), sort_keys=True).encode())
        if (step + 1) % p["progress_every"] == 0:
            writer(out / "trace.json", trace)
            progress = dict(seed=seed, phase="training", updates=step + 1, total=p["updates"],
                PDB_replay_loss=row["PDB_replay_loss"], MD_flow_loss=row["MD_flow_loss"],
                training_seconds=training_seconds + time.perf_counter() - clock, session_updates=step + 1 - session_start)
            atomic_json(out / "progress.json", progress); print("COVERAGE_REPAIR_TRAIN", json.dumps(progress), flush=True)
        if (step + 1) % p["checkpoint_every"] == 0 or step + 1 == p["updates"]:
            assert_frozen(model, codec, fine_hash, codec_hash, buffers)
            if state_digest(teacher) != teacher_hash: raise RuntimeError("Parent teacher changed")
            save_checkpoint(out / "last.pt", model, opt, rng, seed=seed, updates=step + 1,
                training_seconds=training_seconds + time.perf_counter() - clock, protocol_sha256=sha(ROOT / "protocol.json"),
                realized_training_identity_sha256=identity.hexdigest())
    training_seconds += time.perf_counter() - clock
    expected_identity = next(a for a in load_json(PARENT / "report.json")["exposure_audits"] if a["seed"] == seed)["prefix_identity_sha256"]["32768"]
    if identity.hexdigest() != expected_identity: raise RuntimeError("Data/noise/LR do not match original32768 control")
    atomic_json(out / "progress.json", dict(seed=seed, phase="evaluation", updates=p["updates"], total=p["updates"]))
    with torch.no_grad():
        training_probe(p, model, groups, out / "training_probe", seed)
        dev = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
        bank = model.coarse_model.feature_bank
        try:
            attach_development_features(p, model, dev)
            evaluation = evaluate(p, model, codec, {"train": groups["train"], "validation": dev["validation"]}, out / "ensembles", seed)
        finally:
            model.coarse_model.feature_bank = bank; model.fine_model.feature_bank = bank
    maximum = 0.
    for name in dev["validation"]:
        with np.load(out / f"ensembles/{name}/oracle_coarse.npz") as z: after = z["backbone"].copy()
        with np.load(PARENT / f"seed{seed}/step0/ensembles/{name}/oracle_coarse.npz") as z: before = z["backbone"].copy()
        np.testing.assert_allclose(after, before, atol=2e-5, rtol=2e-5)
        maximum = max(maximum, float(np.max(np.abs(after - before))))
    assert_frozen(model, codec, fine_hash, codec_hash, buffers)
    if state_digest(teacher) != teacher_hash: raise RuntimeError("Frozen teacher changed")
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, updates=p["updates"],
        training_seconds=training_seconds, runtime_host=platform.node(),
        realized_training_identity_sha256=identity.hexdigest(), matched_control_cases_draws_lr=True,
        maximum_frozen_fine_roundoff_A=maximum, fine_codec_teacher_frozen=True,
        evaluation=str(out / "ensembles/evaluation.json"), model_default_replaced=False))
    atomic_json(out / "progress.json", dict(seed=seed, phase="complete", updates=p["updates"], total=p["updates"]))


def report(p):
    rows = []; audits = []; numerical = []
    old = load_json(PARENT / "report.json")
    for seed in p["seeds"]:
        audit = load_json(ROOT / f"seed{seed}/solver_audit/complete.json"); numerical.append(audit)
        done = load_json(ROOT / f"seed{seed}/repair/complete.json")
        author = load_json(ROOT / f"seed{seed}/author/complete.json")
        if not done["fine_codec_teacher_frozen"] or not done["matched_control_cases_draws_lr"]:
            raise RuntimeError("Invalid repair audit")
        trace = [json.loads(s) for s in (ROOT / f"seed{seed}/repair/trace.jsonl").read_text().splitlines()]
        identity = hashlib.sha256()
        if len(trace) != p["updates"] or [r["update"] for r in trace] != list(range(1, p["updates"] + 1)):
            raise RuntimeError("Incomplete replay trace")
        for r in trace: identity.update(json.dumps(identity_record(r), sort_keys=True).encode())
        expected = next(a for a in old["exposure_audits"] if a["seed"] == seed)["prefix_identity_sha256"]["32768"]
        if identity.hexdigest() != expected or identity.hexdigest() != done["realized_training_identity_sha256"]:
            raise RuntimeError("Replay/control prefix identities differ")
        audits.append(done)
        for method, evaluation in [("parent", next(t["evaluation"] for t in old["trials"] if t["seed"] == seed and t["step"] == 0)),
            ("static_FM32768", next(t["evaluation"] for t in old["trials"] if t["seed"] == seed and t["step"] == 32768)),
            ("PDB_replay_MD_FM32768", load_json(done["evaluation"])),
            ("author", author)]:
            for x in evaluation["rows"]:
                if x["split"] == "development" and x["outlet"] == "full":
                    rows.append(dict(method=method if method != "author" else "author_steps" + str(x["coarse_steps"]), seed=seed, **x))
    summaries = []
    for method in sorted({r["method"] for r in rows}):
        for seed in p["seeds"]:
            group = [r for r in rows if r["method"] == method and r["seed"] == seed]
            if len(group) != 2: raise RuntimeError("Incomplete same-cohort comparison")
            summaries.append(dict(method=method, seed=seed, **summarize_rows(group)))
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, summaries=summaries,
        numerical_audits=numerical, training_audits=audits, scope=p["scope"],
        independent_test_claim=False, model_default_replaced=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_review", scope=p["scope"],
        next="Review every seed, protein and fixed method before interpreting effects. No automatic promotion."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ["author_steps25", "author_steps5", "parent", "static_FM32768", "PDB_replay_MD_FM32768"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, scale, title in zip(axes,
        ["mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A", "cn_rate"],
        [1., 1., 100.], ["Generated to reference RMSD (A)", "Reference coverage error (A)", "C-N abnormalities (%)"]):
        for j, method in enumerate(methods):
            raw = [r[key] for r in summaries if r["method"] == method]
            if any(x is None for x in raw): continue
            v = [scale * x for x in raw]
            ax.bar(j, np.mean(v), color="#3877a3" if "replay" in method else "#999999", alpha=.7)
            ax.scatter([j - .08, j + .08], v, c="black", s=17)
        ax.set_xticks(range(len(methods)), ["Author25", "Author5", "Our parent", "Static FM", "PDB replay"], rotation=25)
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Same2 reused development proteins /64 raw samples each /2 seeds\nDifferent pretraining and training compute; no independent superiority claim")
    dest = Path("paper/figures/coarse_coverage_repair_v1"); dest.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png", "svg"]: fig.savefig(dest / ("Coverage_Repair_and_Hierarchical_Comparison." + suffix), dpi=170)
    plt.close(fig); print("COVERAGE_REPAIR_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "diagnose", "train", "author", "report"], required=True)
    parser.add_argument("--seed", type=int, choices=[2026, 2027]); parser.add_argument("--resume", action="store_true")
    a = parser.parse_args(); torch.set_num_threads(2)
    if a.stage == "prepare": prepare()
    else:
        protocol = load_json(ROOT / "protocol.json"); verify(protocol)
        if a.stage == "diagnose": diagnose(protocol, a.seed)
        elif a.stage == "train": train(protocol, a.seed, a.resume)
        elif a.stage == "author":
            from compare_hierarchical_ensemble import run
            run(protocol, ROOT, LABELS, a.seed)
        else: report(protocol)
