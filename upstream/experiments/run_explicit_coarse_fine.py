"""Fixed-budget matched adapter comparison on the existing two-state probes."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from diagnostic_io import atomic_json
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from fine_geometry_flow import fine_flow_loss
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_fine_single_structure import key_seed, make_model as parent_model
from run_fine_two_state import evaluate_pair
from run_sequence_multiscale_ensemble import load_codec

ROOT = Path("artifacts/explicit_coarse_fine_v1")
PARENT = Path("artifacts/fine_two_state_v1")
ARMS = ["raw_context", "local_geometry"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed explicit-context input: " + name)


def make_model(p, case, seed, arm):
    base = parent_model(p, case, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(key_seed(seed, case["sequence"], "explicit-coarse-adapter", 0))
        model = ExplicitCoarseFineFlow(**p["normalization"], **p["model"], coarse_dim=3,
            feature_mode="ESM3_context", feature_dim=p["feature_dimension"], context_kind=arm)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    expected = {"context_adapter.0.weight", "context_adapter.0.bias",
                "context_adapter.2.weight", "context_adapter.2.bias"}
    if set(missing) != expected or unexpected:
        raise RuntimeError("Base model parameters were not loaded exactly")
    model.attach_feature_bank(base.feature_bank)
    model.coarse_head.requires_grad_(False)
    return model, base


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh experiment namespace")
    p = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text()))
    verify(p)
    previous = json.loads((PARENT / "assessment.json").read_text())
    if previous["status"] != "complete_selected_two_state_learning_succeeds_for_one_protein_not_the_other":
        raise RuntimeError("Missing assessed two-state result")
    audit = Path("artifacts/fine_condition_path_v1/report.json")
    if not json.loads(audit.read_text())["model_weights_unchanged"]:
        raise RuntimeError("Missing read-only condition-path audit")
    p.update(arms=ARMS, updates=2048, batch_size=4, samples=8,
        training="Fresh original-parent fine weights. Same two native training frames per protein; same AA, noise/time draws, LR, AdamW settings and state slots as the prior two-state probe. Fine-only FM loss. No decoder/codebook/encoder or normalization updates.",
        matching="Two adapters have identical 19,712 added parameters, initial weights and zero output; both exactly reproduce the parent at epoch zero. raw_context inputs normalized XYZ padded with 21 zeros; local_geometry inputs the frozen decoder's 24 deterministic relative-center/frame features. Each adapter is added to the coarse prefix and broadcast to fine. Same Transformer calls, optimizer updates and example/noise identities; analytic geometry preprocessing adds runtime and is reported, not called free. Original no-adapter reference is the frozen prior run, not a new matched-capacity control.",
        primary="Selected two-state TRAIN learnability: report both proteins and both seeds, fine state assignment, decoded matched-state CA/C-N and fine-change magnitude/direction. No model promotion or biological generalization claim.",
        scope="Two selected native training pairs only; capacity-matched engineering comparison, not full-trajectory ensemble fidelity, AA-only generation, or unseen-sequence performance.",
        stop_rule="Exactly 2048 updates in both arms, no best checkpoint, extension, sample filtering, new test/R3 access or automatic model promotion.")
    pairs = torch.load(PARENT / "pairs.pt", map_location="cpu", weights_only=True)
    torch.save(pairs, ROOT / "pairs.pt")
    codec = load_codec(p, "centers3")
    checks = []
    for name, pair in pairs.items():
        for arm in ARMS:
            model, base = make_model(p, pair["a"], 2026, arm)
            model.eval(); base.eval()
            for case in [pair["a"], pair["b"]]:
                aa = encode_aa(case["sequence"])
                c, f = codec.encode(case["target"])
                torch.testing.assert_close(c, case["coarse"], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(f, case["fine_continuous"], atol=1e-5, rtol=1e-5)
                condition = model.normalize_coarse(c)
                if arm == "local_geometry":
                    torch.testing.assert_close(model.context_features(condition, len(aa)),
                        codec.context(c, len(aa))["geometry"], atol=2e-5, rtol=2e-5)
                with torch.no_grad():
                    a = model.fine_velocity(aa, f, .3, condition)
                    b = base.fine_velocity(aa, f, .3, condition)
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                model.zero_grad(set_to_none=True)
                loss, _ = fine_flow_loss(SimpleNamespace(fine_model=model), aa, c, f,
                    torch.Generator().manual_seed(714))
                loss.backward()
                g = model.context_adapter[-1].weight.grad
                if g is None or not torch.isfinite(g).all() or g.norm() <= 0:
                    raise RuntimeError("No adapter gradient on real native state")
                checks.append(dict(protein=name, arm=arm, case_id=case["id"], gradient_norm=float(g.norm())))
    atomic_json(ROOT / "integration_check.json", dict(status="passed", checks=checks,
        added_parameters=19712, exact_epoch_zero=True, optimizer_updates=0))
    sources = [Path(__file__), Path("upstream/experiments/explicit_coarse_fine_flow.py"),
        Path("upstream/tests/test_explicit_coarse_fine_flow.py"),
        Path("upstream/scripts/explicit_coarse_fine.sbatch"),
        PARENT / "protocol.json", PARENT / "report.json", PARENT / "assessment.json", audit,
        ROOT / "pairs.pt", ROOT / "integration_check.json"]
    p["source_sha256"].update({str(x): sha(x) for x in sources})
    atomic_json(ROOT / "protocol.json", p)
    print("EXPLICIT_CONTEXT_READY", len(checks), "real-state checks", flush=True)


def train(p, arm, name, seed):
    pair = torch.load(ROOT / "pairs.pt", map_location="cpu", weights_only=True)[name]
    cases = [pair["a"], pair["b"]]
    model, _ = make_model(p, cases[0], seed, arm)
    codec = load_codec(p, "centers3")
    out = ROOT / arm / name / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists() or (out / "progress.json").exists():
        raise RuntimeError("Fresh matched experiment only")
    for phase in ["initial", "final"]:
        (out / phase).mkdir(exist_ok=True)
    initial = evaluate_pair(p, model, codec, pair, seed, out / "initial")
    frozen = {k: v.clone() for k, v in codec.state_dict().items()}
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    params = [v for v in model.parameters() if v.requires_grad]
    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    aa = encode_aa(cases[0]["sequence"])
    adapter = SimpleNamespace(fine_model=model)
    trace = []; draw_hash = hashlib.sha256(); start = time.perf_counter()
    for step in range(p["updates"]):
        model.train(); opt.zero_grad(set_to_none=True); value = 0.
        lr = p["min_lr"] + .5 * (p["lr"] - p["min_lr"]) * (1 + math.cos(math.pi * step / (p["updates"] - 1)))
        for group in opt.param_groups:
            group["lr"] = lr
        for slot in range(p["batch_size"]):
            case = cases[(step + slot) % 2]
            rng = torch.Generator().manual_seed(key_seed(seed, name, "two-state-training", step * 4 + slot))
            loss, terms = fine_flow_loss(adapter, aa, case["coarse"], case["fine_continuous"], rng)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite matched adapter loss")
            draw_hash.update(terms["draw_sha256"].encode())
            (loss / p["batch_size"]).backward(); value += float(loss.detach()) / p["batch_size"]
        grad = torch.nn.utils.clip_grad_norm_(params, p["gradient_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError("Nonfinite matched adapter gradient")
        opt.step(); trace.append(value)
        if (step + 1) % 128 == 0 or step + 1 == p["updates"]:
            progress = dict(arm=arm, protein=name, seed=seed, updates=step + 1,
                total=p["updates"], loss=value, seconds=time.perf_counter() - start)
            atomic_json(out / "progress.json", progress)
            print("EXPLICIT_CONTEXT_TRAIN", json.dumps(progress), flush=True)
    training_seconds = time.perf_counter() - start
    torch.save(dict(model=model.state_dict(), arm=arm, protein=name, seed=seed,
        updates=p["updates"], protocol_sha256=sha(ROOT / "protocol.json")), out / "last.pt")
    final = evaluate_pair(p, model, codec, pair, seed, out / "final")
    for k, v in codec.state_dict().items():
        torch.testing.assert_close(v, frozen[k], atol=0, rtol=0)
    for k, v in model.named_buffers():
        torch.testing.assert_close(v, buffers[k], atol=0, rtol=0)
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", arm=arm, protein=name, seed=seed,
        initial=initial, final=final, trace=trace, training_seconds=training_seconds,
        seconds_including_final_evaluation=time.perf_counter() - start, realized_draw_sha256=draw_hash.hexdigest(),
        trainable_parameters=sum(v.numel() for v in params), codec_and_normalization_frozen=True,
        training_only=True, no_ensemble_claim=True, test_coordinates_opened=False, R3_opened=False))


def concise(row, phase):
    d = row[phase]
    values = [next(v for v in s["evaluation"]["rows"] if v["nfe"] == 16) for s in d["states"]]
    return dict(ca_rmsd_A=float(np.mean([v["metrics"]["ca_rmsd"] for v in values])),
        cn_rate=float(np.mean([v["metrics"]["cn_rate"] for v in values])),
        fine_state_accuracy=d["state_assignment_accuracy"],
        response_over_native=d["predicted_fine_change_rms"] / d["native_fine_change_rms"],
        response_direction_cosine=d["fine_change_direction_cosine"])


def report(p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for arm in ARMS:
        for name in p["proteins"]:
            for seed in p["seeds"]:
                row = json.loads((ROOT / arm / name / f"seed{seed}" / "complete.json").read_text())
                if len(row["trace"]) != p["updates"]:
                    raise RuntimeError("Incomplete adapter training")
                rows.append(row)
    baseline = json.loads((PARENT / "report.json").read_text())["rows"]
    for name in p["proteins"]:
        for seed in p["seeds"]:
            paired = [r for r in rows if r["protein"] == name and r["seed"] == seed]
            if len({r["realized_draw_sha256"] for r in paired}) != 1:
                raise RuntimeError("Noise/time draws did not match between adapters")
            original = next(r for r in baseline if r["protein"] == name and r["seed"] == seed)
            for row in paired:
                # Both zero adapters must start at the original parent, numerically.
                for key, value in concise(original, "initial").items():
                    if not np.isclose(concise(row, "initial")[key], value, rtol=2e-5, atol=2e-5):
                        raise RuntimeError("Initial evaluation differs from old parent: " + key)
    verify(p)
    summary = []
    for arm, selected in [("no_adapter_reference", baseline)] + [(a, [r for r in rows if r["arm"] == a]) for a in ARMS]:
        for name in p["proteins"]:
            group = [concise(r, "final") for r in selected if r["protein"] == name]
            summary.append(dict(arm=arm, protein=name, **{
                k: float(np.mean([v[k] for v in group])) for k in group[0]}))
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, summary=summary,
        scope=p["scope"], matching=p["matching"], realized_draws_matched=True,
        training_only=True, no_ensemble_claim=True, model_default_replaced=False,
        test_coordinates_opened=False, R3_opened=False))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    colors = ["#777777", "#d4852d", "#24679e"]
    labels = ["Existing interface", "Added raw XYZ path", "Added local geometry"]
    for i, name in enumerate(p["proteins"]):
        group = [r for r in summary if r["protein"] == name]
        for j, (key, scale, ylabel) in enumerate([
            ("fine_state_accuracy", 100., "Correct fine-state assignment (%)"),
            ("ca_rmsd_A", 1., "Matched-state CA RMSD (Å)"),
            ("cn_rate", 100., "C–N abnormalities (%)")]):
            ax = axes[i, j]
            ax.bar(range(3), [r[key] * scale for r in group], color=colors)
            ax.set_xticks(range(3), ["Existing", "Raw XYZ", "Local geom."], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9); ax.set_title(name, fontsize=11)
            ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylim(0, 110); ax.axhline(50, color=".5", ls="--", lw=1)
    fig.suptitle("Explicit coarse conditioning: two selected training pairs, two seeds\nNative coarse supplied; no new-sequence or ensemble-quality claim", fontsize=12)
    folder = Path("paper/figures/explicit_coarse_fine_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Explicit_Coarse_Conditioning." + ext), dpi=180)
    plt.close(fig)
    lines = [p["scope"], p["matching"], ""] + [json.dumps(r) for r in summary]
    (ROOT / "result_summary.txt").write_text("\n".join(lines) + "\n")
    print("EXPLICIT_CONTEXT_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "report"], required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--protein", choices=["1b2s_E", "1ef1_D"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report":
            report(p)
        else:
            if args.seed not in p["seeds"] or args.protein not in p["proteins"] or args.arm not in ARMS:
                raise RuntimeError("Undeclared matched trial")
            train(p, args.arm, args.protein, args.seed)
