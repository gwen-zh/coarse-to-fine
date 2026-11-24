"""Uniform versus early-weighted fine FM; same frozen two-state train probes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from diagnostic_io import atomic_json
from early_time_fine_flow import fine_time_loss
from fine_geometry_flow import fine_flow_loss
from hierarchical_pipeline_io import sha
from joint_sequence_predictor import encode_aa
from run_explicit_coarse_fine import concise
from run_fine_single_structure import make_model
import run_fine_two_state as engine

ROOT = Path("artifacts/early_time_fine_flow_v1")
PARENT = Path("artifacts/fine_two_state_v1")
ARMS = ["uniform", "early"]


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed time-density input: " + name)


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh time-density namespace")
    p = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text()))
    verify(p)
    audit = Path("artifacts/fine_state_difference_v1/report.json")
    if not json.loads(audit.read_text())["exact_decomposition_checked_all_draws"]:
        raise RuntimeError("State-difference audit must be complete")
    p.update(arms=ARMS,
        training="Reuse the frozen original two-state training loop, fresh original-parent weights, 2048 updates and batch4. Network, native coarse/fine labels, state slots, Gaussian/uniform draws, optimizer, LR, structural maps and normalization are unchanged. Only transform sampled time in the early arm.",
        time_density={"uniform": "density 1 on [0,1]", "early": "density 2.5 on [0,.25], .5 on (.25,1]; inverse CDF t=u/2.5 if u<=.625, otherwise t=2u-1"},
        matching="Exactly same architecture/parameters, initialization, training examples, base Gaussian and uniform draws, optimizer updates, LR and flow call counts. Time transformation and resulting interpolated fine inputs differ by design. Uniform is rerun and compared with the frozen prior result; no geometric loss or extra conditioning module.",
        primary="Fixed two-state TRAIN learnability only: both proteins and seeds, initial/final state assignment, CA/C-N, conditional fine movement, native reconstruction floor and Euler64 diagnostic. Not full-trajectory or AA-only generation quality.",
        scope="Two previously selected native training pairs only. Positive time reweighting tests finite-budget optimization; success does not establish ensemble fidelity or unseen-sequence improvement.",
        stop_rule="Exactly 2048 updates, final checkpoint only, no extension or time-density sweep, no filtering, independent test/R3 coordinates or default-model promotion.")
    shutil.copyfile(PARENT / "pairs.pt", ROOT / "pairs.pt")
    for arm in ARMS:
        (ROOT / arm).mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "pairs.pt", ROOT / arm / "pairs.pt")
    pairs = torch.load(ROOT / "pairs.pt", map_location="cpu", weights_only=True)
    checks = []
    for name, pair in pairs.items():
        model = make_model(p, pair["a"], 2026)
        p["trainable_parameters"] = sum(v.numel() for v in model.parameters() if v.requires_grad)
        adapter = type("FineAdapter", (), {"fine_model": model})()
        for case in [pair["a"], pair["b"]]:
            aa = encode_aa(case["sequence"])
            model.zero_grad(set_to_none=True)
            old, terms = fine_flow_loss(adapter, aa, case["coarse"], case["fine_continuous"],
                                        torch.Generator().manual_seed(714))
            old.backward()
            gradients = {k: v.grad.clone() for k, v in model.named_parameters() if v.grad is not None}
            model.zero_grad(set_to_none=True)
            new, replay = fine_time_loss(adapter, aa, case["coarse"], case["fine_continuous"],
                                        torch.Generator().manual_seed(714), "uniform")
            torch.testing.assert_close(old, new, atol=0, rtol=0)
            if terms["draw_sha256"] != replay["draw_sha256"]:
                raise RuntimeError("Uniform draw replay failed")
            new.backward()
            for key, value in model.named_parameters():
                if key in gradients:
                    torch.testing.assert_close(value.grad, gradients[key], atol=0, rtol=0)
            model.zero_grad(set_to_none=True)
            early, early_terms = fine_time_loss(adapter, aa, case["coarse"], case["fine_continuous"],
                                               torch.Generator().manual_seed(714), "early")
            early.backward()
            grad = model.fine_head[-1].weight.grad
            if not torch.isfinite(early) or not torch.isfinite(grad).all() or grad.norm() <= 0:
                raise RuntimeError("Invalid real-state early-time gradient")
            if replay["draw_sha256"] != early_terms["draw_sha256"]:
                raise RuntimeError("Base random inputs differ")
            checks.append(dict(protein=name, case_id=case["id"], exact_uniform_loss_and_all_gradients=True,
                               uniform_time=replay["time"], early_time=early_terms["time"]))
    atomic_json(ROOT / "integration_check.json", dict(status="passed", real_states_checked=4,
        checks=checks, optimizer_updates=0, unit_tests_passed=3))
    sources = [Path(__file__), Path("upstream/experiments/early_time_fine_flow.py"),
        Path("upstream/tests/test_early_time_fine_flow.py"), Path("upstream/scripts/early_time_fine_flow.sbatch"),
        Path("upstream/experiments/run_explicit_coarse_fine.py"),
        PARENT / "protocol.json", PARENT / "report.json", audit,
        ROOT / "pairs.pt", ROOT / "integration_check.json"]
    sources += [ROOT / a / "pairs.pt" for a in ARMS]
    sources += [PARENT / n / f"seed{s}" / "last.pt" for n in p["proteins"] for s in p["seeds"]]
    p["source_sha256"].update({str(x): sha(x) for x in sources})
    atomic_json(ROOT / "protocol.json", p)
    for arm in ARMS:
        shutil.copyfile(ROOT / "protocol.json", ROOT / arm / "protocol.json")
    print("EARLY_TIME_FINE_READY", flush=True)


def train(p, arm, protein, seed):
    if sha(ROOT / arm / "protocol.json") != sha(ROOT / "protocol.json"):
        raise RuntimeError("Arm protocol differs")
    digest = hashlib.sha256(); times = []
    def loss_fn(model, aa, c, f, generator):
        loss, terms = fine_time_loss(model, aa, c, f, generator, arm)
        digest.update(terms["draw_sha256"].encode()); times.append(terms["time"])
        return loss, terms
    old_root, old_loss = engine.RUN, engine.fine_flow_loss
    try:
        engine.RUN = ROOT / arm
        engine.fine_flow_loss = loss_fn
        engine.train(p, protein, seed)
    finally:
        engine.RUN, engine.fine_flow_loss = old_root, old_loss
    if len(times) != p["updates"] * p["batch_size"]:
        raise RuntimeError("Unexpected training call count")
    out = ROOT / arm / protein / f"seed{seed}"
    row = json.loads((out / "complete.json").read_text())
    row.update(arm=arm, realized_base_draw_sha256=digest.hexdigest(),
        mean_training_time=float(np.mean(times)), fraction_time_le_quarter=float(np.mean(np.array(times) <= .25)),
        trainable_parameters=p["trainable_parameters"])
    atomic_json(out / "complete.json", row)


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
                    raise RuntimeError("Incomplete time-density training")
                rows.append(row)
    replay = []
    old_rows = json.loads((PARENT / "report.json").read_text())["rows"]
    for name in p["proteins"]:
        for seed in p["seeds"]:
            pair = [r for r in rows if r["protein"] == name and r["seed"] == seed]
            if len({r["realized_base_draw_sha256"] for r in pair}) != 1:
                raise RuntimeError("Gaussian/uniform draws did not match")
            old = next(r for r in old_rows if r["protein"] == name and r["seed"] == seed)
            uniform = next(r for r in pair if r["arm"] == "uniform")
            delta = {k: concise(uniform, "final")[k] - v for k, v in concise(old, "final").items()}
            replay.append(dict(protein=name, seed=seed, final_metric_differences=delta))
    verify(p)
    summary = []
    for name in p["proteins"]:
        for arm in ARMS:
            group = [concise(r, "final") for r in rows if r["protein"] == name and r["arm"] == arm]
            summary.append(dict(protein=name, arm=arm, **{k: float(np.mean([r[k] for r in group])) for k in group[0]}))
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, summary=summary,
        uniform_replay_differences=replay, scope=p["scope"], matching=p["matching"],
        realized_base_draws_matched=True, model_default_replaced=False,
        training_only=True, no_ensemble_claim=True, test_coordinates_opened=False, R3_opened=False))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for i, name in enumerate(p["proteins"]):
        group = [r for r in summary if r["protein"] == name]
        for j, (key, scale, label) in enumerate([
            ("fine_state_accuracy", 100., "Correct fine state (%)"),
            ("ca_rmsd_A", 1., "Matched-state CA RMSD (Å)"),
            ("cn_rate", 100., "C–N abnormalities (%)")]):
            ax = axes[i, j]
            ax.bar([0, 1], [r[key] * scale for r in group], color=["#777777", "#24679e"])
            for x, arm in enumerate(ARMS):
                values = [concise(r, "final")[key] * scale for r in rows if r["protein"] == name and r["arm"] == arm]
                ax.scatter([x - .07, x + .07], values, s=16, color="black", zorder=3)
            ax.set_xticks([0, 1], ["Uniform time", "Early weighted"], fontsize=8)
            ax.set_title(name); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
            if j == 0:
                ax.axhline(50, color=".4", ls="--", lw=1); ax.set_ylim(0, 110)
    fig.suptitle("Same fine flow: uniform versus early-time weighting\nTwo selected training pairs; native coarse supplied; dots are two seeds", fontsize=11)
    folder = Path("paper/figures/early_time_fine_flow_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Early_Time_Fine_Flow." + ext), dpi=180)
    plt.close(fig)
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n\n" + "\n".join(json.dumps(r) for r in summary) + "\n")
    print("EARLY_TIME_FINE_COMPLETE", json.dumps(summary), flush=True)


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
                raise RuntimeError("Undeclared time-density trial")
            train(p, args.arm, args.protein, args.seed)
