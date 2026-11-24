"""CPU resource backstop for the fixed PAR adapter pilot, primary budget only.

All four arms/seeds are rerun in one CPU environment. GPU controls are never
mixed into these comparisons. Numerical model/update functions are unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from diagnostic_io import atomic_json
from fine_geometry_flow import sample_training_batch
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha, save_checkpoint, restore_checkpoint
from par_coarse_backbone import sample_coarse_batch
from par_coarse_transfer import ARMS, conditioning_features, update, identity, frozen_versions, assert_frozen
from report_par_coarse_transfer import summary
import run_par_coarse_transfer as original
from run_multisequence_ensemble_flow import sample_seed


ROOT = Path("artifacts/par_coarse_cpu_transfer_v1")
PARENT = original.ROOT


def prepare():
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("A fresh CPU resource-backstop namespace is required")
    p = original.load(PARENT / "protocol.json")
    original.verify(p)
    p.update(
        sampling={"16": 64},
        execution_backend="CPU float32, two Torch threads, all four arms/seeds on the same CPU node. No GPU control reused.",
        resource_backstop="Original GPU arms remain scheduled. This separate CPU-only study preserves the 512 native updates and primary16-step/64-sample comparison; it does not run the400-step diagnostic. Backend choice was based only on queue availability and a discarded training-only speed benchmark, before sequence-arm scores existed.",
        scope="CPU resource backstop, two repeatedly reused development proteins and two training-draw seeds. No independent test, calibrated thermodynamics or published-PAR reproduction claim.",
        evaluation="Save AA-only coarse outputs at16steps x64samples before/after512; same noise keys for both arms/endpoints. All full structures are saved with original frozen fine16+codec before loading native development coordinates. No400-step diagnostic in this CPU backstop.",
        stop_rule="Exactly512 native updates per arm/seed; final endpoint only. Report all four CPU runs, regardless of their quality or the GPU queue. No default promotion or adaptive extension.",
    )
    for path in [Path(__file__), PARENT / "protocol.json", PARENT / "cpu_resource_benchmark.json"]:
        p["source_sha256"][str(path)] = sha(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "protocol.json", p)
    print("CPU_TRANSFER_READY", flush=True)


@torch.no_grad()
def sample_endpoint(p, model, bank, arm, seed, out, endpoint):
    for item in p["targets"]:
        features = conditioning_features(bank(item["sequence"]), arm)
        keys = [component_seed(sample_seed(seed, item["protein"], i), "coarse") for i in range(64)]
        start = time.perf_counter()
        values = sample_coarse_batch(model, features, item["length"], keys, 16).numpy()
        folder = out / f"step{endpoint}" / item["protein"]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "coarse_nfe16.npz", coarse=values,
                            valid=np.isfinite(values).all(axis=(1, 2)), seeds=np.asarray(keys, dtype=np.uint32))
        atomic_json(folder / "coarse_nfe16.json", dict(samples=64, steps=16,
                    batch_seconds=time.perf_counter()-start, backend="CPU", native_coordinates_loaded=False))
        atomic_json(out / "progress.json", dict(phase="sampling", endpoint=endpoint, protein=item["protein"]))
    model.nn.init_repr_factory.features = None


def train(p, arm, seed, resume):
    model, factory, view = original.model_for(seed, "cpu")
    guard = frozen_versions(model)
    optimizer = torch.optim.AdamW(view.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = original.batch_rng(seed)
    bank = original.FeatureBank(p["training_feature_index"])
    evaluation_bank = original.FeatureBank(p["feature_index"])
    groups = torch.load(original.LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    out = ROOT / arm / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("CPU arm already complete")
    start = 0
    trace = []
    elapsed = 0.
    digest = hashlib.sha256()
    if resume:
        saved = restore_checkpoint(out / "last.pt", view, optimizer, rng, "cpu")
        if saved["protocol_sha256"] != sha(ROOT / "protocol.json") or saved["arm"] != arm or saved["seed"] != seed:
            raise RuntimeError("Wrong CPU resume checkpoint")
        start = saved["updates"]
        elapsed = saved["training_seconds"]
        restore_trace_prefix(out, start)
        trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
        for row in trace:
            digest.update(json.dumps(identity(row), sort_keys=True).encode())
        if digest.hexdigest() != saved["paired_identity_sha256"]:
            raise RuntimeError("CPU checkpoint and trace differ")
    else:
        if (out / "last.pt").exists() or (out / "trace.jsonl").exists():
            raise RuntimeError("Explicit CPU resume required")
        sample_endpoint(p, model, evaluation_bank, arm, seed, out, 0)
    writer = CompactTrainingTrace(512, start)
    clock = time.perf_counter()
    for step in range(start, 512):
        row = update(model, optimizer, sample_training_batch(groups, rng), bank, arm, seed, step, p)
        trace.append(row)
        digest.update(json.dumps(identity(row), sort_keys=True).encode())
        if (step+1) % 32 == 0:
            writer(out / "trace.json", trace)
            atomic_json(out / "progress.json", dict(phase="training", updates=step+1, total=512,
                        training_seconds=elapsed+time.perf_counter()-clock, loss=row["loss"]))
        if (step+1) % 128 == 0:
            assert_frozen(model, guard)
            save_checkpoint(out / "last.pt", view, optimizer, rng, updates=step+1, arm=arm, seed=seed,
                protocol_sha256=sha(ROOT / "protocol.json"), training_seconds=elapsed+time.perf_counter()-clock,
                paired_identity_sha256=digest.hexdigest())
    seconds = elapsed + time.perf_counter() - clock
    sample_endpoint(p, model, evaluation_bank, arm, seed, out, 512)
    assert_frozen(model, guard)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, arm=arm, updates=512,
                paired_identity_sha256=digest.hexdigest(), training_seconds=seconds,
                official_weights_unchanged=True, trainable_parameters=sum(v.numel() for v in view.parameters()),
                host=platform.node(), backend="CPU float32,2threads", native_development_coordinates_loaded=False))
    atomic_json(out / "progress.json", dict(phase="complete", updates=512, total=512))


def report(p):
    rows = []
    audits = []
    for seed in p["seeds"]:
        traces = []
        for arm in ARMS:
            folder = ROOT / arm / f"seed{seed}"
            done = original.load(folder / "complete.json")
            trace = original.load(folder / "trace.json")
            evaluated = original.load(folder / "evaluation.json")
            if done["updates"] != 512 or not done["official_weights_unchanged"] or len(trace) != 512 or len(evaluated["rows"]) != 4:
                raise RuntimeError("Incomplete fixed CPU trial")
            traces.append([identity(row) for row in trace])
            rows.extend(evaluated["rows"])
        if traces[0] != traces[1]:
            raise RuntimeError("CPU control training events differ")
        for target in p["targets"]:
            arrays = []
            for arm in ARMS:
                with np.load(ROOT / arm / f"seed{seed}/step0/{target['protein']}/coarse_nfe16.npz") as z:
                    arrays.append(z["coarse"].copy())
            np.testing.assert_array_equal(*arrays)
        audits.append(dict(seed=seed, all512_training_events_paired=True, initial_coarse_samples_bitwise_equal=True))
    result = {arm: {str(ep): summary([r for r in rows if r["arm"] == arm and r["endpoint"] == ep])
                   for ep in [0, 512]} for arm in ARMS}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    selected = [result["sequence"]["0"], result["position_only"]["512"], result["sequence"]["512"]]
    for ax, key, label, geom, scale in zip(axes,
        ["mean_sample_nearest_reference_ca_rmsd_A", "mean_reference_nearest_sample_ca_rmsd_A", "cn_rate"],
        ["Reference CA RMSD (Å)", "Reverse coverage error (Å)", "C–N outliers (%)"], [False, False, True], [1, 1, 100]):
        values = [(r["geometry"] if geom else r)[key]*scale if r["metrics_available"] else np.nan for r in selected]
        ax.bar(range(3), values, color=["#777777", "#699c74", "#3577a7"])
        for i, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 4), textcoords="offset points", ha="center")
        ax.set_xticks(range(3), ["Zero init", "Position only", "AA condition"], fontsize=9)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen PAR coarse transfer · CPU resource backstop\n512 matched updates · 2 seeds · 2 reused development proteins · coarse16 / fine16", fontsize=12)
    folder = Path("paper/figures/par_coarse_cpu_transfer_v1")
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / f"PAR_Coarse_CPU_Transfer.{ext}", dpi=170)
    plt.close(fig)
    atomic_json(ROOT / "report.json", dict(status="complete_awaiting_review", summary=result, rows=rows,
        paired_audit=audits, protocol_sha256=sha(ROOT / "protocol.json"), scope=p["scope"],
        figure=str(folder / "PAR_Coarse_CPU_Transfer.pdf"), model_default_replaced=False,
        test_coordinates_opened=False, R3_opened=False, independent_test_claim=False))
    print("CPU_TRANSFER_REPORT", json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "evaluate", "report"])
    parser.add_argument("--seed", type=int, choices=[2026, 2027])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
        return
    p = original.load(ROOT / "protocol.json")
    original.verify(p)
    if args.stage == "report":
        report(p)
    elif args.stage == "train":
        train(p, args.arm, args.seed, args.resume)
    else:
        # Output namespace only; reuse the unchanged frozen prediction/evaluator.
        original.ROOT = ROOT
        original.evaluate(p, args.seed, args.arm)


if __name__ == "__main__":
    main()
