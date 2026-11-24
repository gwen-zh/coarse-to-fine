"""Frozen coarse sampling, local support and training-data coverage audit."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from coarse_generation_diagnostic import (block_counts, coarse_metrics,
    pool_backbones, scalar_features, weighted_rmsd_matrix)
from diagnostic_io import atomic_json
from hierarchical_ensemble_flow import component_seed
from hierarchical_pipeline_io import sha
from run_fine_geometry_flow import make_model, state_digest
from run_multiframe_early_fine import BASE, encode_aa
from run_multisequence_ensemble_flow import sample_seed
from structural_metrics import aligned_rmsd


ROOT = Path("artifacts/coarse_generation_diagnostic_v1")
PARENT = Path("artifacts/multiframe_coarse_context_v1")
LABELS = Path("artifacts/multiframe_early_fine_v1")


def verify(p):
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest:
            raise RuntimeError("Changed coarse diagnostic input: " + name)


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in [0., .05, .5, .95, 1.]}


def native_descriptor(cases, cap):
    length = len(cases[0]["sequence"])
    centers = np.stack([c["coarse"].numpy() for c in cases])
    f = scalar_features(centers, length, cap)
    ca_steps = np.concatenate([np.linalg.norm(np.diff(c["target"][:, 1].numpy(), axis=0), axis=-1) for c in cases])
    return dict(protein=cases[0]["protein"], length=length, frames=len(cases),
        center_rg_quantiles_A=quantiles(f["center_rg_A"]),
        adjacent_center_distance_quantiles_A=quantiles(f["adjacent_distance_A"]),
        native_CA_step_above_declared_cap_fraction=float((ca_steps > cap).mean()),
        native_maximum_CA_step_A=float(ca_steps.max()))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        raise RuntimeError("Use a fresh diagnostic namespace")
    p = copy.deepcopy(json.loads((PARENT / "protocol.json").read_text())); verify(p)
    previous = json.loads((PARENT / "assessment.json").read_text())
    if previous["status"] != "complete_native_coarse_fine_gain_full_geometry_gain_no_material_accuracy_gain":
        raise RuntimeError("Complete and assess the conditioning experiment first")
    train = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    dev = torch.load(LABELS / "development_labels.pt", map_location="cpu", weights_only=True)
    cap = 4.1; selected = []
    for domain, groups in [("MD_train", train["train"]), ("development", dev["validation"])]:
        for name, cases in sorted(groups.items()):
            indices = np.linspace(0, len(cases) - 1, min(128, len(cases)), dtype=int)
            refs = [cases[int(i)] for i in indices]
            selected.append(dict(protein=name, domain=domain, length=len(refs[0]["sequence"]),
                sequence=refs[0]["sequence"], native_case_ids=[c["id"] for c in refs],
                centers=torch.stack([c["coarse"] for c in refs]),
                backbones=torch.stack([c["target"] for c in refs]), samples=64))
    by_id = {c["id"]: c for c in train["PDB_train"]}
    for name in p["PDB_probe_ids"]:
        case = by_id[name]
        selected.append(dict(protein=case["protein"], domain="PDB_train_single_structure", length=len(case["sequence"]),
            sequence=case["sequence"], native_case_ids=[case["id"]],
            centers=case["coarse"][None], backbones=case["target"][None], samples=16))
    if len(selected) != 15 or any(len(block_counts(c["length"])) < 5 for c in selected):
        raise RuntimeError("Unexpected fixed diagnostic cohort")
    torch.save(selected, ROOT / "references.pt")
    md_desc = [native_descriptor(cases, cap) for _, cases in sorted(train["train"].items())]
    pdb_desc = [native_descriptor([c], cap) for c in train["PDB_train"]]
    dev_desc = [native_descriptor(cases, cap) for _, cases in sorted(dev["validation"].items())]
    coverage = dict(status="complete", MD_training=md_desc, PDB_training=pdb_desc,
        development=dev_desc, MD_training_proteins=5, MD_training_frames=sum(len(x) for x in train["train"].values()),
        PDB_training_structures=len(train["PDB_train"]),
        note="Descriptive ranges and cached frame counts, not a proof of sampling sufficiency or a causal attribution. Rg here is the residue-count-weighted spread of BLOCK CENTERS, not atomistic Rg. Static PDB targets are not MD ensembles.")
    atomic_json(ROOT / "native_training_coverage.json", coverage)
    p.update(training_performed=False, optimizer_updates=0, ca_step_cap_A=cap,
        diagnostic_proteins=[dict(protein=c["protein"], domain=c["domain"], length=c["length"],
            reference_frames=len(c["centers"]), generated_samples=c["samples"]) for c in selected],
        architecture="Original parent coarse sampler held fixed; the completed geometry-aware fine model and static control supply already saved full AA-only outputs. No model, structure encoder, decoder or codebook is trained or changed.",
        training="No optimizer, adaptation or model selection. Only cached train/development coordinates and frozen AA-only coarse inference are used.",
        matching="Same two original coarse checkpoints and component seeds as existing generation. Development coarse centers are pooled from saved local-geometry outputs, with first-sample AA-only replay checked. Newly sampled coarse-only draws for train proteins use the same seed namespace and Euler16 solver. Reference counts are capped at 128 by fixed equally spaced indices; no reference-nearest candidate acceptance.",
        primary="Report every protein/seed: residue-count-weighted center RMSD and reverse coverage, center spread/Rg, long-range pair-distance JS, adjacency distances and conditional span flags. For development, compare the exact fixed-center CA-RMSD lower bound with both saved fine conditions' actual CA error against the SAME reference ensemble.",
        lower_bound="For any proper rigid alignment Jensen gives CA_RMSD >= sqrt(sum_b n_b/N * center_error_b^2). Minimizing this center objective over alignment and then over references preserves a lower bound. It is not an achievable decoder floor, percentage of error caused by coarse, or a bound after changing block centers.",
        geometry_flag="If EVERY consecutive CA step is at most 4.1 A, adjacent mean centers of contiguous blocks must be separated by at most 4.1*(n_b+n_{b+1})/2 A. Count upper-bound violations as conditional flags, not a universal physical validity verdict. Include partial final blocks and report the native CA steps exceeding 4.1 A. There is no lower center-spacing cutoff.",
        histograms="Existing fixed physical center-distance bins 0..66 by 2 A (long-range block index gap >=4), and weighted center-Rg bins 0..33 by 1 A, with under/overflow. Plot-only bins may span all observed values and do not select models.",
        scope="Read-only exploratory diagnostic, five seen MD training sequences, eight preselected static PDB training targets and two repeatedly used development sequences. No reserved test or R3 coordinates. Static-PDB deviation from one available target is not ensemble-fidelity evidence. No causal inference that data or architecture alone explains a gap.",
        stop_rule="One fixed pass per seed and one aggregate report; no optimization, candidate filtering, checkpoint selection, automatic model promotion or automatic training extension.")
    paths = [Path(__file__), Path("upstream/experiments/coarse_generation_diagnostic.py"),
        Path("upstream/tests/test_coarse_generation_diagnostic.py"),
        Path("upstream/scripts/coarse_generation_diagnostic.sbatch"),
        PARENT / "protocol.json", PARENT / "report.json", PARENT / "assessment.json",
        ROOT / "references.pt", ROOT / "native_training_coverage.json"]
    for seed in p["seeds"]:
        for arm in ["raw_context", "local_geometry"]:
            for name in dev["validation"]:
                paths.append(PARENT / arm / f"seed{seed}" / name / "full.npz")
    p["source_sha256"].update({str(x): sha(x) for x in paths})
    atomic_json(ROOT / "protocol.json", p)
    print("COARSE_DIAGNOSTIC_READY", len(selected), "proteins; no optimization", flush=True)


def project_centers(centers, length):
    counts = centers.new_tensor(block_counts(length))
    return centers - (centers * counts[:, None]).sum(0) / length


@torch.no_grad()
def diagnose(p, seed):
    references = torch.load(ROOT / "references.pt", map_location="cpu", weights_only=True)
    wrapper = make_model(p, seed); model = wrapper.coarse_model.eval().requires_grad_(False)
    original = state_digest(model); out = ROOT / f"seed{seed}"; out.mkdir(exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Diagnostic already complete")
    rows = []; started = time.perf_counter()
    # Development feature tensors are attached only when its inference begins.
    bank = dict(model.feature_bank)
    index = json.loads(Path(p["all_feature_index"]).read_text())
    order = sorted(references, key=lambda c: (c["domain"] == "development", c["domain"], c["protein"]))
    for position, case in enumerate(order):
        name = case["protein"]; domain = case["domain"]; length = case["length"]; aa = encode_aa(case["sequence"])
        if case["sequence"] not in bank:
            if domain != "development":
                raise RuntimeError("Training AA feature bank is incomplete")
            entry = index[case["sequence"]]
            if sha(entry["path"]) != entry["sha256"]:
                raise RuntimeError("Changed development AA-only features")
            with np.load(entry["path"], allow_pickle=False) as z:
                bank[case["sequence"]] = torch.from_numpy(z["h"].copy()).float()
            model.feature_bank = bank
        native = case["centers"].numpy()
        if domain == "development":
            with np.load(PARENT / "local_geometry" / f"seed{seed}" / name / "full.npz") as z:
                saved = z["backbone"].copy()
            generated = pool_backbones(saved)
            key = component_seed(sample_seed(seed, name, 0), "coarse")
            replay = project_centers(model.sample_coarse(aa, key, p["coarse_nfe"]), length)
            torch.testing.assert_close(torch.from_numpy(generated[0]).float(), replay, atol=2e-5, rtol=2e-5)
        else:
            generated = np.stack([project_centers(model.sample_coarse(aa,
                component_seed(sample_seed(seed, name, i), "coarse"), p["coarse_nfe"]), length).numpy()
                for i in range(case["samples"])])
        if len(generated) != case["samples"]:
            raise RuntimeError("Wrong sample count")
        stats = coarse_metrics(generated, native, length, p["ca_step_cap_A"])
        gf = scalar_features(generated, length, p["ca_step_cap_A"])
        rf = scalar_features(native, length, p["ca_step_cap_A"])
        ca = case["backbones"][:, :, 1].numpy()
        native_step = np.linalg.norm(np.diff(ca, axis=1), axis=-1)
        stats.update(native_maximum_CA_step_A=float(native_step.max()),
            native_CA_step_above_cap_fraction=float((native_step > p["ca_step_cap_A"]).mean()))
        bound_comparison = []
        if domain == "development":
            for arm in ["raw_context", "local_geometry"]:
                with np.load(PARENT / arm / f"seed{seed}" / name / "full.npz") as z:
                    bb = z["backbone"].copy()
                c = pool_backbones(bb)
                # Use this arm's ACTUAL decoded means, including roundoff.
                lower = weighted_rmsd_matrix(c, native, length).min(1)
                actual = np.array([[aligned_rmsd(sample[:, 1], target) for target in ca] for sample in bb]).min(1)
                if (lower > actual + 1e-4).any():
                    raise RuntimeError("CA lower-bound check failed")
                bound_comparison.append(dict(arm=arm, mean_CA_lower_bound_A=float(lower.mean()),
                    mean_actual_CA_error_A=float(actual.mean()),
                    maximum_possible_mean_CA_improvement_at_fixed_centers_A=float((actual - lower).mean()),
                    lower_bound_A_by_sample=lower.tolist(), actual_CA_error_A_by_sample=actual.tolist()))
        rows.append(dict(seed=seed, protein=name, domain=domain, length=length,
            metrics=stats, fixed_center_CA_lower_bound=bound_comparison))
        np.savez_compressed(out / (name + ".npz"), generated_centers=generated, reference_centers=native,
            generated_center_rg_A=gf["center_rg_A"], reference_center_rg_A=rf["center_rg_A"],
            generated_adjacent_distance_A=gf["adjacent_distance_A"], reference_adjacent_distance_A=rf["adjacent_distance_A"],
            adjacent_span_bound_A=gf["adjacent_span_bound_A"])
        atomic_json(out / "progress.json", dict(completed=position + 1, total=len(order), protein=name,
            optimizer_updates=0, seconds=time.perf_counter() - started))
        print("COARSE_DIAGNOSTIC", seed, position + 1, len(order), name, flush=True)
    if state_digest(model) != original:
        raise RuntimeError("Read-only diagnostic changed weights")
    verify(p)
    atomic_json(out / "complete.json", dict(status="complete", seed=seed, rows=rows,
        seconds=time.perf_counter() - started, model_weights_unchanged=True, optimizer_updates=0,
        existing_AA_only_sampling_replayed=True, test_coordinates_opened=False, R3_opened=False))


def report(p):
    rows = []; complete = []
    for seed in p["seeds"]:
        result = json.loads((ROOT / f"seed{seed}" / "complete.json").read_text())
        if len(result["rows"]) != 15 or not result["model_weights_unchanged"] or result["optimizer_updates"]:
            raise RuntimeError("Incomplete or non-read-only diagnostic")
        complete.append({k: v for k, v in result.items() if k != "rows"}); rows += result["rows"]
    macro = []
    for domain in sorted({r["domain"] for r in rows}):
        group = [r["metrics"] for r in rows if r["domain"] == domain]
        keys = [k for k in group[0] if all(v[k] is not None for v in group)]
        macro.append(dict(domain=domain, metrics={key: float(np.mean([v[key] for v in group])) for key in keys}))
    verify(p)
    atomic_json(ROOT / "report.json", dict(status="complete", rows=rows, domain_means=macro,
        completed_passes=complete, lower_bound=p["lower_bound"], geometry_flag=p["geometry_flag"],
        scope=p["scope"], training_performed=False, model_default_replaced=False,
        test_coordinates_opened=False, R3_opened=False))
    atomic_json(ROOT / "assessment.json", dict(status="complete_awaiting_result_review", scope=p["scope"],
        next="Review coarse bounds, local support and train/development differences before designing a bounded training intervention. No automatic optimization or data download.",
        training_performed=False, model_default_replaced=False, test_coordinates_opened=False, R3_opened=False))
    (ROOT / "result_summary.txt").write_text(p["scope"] + "\n" + p["lower_bound"] + "\n\n" +
        "\n".join(json.dumps(r) for r in macro) + "\n")
    plot(rows)
    print("COARSE_DIAGNOSTIC_COMPLETE", json.dumps(macro), flush=True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    domains = ["MD_train", "PDB_train_single_structure", "development"]
    for row_index, key, scale in [(0, "mean_sample_nearest_reference_center_rmsd_A", 1.),
                                  (1, "generated_conditional_span_flag_rate", 100.)]:
        ax = axes[row_index, 0]
        for i, domain in enumerate(domains):
            group = [r for r in rows if r["domain"] == domain]
            ax.bar(i, scale * np.mean([r["metrics"][key] for r in group]), color=["#777777", "#b9b9b9", "#24679e"][i], alpha=.6)
            for seed, offset, marker in [(2026, -.1, "o"), (2027, .1, "^")]:
                values = [scale * r["metrics"][key] for r in group if r["seed"] == seed]
                ax.scatter(np.full(len(values), i + offset), values, s=18, marker=marker, color="black", alpha=.7)
        ax.set_xticks(range(3), ["5 MD train", "8 static train", "2 dev"], fontsize=8)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax.set_ylabel("Weighted center RMSD to nearest reference (Å)" if row_index == 0 else "Conditional span flags (%)", fontsize=9)
    for column, name in enumerate(["5w82_E", "5ydn_A"], start=1):
        group = [r for r in rows if r["protein"] == name]
        ax = axes[0, column]
        for i, (label, key, color) in enumerate([
                ("Center bound", "mean_CA_lower_bound_A", "#777777"),
                ("Raw fine", "mean_actual_CA_error_A", "#c27b24"),
                ("Geom. fine", "mean_actual_CA_error_A", "#24679e")]):
            values = [next(v for v in r["fixed_center_CA_lower_bound"] if v["arm"] == ("raw_context" if i == 1 else "local_geometry"))[key] for r in group]
            ax.bar(i, np.mean(values), color=color, alpha=.75)
            for value, offset, marker in zip(values, [-.08, .08], ["o", "^"]): ax.plot(i + offset, value, marker, color="black", ms=4)
        ax.set_xticks(range(3), ["Center bound", "Raw fine", "Geom. fine"], fontsize=8)
        ax.set_ylabel("Nearest-reference CA RMSD (Å)", fontsize=9); ax.set_title(name)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax = axes[1, column]; generated = []; reference = None
        for r in group:
            with np.load(ROOT / f"seed{r['seed']}" / (name + ".npz")) as z:
                generated.append(z["generated_center_rg_A"].copy()); reference = z["reference_center_rg_A"].copy()
        generated = np.concatenate(generated)
        bins = np.linspace(0, 1.05 * max(generated.max(), reference.max()), 25)
        ax.hist(reference, bins=bins, density=True, histtype="step", lw=2, color="black", label="Native MD")
        ax.hist(generated, bins=bins, density=True, alpha=.5, color="#24679e", label="Generated coarse")
        ax.set_xlabel("Residue-weighted block-center Rg (Å)"); ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen coarse generator: geometric constraints and reference coverage\nNo optimization · two seeds · train / reused development only · span flags assume CA steps ≤ 4.1 Å", fontsize=12)
    folder = Path("paper/figures/coarse_generation_diagnostic_v1"); folder.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(folder / ("Coarse_Generation_Diagnostic." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=["prepare", "diagnose", "report"], required=True)
    parser.add_argument("--seed", type=int, default=2026); args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare()
    else:
        p = json.loads((ROOT / "protocol.json").read_text()); verify(p)
        if args.stage == "report": report(p)
        else:
            if args.seed not in p["seeds"]: raise RuntimeError("Undeclared seed")
            diagnose(p, args.seed)
