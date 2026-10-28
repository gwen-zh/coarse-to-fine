"""Cached descriptive search diagnostics; no fitting or test-set access."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from structural_metrics import aligned_rmsd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "artifacts/development/objective_geometry_diagnosis"
PROBE = ROOT / "artifacts/development/geometry_gradient_probe"
OUT = ROOT / "artifacts/development/fine_search_headroom"


def dump_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = json.loads((A / "protocol.json").read_text())["geometry_targets"]
    inputs = [A / "protocol.json"]
    bins, candidates, probes = [], [], []
    for target in ids:
        jp, npz = A / "A" / (target + ".json"), A / "A" / (target + ".npz")
        inputs.extend([jp, npz])
        data = json.loads(jp.read_text())
        records = data["records"]
        tokens = np.load(npz)
        fine = tokens["fine"]
        assert len(records) == len(fine) == 50
        baseline_index = next(i for i, r in enumerate(records) if r["source"] == "argmax")
        baseline = records[baseline_index]
        for temperature in [0.5, 1.0, 2.0]:
            indices = [i for i, r in enumerate(records) if r["source"] == "sampled" and r["temperature"] == temperature]
            assert len(indices) == 16
            order = sorted(indices, key=lambda i: records[i]["log_score"])
            for count in [2, 4]:
                for b, group in enumerate(np.array_split(order, count)):
                    scores = [records[i]["log_score"] for i in group]
                    rates = [records[i]["cn_rate"] for i in group]
                    q10, median = np.quantile(rates, [0.1, 0.5], method="linear")
                    bins.append(dict(target=target, temperature=temperature, bin_count=count, bin=b,
                                     n=len(group), distinct_scores=len(set(scores)), score_width=max(scores)-min(scores),
                                     cn_q10=float(q10), cn_median=float(median),
                                     median_minus_q10_pp=100*float(median-q10)))
            for i in indices:
                r = records[i]
                drop = baseline["log_score"] - r["log_score"]
                assert drop >= -1e-6, "Argmax must maximize this factorized score"
                candidates.append(dict(target=target, temperature=temperature, sample=r["sample"],
                                       score_drop_nats_per_residue=drop,
                                       changed_fraction=float(np.mean(fine[i] != fine[baseline_index])),
                                       cn_delta_pp=100*(r["cn_rate"]-baseline["cn_rate"]),
                                       ca_clash_delta=r["metrics"]["upstream_ca_clashes"]-baseline["metrics"]["upstream_ca_clashes"],
                                       geometry_pareto=bool(r["cn_rate"] <= baseline["cn_rate"] and
                                          r["ca_clashes_per_residue"] <= baseline["ca_clashes_per_residue"] and
                                          (r["cn_rate"] < baseline["cn_rate"] or r["ca_clashes_per_residue"] < baseline["ca_clashes_per_residue"]))))
    manifest = PROBE / "manifest.json"
    inputs.append(manifest)
    for target in json.loads(manifest.read_text())["targets"]:
        name = target["target"]
        bp = PROBE / name / "kl_only.npz"
        base = np.load(bp)
        inputs.append(bp)
        for arm in ["geometry_gradient", "shuffled_geometry_gradient"]:
            cp, jp = PROBE / name / (arm+".npz"), PROBE / name / (arm+".json")
            inputs.extend([cp, jp])
            candidate, data = np.load(cp), json.loads(jp.read_text())
            best, original = data["best"], data["baseline"]
            changed = int(np.sum(candidate["fine"] != base["fine"]))
            assert changed == best["changed_tokens"]
            probes.append(dict(target=name, arm=arm, selected_step=best["step"], length=len(base["fine"]),
                               changed_tokens=changed, changed_fraction=changed/len(base["fine"]),
                               aligned_ca_displacement=aligned_rmsd(candidate["backbone"][:,1], base["backbone"][:,1]),
                               cn_before=original["backbone_cn_outliers"], cn_after=best["metrics"]["backbone_cn_outliers"],
                               ca_clashes_before=original["upstream_ca_clashes"], ca_clashes_after=best["metrics"]["upstream_ca_clashes"],
                               selected_kl=best["kl"]))
    strata = []
    for t in [0.5, 1.0, 2.0]:
        for count in [2, 4]:
            rows = [r for r in bins if r["temperature"] == t and r["bin_count"] == count]
            strata.append(dict(temperature=t, bin_count=count, samples_per_bin=16//count,
                               protein_count=len(ids), mean_within_bin_spread_pp=float(np.mean([r["median_minus_q10_pp"] for r in rows])),
                               median_score_width=float(np.median([r["score_width"] for r in rows]))))
    # Descriptive grid only: not a selection of deployment settings or a coordinate-feasible bound.
    grid=[]
    for drop in [0.05, 0.1, 0.25, 0.5, 1.0]:
        for edits in [0.05, 0.1, 0.2, 1.0]:
            feasible=[r for r in candidates if r["score_drop_nats_per_residue"] <= drop and r["changed_fraction"] <= edits]
            better=[r for r in feasible if r["geometry_pareto"]]
            grid.append(dict(score_drop_cap=drop, edit_fraction_cap=edits,
                             proteins_with_candidate=len(set(r["target"] for r in feasible)),
                             proteins_with_pareto_candidate=len(set(r["target"] for r in better)),
                             candidate_count=len(feasible), pareto_count=len(better)))
    pareto=[r for r in candidates if r["geometry_pareto"]]
    summary=dict(scope="24 reused validation proteins and four selected TRAIN probes; descriptive only; no holdout access",
                 definitions=dict(score="Mean original T=1 fine-only factorized log probability, conditional on one fixed coarse realization",
                                  spread="Within equal-count likelihood bins, median CN rate minus interpolated tenth percentile; NOT an upper bound",
                                  probe="Saved geometry-selected best-step candidates, NOT final-step or minimum-required displacement"),
                 protein_count=len(ids), sampled_candidates=len(candidates), strata=strata,
                 sampled_pareto_proteins=len(set(r["target"] for r in pareto)), sampled_pareto_candidates=len(pareto),
                 pareto_candidate_median_score_drop=float(np.median([r["score_drop_nats_per_residue"] for r in pareto])) if pareto else None,
                 pareto_candidate_median_changed_fraction=float(np.median([r["changed_fraction"] for r in pareto])) if pareto else None,
                 descriptive_constraint_grid=grid, selected_train_probe_displacements=probes,
                 limitations=["No A-sample coordinates cached: no coordinate feasibility or ensemble spread claim",
                              "Sixteen samples per protein-temperature provide sparse conditional quantiles",
                              "Same coarse realization is not a full-generation ensemble",
                              "Near-zero rank correlation does not establish independence or equivalence",
                              "Argmax maximizes this original factorized score; distinct tokens generally reduce it",
                              "Gated metrics describe acceptance behavior, not independent evidence of effectiveness"],
                 input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    dump_csv(OUT/"within_bin_spread.csv",bins)
    dump_csv(OUT/"sample_candidates.csv",candidates)
    dump_csv(OUT/"selected_train_probe_displacements.csv",probes)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    # Standalone scientific figure; each point is one protein-bin, not an independent protein.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for t, color in [(0.5, "tab:blue"), (1.0, "tab:orange"), (2.0, "tab:green")]:
        rows = [r for r in candidates if r["temperature"] == t]
        axes[0].scatter([r["score_drop_nats_per_residue"] for r in rows],
                        [r["cn_delta_pp"] for r in rows], s=8, alpha=0.3, color=color, label=f"T={t}")
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(xlabel="Original factorized score loss vs argmax (nats/residue)", ylabel="CN outlier change (percentage points)")
    axes[0].legend()
    normal = [r for r in probes if r["arm"] == "geometry_gradient"]
    axes[1].bar([r["target"] for r in normal], [r["aligned_ca_displacement"] for r in normal])
    axes[1].set(ylabel="Aligned CA displacement from initial output (A)", title="Four selected TRAIN probes: saved best step")
    axes[1].tick_params(axis="x", rotation=30)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / ("headroom_diagnostics." + ext), dpi=180)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["input_sha256","descriptive_constraint_grid","limitations","definitions"]},indent=2))


if __name__ == "__main__":
    analyze()
