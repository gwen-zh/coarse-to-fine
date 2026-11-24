"""Audit the CPU primary comparison without selection, promotion or new sampling."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha
from run_author_codec_interface import METRIC_KEYS, GEOMETRY_KEYS
from run_par_coarse_cpu_transfer import ROOT
from run_par_coarse_transfer import verify


def difference(a, b):
    value = {key: a[key]-b[key] for key in METRIC_KEYS}
    value["geometry"] = {key: a["geometry"][key]-b["geometry"][key] for key in GEOMETRY_KEYS}
    value["coarse"] = {key: a["coarse"][key]-b["coarse"][key] for key in a["coarse"]}
    return value


def main():
    p = json.loads((ROOT / "protocol.json").read_text())
    verify(p)
    report = json.loads((ROOT / "report.json").read_text())
    if report["protocol_sha256"] != sha(ROOT / "protocol.json"):
        raise RuntimeError("Wrong CPU report protocol")
    if len(report["rows"]) != 16 or len(report["paired_audit"]) != 2:
        raise RuntimeError("Missing CPU groups")
    if not all(a["all512_training_events_paired"] and a["initial_coarse_samples_bitwise_equal"] for a in report["paired_audit"]):
        raise RuntimeError("CPU pairing audit failed")
    count = 0
    failed = 0
    zero_full_equal = True
    for seed in p["seeds"]:
        for target in p["targets"]:
            initial_full = []
            for arm in p["arms"]:
                for endpoint in [0, 512]:
                    folder = ROOT / arm / f"seed{seed}/step{endpoint}/{target['protein']}"
                    with np.load(folder / "full_nfe16.npz") as z:
                        full = z["backbone"].copy()
                        coarse = z["coarse"].copy()
                    with np.load(folder / "coarse_nfe16.npz") as z:
                        np.testing.assert_array_equal(coarse, z["coarse"])
                    if full.shape != (64, target["length"], 3, 3):
                        raise RuntimeError("A requested full sample is missing")
                    count += len(full)
                    failed += int((~np.isfinite(full).all(axis=(1, 2, 3))).sum())
                    if endpoint == 0:
                        initial_full.append(full)
            zero_full_equal &= bool(np.array_equal(*initial_full, equal_nan=True))
    if not zero_full_equal:
        raise RuntimeError("CPU zero-initial full predictions differ across arms")
    group_failures = sum(row["metrics"]["failure_count"] for row in report["rows"])
    if group_failures != failed:
        raise RuntimeError("Prediction and metric failure counts differ")
    measured = report["summary"]
    available = all(v[ep]["metrics_available"] for v in measured.values() for ep in ["0", "512"])
    deltas = None
    by_group = []
    if available:
        deltas = dict(
            sequence_minus_position=difference(measured["sequence"]["512"], measured["position_only"]["512"]),
            sequence_minus_zero=difference(measured["sequence"]["512"], measured["sequence"]["0"]),
            position_minus_zero=difference(measured["position_only"]["512"], measured["position_only"]["0"]),
        )
        for seed in p["seeds"]:
            for target in p["targets"]:
                selected = {r["arm"]: r for r in report["rows"] if r["seed"] == seed
                            and r["protein"] == target["protein"] and r["endpoint"] == 512}
                a, b = selected["sequence"], selected["position_only"]
                by_group.append(dict(seed=seed, protein=target["protein"],
                    reference_RMSD_delta_A=a["metrics"][METRIC_KEYS[0]]-b["metrics"][METRIC_KEYS[0]],
                    reverse_coverage_delta_A=a["metrics"][METRIC_KEYS[1]]-b["metrics"][METRIC_KEYS[1]],
                    CN_delta_percentage_points=100*(a["metrics"]["geometry"]["cn_rate"]-b["metrics"]["geometry"]["cn_rate"]),
                    coarse_reference_RMSD_delta_A=a["coarse_metrics"]["mean_sample_nearest_reference_center_rmsd_A"]-b["coarse_metrics"]["mean_sample_nearest_reference_center_rmsd_A"] ))
    reviewed = dict(
        status="complete_audited_primary_comparison_awaiting_interpretation",
        reviewed_at_UTC=datetime.now(timezone.utc).isoformat(),
        summary=measured, paired_deltas=deltas, sequence_minus_position_by_group=by_group,
        audits=dict(all_frozen_source_hashes_passed=True, all512_events_paired=True,
                    initial_coarse_and_full_predictions_bitwise_equal=True,
                    saved_coarse_equals_decoded_coarse=True, requested_predictions_retained=count,
                    failed_samples=failed, metrics_available=available),
        figure=report["figure"], figure_visually_inspected=False,
        model_default_replaced=False, independent_test_claim=False,
        test_coordinates_opened=False, R3_opened=False,
        limits=["Two repeatedly reused development proteins; two seeds are not independent protein replication.",
                "All controls use CPU float32 on the same node; GPU control scores are not merged.",
                "16-step coarse/64-sample primary only; no400-step diagnostic or inter-budget claim.",
                "Nominal60M pretrained PAR contains117107160 AR+flow parameters;464512 adapter parameters are trained.",
                "Only the coarsest branch is transferred; ceil(N/4) block CA means replace PAR floor(N/4) interpolation.",
                "Different pretraining/capacity/compute from the old prototype or Bioinformatics author; not a matched-compute comparison.",
                "Static PDB is not an equilibrium MD ensemble; no calibrated thermodynamic or novelty claim."],
        protocol_sha256=sha(ROOT / "protocol.json"),
        analysis_source_sha256={str(Path(__file__)): sha(__file__)},
    )
    atomic_json(ROOT / "assessment.json", reviewed)
    jobs = ROOT / "jobs.json"
    if jobs.is_file():
        state = json.loads(jobs.read_text())
        state.update(status=reviewed["status"], all_models_running=False,
                     report_status="complete", audit_status="passed",
                     all_requested_sample_failures=failed,
                     assessment=str(ROOT / "assessment.json"), figure=report["figure"])
        atomic_json(jobs, state)
        run_state = Path("artifacts/run_state.json")
        if run_state.is_file():
            overall = json.loads(run_state.read_text())
            overall["par_coarse_cpu_transfer_v1"] = state
            atomic_json(run_state, overall)
    print("CPU_TRANSFER_AUDITED", json.dumps(dict(deltas=deltas, by_group=by_group, audits=reviewed["audits"])), flush=True)


if __name__ == "__main__":
    main()
