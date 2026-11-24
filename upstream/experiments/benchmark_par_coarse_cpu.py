"""Resource-only CPU benchmark; discard its adapter and generated structures."""
import json
from pathlib import Path
import platform
import resource
import time

import torch

from diagnostic_io import atomic_json
from fine_geometry_flow import sample_training_batch
from hierarchical_pipeline_io import sha
from par_coarse_backbone import sample_coarse_batch
from par_coarse_transfer import assert_frozen, frozen_versions, update
from run_par_coarse_transfer import ROOT, LABELS, FeatureBank, batch_rng, model_for, verify


def main():
    torch.set_num_threads(2)
    p = json.loads((ROOT / "protocol.json").read_text())
    verify(p)
    start = time.perf_counter()
    model, factory, view = model_for(2026, "cpu")
    loading_seconds = time.perf_counter() - start
    guard = frozen_versions(model)
    opt = torch.optim.AdamW(view.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    bank = FeatureBank(p["training_feature_index"])
    groups = torch.load(LABELS / "train_labels.pt", map_location="cpu", weights_only=True)
    rng = batch_rng(2026)
    timings = []
    for step in range(4):
        batch = sample_training_batch(groups, rng)
        start = time.perf_counter()
        row = update(model, opt, batch, bank, "sequence", 2026, step, p)
        timings.append(time.perf_counter() - start)
        print("CPU_BENCH_UPDATE", step + 1, timings[-1], row["loss"], flush=True)
    case = groups["PDB_train"][0]
    start = time.perf_counter()
    sample_coarse_batch(model, bank(case["sequence"]), len(case["sequence"]), list(range(8)), 16)
    sampling_seconds = time.perf_counter() - start
    assert_frozen(model, guard)
    result = dict(
        status="complete_resource_benchmark_only", host=platform.node(), threads=2,
        loading_seconds=loading_seconds, four_update_seconds=timings,
        sampling_8_by_16_seconds=sampling_seconds,
        sampling_sequence_length=len(case["sequence"]),
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        official_weights_unchanged=True, adapter_discarded=True,
        development_or_test_coordinates_loaded=False, quality_claim=False,
        source_sha256={str(Path(__file__)): sha(__file__),
                       str(ROOT / "protocol.json"): sha(ROOT / "protocol.json")},
    )
    atomic_json(ROOT / "cpu_resource_benchmark.json", result)
    print("CPU_BENCH_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
