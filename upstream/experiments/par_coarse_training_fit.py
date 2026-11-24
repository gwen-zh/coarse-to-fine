"""Fixed cohort and paired corruption utilities for a read-only fit diagnosis."""
from collections import Counter

import numpy as np
import torch

from par_coarse_transfer import draw


def presented(trace):
    return Counter(event["case_id"] for row in trace for event in row["batch"])


def select_targets(groups, development, traces, parent_targets):
    counts = {seed: presented(trace) for seed, trace in traces.items()}
    common = set.intersection(*(set(value) for value in counts.values()))
    result = []
    for protein, cases in sorted(groups["train"].items()):
        seen = sorted((c for c in cases if c["id"] in common), key=lambda c: c["id"])
        if len(seen) < 4:
            raise ValueError("Need four MD cases actually presented in both training seeds")
        sample = cases[0]
        result.append(dict(protein=protein, sequence=sample["sequence"], length=len(sample["sequence"]),
            domain="MD_train", probe_case_ids=[c["id"] for c in seen[:4]],
            reference_indices=reference_indices(len(cases)), reference_count=min(128, len(cases)),
            exposure={str(seed):sum(counter.get(c["id"], 0) for c in cases) for seed,counter in counts.items()},
            probe_exposure={str(seed):{c["id"]:counter[c["id"]] for c in seen[:4]} for seed,counter in counts.items()}))
    used_sequences = {r["sequence"] for r in result} | {r["sequence"] for r in parent_targets}
    candidates = [c for c in groups["PDB_train"] if c["id"] in common and c["sequence"] not in used_sequences]
    for length in [103, 109]:
        if not candidates:
            raise ValueError("No exposed PDB cases remain")
        chosen = min(candidates, key=lambda c:(abs(len(c["sequence"])-length), c["id"]))
        result.append(dict(protein=chosen["protein"], sequence=chosen["sequence"], length=len(chosen["sequence"]),
            domain="PDB_train", probe_case_ids=[chosen["id"]], reference_indices=[0], reference_count=1,
            exposure={str(seed):counter[chosen["id"]] for seed,counter in counts.items()},
            probe_exposure={str(seed):{chosen["id"]:counter[chosen["id"]]} for seed,counter in counts.items()}))
        candidates = [c for c in candidates if c["sequence"] != chosen["sequence"]]
    for target in parent_targets:
        cases = development["validation"][target["protein"]]
        indices = reference_indices(len(cases))
        chosen = [cases[i] for i in reference_indices(len(cases), 4)]
        if any(c["id"] in common for c in chosen):
            raise ValueError("Development probe was presented in training")
        result.append(dict(protein=target["protein"], sequence=target["sequence"], length=target["length"],
            domain="development", probe_case_ids=[c["id"] for c in chosen],
            reference_indices=indices, reference_count=len(indices), exposure={str(seed):0 for seed in counts}))
    if len({r["protein"] for r in result}) != len(result):
        raise ValueError("Ambiguous target identifiers")
    return result


def reference_indices(count, budget=128):
    if count < 1 or budget < 1:
        raise ValueError("A positive reference cohort is required")
    return np.linspace(0, count-1, min(count, budget), dtype=int).tolist()


def fixed_probe(case, seed, case_index, replicate, time):
    """Preserve the original proper rotation/OT noise, then set a declared time.

    This uses native coordinates as a denoising probe. It is not AA-only
    generation and cannot be interpreted as generated structure accuracy.
    """
    if time not in [.05, .5, .9] or replicate not in [0, 1]:
        raise ValueError("Use the fixed denoising probe grid")
    state, velocity, sampled_time, self_condition, digest = draw(case["coarse"], seed, 800000+case_index, replicate)
    noise = state-sampled_time*velocity
    return noise+time*velocity, velocity, time, self_condition, digest

