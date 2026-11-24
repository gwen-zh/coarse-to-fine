"""Balanced training draws and strict frame conventions for multiple proteins."""
import hashlib

import numpy as np
import torch

from atlas_shared_frame import align_backbone_to_training_reference
from hierarchical_rigid_vq import canonical_backbone


def align_trajectory(backbones, reference):
    """One reference per protein; do not recanonicalize every moving residue 1."""
    return [align_backbone_to_training_reference(torch.as_tensor(x).float(), reference)
            for x in backbones]


def canonical_training_reference(backbone):
    return canonical_backbone(torch.as_tensor(backbone).float())


def check_case(case, expected_split):
    if case["split"] != expected_split:
        raise ValueError("A development protein entered a training batch")
    if len(case["sequence"]) != len(case["target"]):
        raise ValueError("Sequence and backbone residue counts differ")
    if case["coarse"].shape != ((len(case["sequence"]) + 3) // 4, 3):
        raise ValueError("Physical block-center shape mismatch")


def balanced_batch(md_groups, pdb_cases, rng, replay):
    """Always draw the same four MD slots; replace only slot four with PDB."""
    names = sorted(md_groups)
    chosen = []
    for _ in range(4):
        name = names[int(rng.integers(len(names)))]
        cases = md_groups[name]
        case = cases[int(rng.integers(len(cases)))]
        check_case(case, "train")
        chosen.append(case)
    pdb = pdb_cases[int(rng.integers(len(pdb_cases)))]
    check_case(pdb, "train")
    if replay:
        chosen[-1] = pdb
    return chosen


def slot_seed(training_seed, step, slot):
    message = f"multisequence-flow-v1:{training_seed}:{step}:{slot}"
    return int(hashlib.sha256(message.encode()).hexdigest()[:8], 16)


def fit_train_normalization(md_groups):
    """Equal weight per training protein; PDB and development never fit stats."""
    moments = {"coarse": [], "fine": []}
    for cases in md_groups.values():
        for case in cases:
            check_case(case, "train")
        for name, key in [("coarse", "coarse"), ("fine", "fine_continuous")]:
            values = torch.cat([c[key] for c in cases])
            moments[name].append((values.mean(0), values.square().mean(0)))
    result = {}
    for name, rows in moments.items():
        mean = torch.stack([row[0] for row in rows]).mean(0)
        second = torch.stack([row[1] for row in rows]).mean(0)
        result[name + "_mean"] = mean.tolist()
        result[name + "_std"] = (second - mean.square()).clamp_min(.05 ** 2).sqrt().tolist()
    return result


def physical_histogram_edges(length):
    """Predeclared physical bins; new-protein reference data do not fit bins."""
    i, _ = np.triu_indices(length, 4)
    distance = np.concatenate(([-np.inf], np.arange(0., 66., 2.), [np.inf]))
    rg = np.concatenate(([-np.inf], np.arange(0., 33., 1.), [np.inf]))
    return np.repeat(distance[None], len(i), axis=0), rg[None]
