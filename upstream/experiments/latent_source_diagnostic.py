"""Training-only source swaps and fine-solver/support diagnostics.

Oracle paths read a native structure and are diagnostic, never AA-only results.
Changing coarse frames requires recomputing the native fine encoding.
"""
import hashlib

import numpy as np
import torch

from hierarchical_ensemble_flow import component_seed, snap_fine
from multiscale_flow_solver import integrate_velocity
from multisequence_flow_data import check_case


METHODS = [
    "native_reconstruction", "true_coarse_generated_fine16",
    "generated_coarse_reencoded_native_fine", "full16",
    "true_coarse_generated_fine64", "coarse16_fine64",
    "true_coarse_clipped_fine16", "true_coarse_snapped_fine16",
    "true_coarse_cycle_fine16",
]


def training_cases(groups, pdb_ids, samples_per_protein=4):
    if samples_per_protein < 1 or len(set(pdb_ids)) != len(pdb_ids):
        raise ValueError("Unique fixed training IDs and a positive count required")
    pdb = {case["id"]: case for case in groups["PDB_train"]}
    rows = []
    for name in pdb_ids:
        case = pdb[name]
        check_case(case, "train")
        for draw in range(samples_per_protein):
            rows.append(dict(domain="PDB_train", protein=case["protein"],
                             draw=draw, case=case))
    for name, cases in sorted(groups["train"].items()):
        indices = np.linspace(0, len(cases)-1, samples_per_protein, dtype=int)
        for draw, index in enumerate(indices):
            case = cases[int(index)]
            check_case(case, "train")
            rows.append(dict(domain="MD_train", protein=name, draw=draw, case=case))
    return rows


def diagnostic_seed(seed, case_id, draw):
    message = f"latent-source-diagnostic-v1:{seed}:{case_id}:{draw}"
    return int(hashlib.sha256(message.encode()).hexdigest()[:8], 16)


@torch.no_grad()
def sample_fine(model, aa, coarse, seed, nfe):
    rng = torch.Generator(device=aa.device).manual_seed(component_seed(seed, "fine"))
    initial = torch.randn((len(aa), model.fine_dim), generator=rng, device=aa.device)
    condition = model.normalize_coarse(coarse)
    value = integrate_velocity(initial,
        lambda state, t: model.fine_velocity(aa, state, t, condition), nfe, "euler")
    return model.denormalize_fine(value)


def fine_support(fine, codes):
    if fine.ndim != 2 or not torch.isfinite(fine).all():
        raise ValueError("Finite fine vectors required")
    _, distances = snap_fine(fine, codes)
    return dict(outside_encoder_range_fraction=float((fine.abs() > 1.).float().mean()),
                max_absolute_value=float(fine.abs().max()),
                mean_absolute_value=float(fine.abs().mean()),
                nearest_code_rmse=float(distances.mean()))


@torch.no_grad()
def decode_sources(model, codec, aa, target, seed):
    native_coarse, native_fine = codec.encode(target)
    coarse = codec.project_coarse(model.sample_coarse(
        aa, component_seed(seed, "coarse"), 16), len(aa))
    oracle16 = sample_fine(model, aa, native_coarse, seed, 16)
    generated16 = sample_fine(model, aa, coarse, seed, 16)
    # Never reuse native_fine under a different coarse-derived frame.
    reencoded = codec.fine_target(target, coarse)
    oracle_bb = codec.decode(native_coarse, oracle16)
    cycle = codec.fine_target(oracle_bb, native_coarse)
    decoded = {
        "native_reconstruction": codec.decode(native_coarse, native_fine),
        "true_coarse_generated_fine16": oracle_bb,
        "generated_coarse_reencoded_native_fine": codec.decode(coarse, reencoded),
        "full16": codec.decode(coarse, generated16),
        "true_coarse_generated_fine64": codec.decode(native_coarse,
            sample_fine(model, aa, native_coarse, seed, 64)),
        "coarse16_fine64": codec.decode(coarse,
            sample_fine(model, aa, coarse, seed, 64)),
        "true_coarse_clipped_fine16": codec.decode(native_coarse, oracle16.clamp(-1., 1.)),
        "true_coarse_snapped_fine16": codec.decode(native_coarse, oracle16, quantized=True),
        "true_coarse_cycle_fine16": codec.decode(native_coarse, cycle),
    }
    support = {name: fine_support(value, codec.fine_tokenizer.book.codes)
               for name, value in [("native", native_fine), ("true_coarse_generated", oracle16),
                                   ("generated_coarse_generated", generated16), ("cycle", cycle)]}
    fine_stats = dict(support=support,
        true_coarse_fine_cycle_rmse=float((oracle16-cycle).square().mean().sqrt()),
        native_coarse=native_coarse, generated_coarse=coarse,
        native_fine=native_fine, oracle_fine16=oracle16, generated_fine16=generated16)
    if list(decoded) != METHODS:
        raise RuntimeError("Missing diagnostic path")
    return decoded, fine_stats
