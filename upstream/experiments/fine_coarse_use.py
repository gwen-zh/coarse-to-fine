"""Cross the physical decoder coarse and the fine sampler's conditioning source."""
import torch


CELLS = ["ownC_ownF", "priorC_ownF", "ownC_priorF", "priorC_priorF"]
COMPARISONS = {"same_own_coarse": ("ownC_ownF", "ownC_priorF"),
               "same_prior_coarse": ("priorC_ownF", "priorC_priorF")}


def crossed_inputs(own_coarse, own_fine, prior_coarse, prior_fine):
    if own_fine.ndim != 3 or own_fine.shape[-1] != 8 or own_fine.shape != prior_fine.shape:
        raise ValueError("Matching sample x residue x fine8 arrays required")
    samples, length, _ = own_fine.shape
    expected = (samples, (length + 3) // 4, 3)
    if own_coarse.shape != expected or prior_coarse.shape != expected:
        raise ValueError("Coarse/sample/residue correspondence differs")
    if not all(torch.isfinite(x).all() for x in [own_coarse, own_fine, prior_coarse, prior_fine]):
        raise ValueError("Finite cached latent arrays required; never omit failures")
    return {"ownC_ownF": (own_coarse, own_fine),
            "priorC_ownF": (prior_coarse, own_fine),
            "ownC_priorF": (own_coarse, prior_fine),
            "priorC_priorF": (prior_coarse, prior_fine)}


def response(own_fine, prior_fine, fine_std):
    scale = torch.as_tensor(fine_std, dtype=own_fine.dtype, device=own_fine.device)
    if own_fine.shape != prior_fine.shape or scale.shape != (8,) or not (scale > 0).all():
        raise ValueError("Matching latent pairs and positive fixed fine normalization required")
    delta = prior_fine - own_fine
    if not torch.isfinite(delta).all() or not torch.isfinite(scale).all():
        raise ValueError("Finite latent response required")
    return dict(fine_change_RMS=float(delta.square().mean().sqrt()),
        train_normalized_fine_change_RMS=float((delta / scale).square().mean().sqrt()),
        maximum_component_change=float(delta.abs().max()))


def validate_cells(records):
    if len(records) != 4 or {r["cell"] for r in records} != set(CELLS):
        raise ValueError("Exactly one result for each declared crossed cell required")
    if len({(r["seed"], r["protein"], r["gauge"]) for r in records}) != 1:
        raise ValueError("Crossed cells mix sequences, seeds or gauges")
    return {r["cell"]: r for r in records}
