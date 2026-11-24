"""Reference-free physical coarse prompts from AA-only predicted structures."""
import torch

from block_local_frames import FrameGeometryError
from hierarchical_latent_contract import pool_ca


GAUGES = ["raw", "first_residue", "align_to_own"]


def center_distances(centers):
    """Direct FP64 differences avoid cancellation in FP32 cdist's MM path."""
    x = centers.double()
    return torch.linalg.vector_norm(x[:, None] - x[None], dim=-1)


def counts(length, reference):
    return reference.new_tensor([min(4, length - 4 * i) for i in range((length + 3) // 4)])


def proper_rotation(source, target, length):
    """Weighted Kabsch; target is another AA-only prediction, never native."""
    if source.shape != target.shape or source.shape != ((length + 3) // 4, 3):
        raise ValueError("Matching physical block centers required")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Finite physical block centers required")
    dtype = source.dtype; x = source.double(); y = target.double()
    weight = counts(length, x) / length
    x = x - (weight[:, None] * x).sum(0)
    y = y - (weight[:, None] * y).sum(0)
    u, _, vh = torch.linalg.svd((weight[:, None] * x).T @ y)
    correction = torch.eye(3, dtype=x.dtype, device=x.device)
    correction[-1, -1] = torch.linalg.det(u @ vh)
    return (u @ correction @ vh).to(dtype)


def coarse_prompt(predicted_backbone, own_centers, gauge):
    """Return centered physical means and the applied proper rotation.

    No native structure, fine label, or score is an argument. own_centers is
    used only to set orientation in align_to_own; distances cannot change.
    """
    x = predicted_backbone
    if gauge not in GAUGES or x.ndim != 3 or x.shape[1:] != (3, 3) or not torch.isfinite(x).all():
        raise ValueError("Declared gauge and finite N/CA/C prediction required")
    n = len(x)
    centered = x - x[:, 1].mean(0)
    centers = pool_ca(centered)
    if own_centers.shape != centers.shape:
        raise ValueError("Own coarse prediction length differs")
    rotation = torch.eye(3, dtype=x.dtype, device=x.device)
    if gauge == "first_residue":
        first = x[0, 1] - x[0, 2]
        if first.norm() <= 1e-6:
            raise FrameGeometryError("Predicted first residue lacks a CA-C axis")
        first = first / first.norm()
        second = x[0, 0] - x[0, 1]
        second = second - (first * second).sum() * first
        if second.norm() <= 1e-6:
            raise FrameGeometryError("Predicted first residue lacks a backbone plane")
        second = second / second.norm()
        third = torch.cross(first, second, dim=0)
        rotation = torch.stack([first, second, third], -1)
    elif gauge == "align_to_own":
        rotation = proper_rotation(centers, own_centers, n)
    result = centers @ rotation
    # Remove FP32 translation roundoff using residue-count weights.
    result = result - (counts(n, result)[:, None] * result).sum(0) / n
    return result, rotation
