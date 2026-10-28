"""Evaluation summaries independent of the feedback acceptance score."""
import numpy as np


def aligned_rmsd(mobile, reference):
    mobile, reference = np.asarray(mobile, float), np.asarray(reference, float)
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Coordinate shapes must agree and be [N,3]")
    if not np.isfinite(mobile).all() or not np.isfinite(reference).all():
        return None
    x, y = mobile - mobile.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return float(np.sqrt(np.mean(np.sum((x @ (u @ correction @ vt) - y) ** 2, axis=-1))))


def summarize_backbone(atom37, references=()):
    xyz = np.asarray(atom37, float)
    ca = xyz[:, 1]
    finite = np.isfinite(ca).all()
    result = {"all_ca_finite": bool(finite)}
    if not finite:
        return result
    pairwise = np.linalg.norm(ca[:, None] - ca[None], axis=-1)
    distances = pairwise[np.triu_indices(len(ca), 1)]
    # Same 3.0 A cutoff and self-only exclusion as upstream validity().
    result["upstream_ca_clash_valid"] = bool((distances >= 3.0).all())
    result["upstream_ca_clashes"] = int((distances < 3.0).sum())
    result["rg"] = float(np.sqrt(np.mean(np.sum((ca - ca.mean(0)) ** 2, axis=-1))))
    bonds = np.linalg.norm(xyz[:-1, 2] - xyz[1:, 0], axis=-1)
    result["backbone_cn_finite"] = bool(np.isfinite(bonds).all())
    result["backbone_cn_outliers"] = int((~np.isfinite(bonds) | (bonds < 1.0) | (bonds > 1.8)).sum())
    rmsds = [aligned_rmsd(ca, ref) for ref in references]
    rmsds = [r for r in rmsds if r is not None]
    result["closest_available_reference_ca_rmsd"] = min(rmsds) if rmsds else None
    return result


def ensemble_summary(ca_samples, references=()):
    pairs = [aligned_rmsd(x, y) for i, x in enumerate(ca_samples) for y in ca_samples[i + 1:]]
    pairs = [r for r in pairs if r is not None]
    coverage = [[aligned_rmsd(sample, ref) for sample in ca_samples] for ref in references]
    return dict(mean_pairwise_ca_rmsd=float(np.mean(pairs)) if pairs else None,
                reference_best_ca_rmsd=[min(r for r in values if r is not None)
                                        if any(r is not None for r in values) else None
                                        for values in coverage])
