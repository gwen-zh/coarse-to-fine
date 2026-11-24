"""Read-only block-center geometry and a rigorous CA-RMSD lower bound.

For any rigid alignment, Jensen's inequality in each sequence block gives
CA MSE >= sum_b (n_b/N) * ||aligned generated center_b - native center_b||^2.
Minimizing the right side over proper rigid alignments preserves the bound.
It also holds after minimizing over a fixed reference ensemble. It is a
constraint on outputs retaining these block means, not an error attribution
percentage, an achievable decoder floor, or a claim about a different codec.
"""
import numpy as np

from multisequence_flow_data import physical_histogram_edges
from native_ensemble_metrics import distance_features, histogram_probabilities, js_divergence


def block_counts(length, block_size=4):
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 1 or block_size < 1:
        raise ValueError("Positive residue length and block size required")
    return np.minimum(block_size, length - block_size * np.arange((length + block_size - 1) // block_size)).astype(float)


def check_centers(values, length):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(block_counts(length)), 3) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite frame x ceil(N/4) x 3 centers required")
    return values


def pool_backbones(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4 or values.shape[2:] != (3, 3) or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Finite frame x residue x [N,CA,C] x xyz backbones required")
    ca = values[:, :, 1]
    return np.stack([ca[:, i:i + 4].mean(1) for i in range(0, ca.shape[1], 4)], axis=1)


def weighted_rmsd_matrix(first, second, length):
    """Optimal proper-rotation RMSD of block means, with n_b/N weights."""
    first = check_centers(first, length); second = check_centers(second, length)
    weight = block_counts(length) / length
    x = first - np.einsum("smj,m->sj", first, weight)[:, None]
    y = second - np.einsum("smj,m->sj", second, weight)[:, None]
    covariance = np.einsum("smi,m,rmj->srij", x, weight, y)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.where(np.linalg.det(u @ vt) < 0, -1., 1.)
    trace = singular[..., 0] + singular[..., 1] + sign * singular[..., 2]
    norm_x = np.einsum("smi,m,smi->s", x, weight, x)
    norm_y = np.einsum("rmi,m,rmi->r", y, weight, y)
    return np.sqrt(np.maximum(norm_x[:, None] + norm_y[None] - 2 * trace, 0.))


def scalar_features(centers, length, ca_step_cap=4.1):
    """The span flag assumes every consecutive CA distance <= declared cap.

    Adjacent block means obey d <= cap*(n_b+n_{b+1})/2 under that assumption.
    This is not an unconditional physical law; native cap violations are
    reported separately. Means can coincide, so no minimum center spacing
    is imposed.
    """
    centers = check_centers(centers, length); counts = block_counts(length)
    if len(counts) < 2 or not np.isfinite(ca_step_cap) or ca_step_cap <= 0:
        raise ValueError("At least two blocks and a positive step cap required")
    mean = np.einsum("smj,m->sj", centers, counts / length)
    rg = np.sqrt(np.einsum("smj,m,smj->s", centers - mean[:, None], counts / length, centers - mean[:, None]))
    span = np.linalg.norm(np.diff(centers, axis=1), axis=-1)
    bound = ca_step_cap * (counts[:-1] + counts[1:]) / 2
    return dict(center_rg_A=rg, adjacent_distance_A=span,
        adjacent_span_ratio=span / bound[None],
        conditional_span_flag=(span > bound[None] + 1e-8), adjacent_span_bound_A=bound)


def coarse_metrics(generated, reference, length, ca_step_cap=4.1):
    generated = check_centers(generated, length); reference = check_centers(reference, length)
    if generated.shape[1] < 5:
        raise ValueError("Existing long-range histogram descriptor needs at least five blocks")
    distance = weighted_rmsd_matrix(generated, reference, length)
    gg = weighted_rmsd_matrix(generated, generated, length)
    rr = weighted_rmsd_matrix(reference, reference, length)
    gi = np.triu_indices(len(generated), 1); ri = np.triu_indices(len(reference), 1)
    gs = float(gg[gi].mean()) if len(gi[0]) else 0.
    rs = float(rr[ri].mean()) if len(ri[0]) else 0.
    gf = scalar_features(generated, length, ca_step_cap); rf = scalar_features(reference, length, ca_step_cap)
    gd, _ = distance_features(generated); rd, _ = distance_features(reference)
    edges, rg_edges = physical_histogram_edges(generated.shape[1])
    return dict(generated_samples=len(generated), reference_frames=len(reference),
        mean_sample_nearest_reference_center_rmsd_A=float(distance.min(1).mean()),
        mean_reference_nearest_sample_center_rmsd_A=float(distance.min(0).mean()),
        generated_spread_center_rmsd_A=gs, reference_spread_center_rmsd_A=rs,
        spread_ratio=gs / rs if rs > 1e-8 else None,
        mean_generated_center_rg_A=float(gf["center_rg_A"].mean()),
        mean_reference_center_rg_A=float(rf["center_rg_A"].mean()),
        generated_adjacent_distance_mean_A=float(gf["adjacent_distance_A"].mean()),
        reference_adjacent_distance_mean_A=float(rf["adjacent_distance_A"].mean()),
        generated_conditional_span_flag_rate=float(gf["conditional_span_flag"].mean()),
        reference_conditional_span_flag_rate=float(rf["conditional_span_flag"].mean()),
        js_pairwise_center_distance_nats=float(js_divergence(histogram_probabilities(gd, edges), histogram_probabilities(rd, edges)).mean()),
        js_center_rg_nats=float(js_divergence(histogram_probabilities(gf["center_rg_A"][:, None], rg_edges),
            histogram_probabilities(rf["center_rg_A"][:, None], rg_edges)).mean()))
