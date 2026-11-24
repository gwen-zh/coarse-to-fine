"""Read-only trajectory and training-gradient diagnostics for coarse flows.

Distance-feature variance excludes pure rigid-body motion. Gradient projections
are local first-order SGD directions, not observed AdamW updates or proof of
the cause of an entire training trajectory. No optimizer is created here.
"""
from contextlib import contextmanager
import hashlib

import torch

from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case


@contextmanager
def same_forward_kernel():
    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def project(centers, length):
    counts = torch.minimum(torch.full((len(centers),), 4., device=centers.device, dtype=centers.dtype),
        length - 4 * torch.arange(len(centers), device=centers.device, dtype=centers.dtype))
    if centers.shape != ((length + 3) // 4, 3) or (counts <= 0).any():
        raise ValueError("Matching physical block centers required")
    return centers - (centers * counts[:, None]).sum(0) / length


def trajectory(branch, aa, noise, steps=16):
    """Store physical centered snapshots; do not alter the integrated states."""
    if steps < 1 or noise.shape != ((len(aa) + 3) // 4, 3):
        raise ValueError("Positive steps and matching coarse noise required")
    state = noise
    path = [project(branch.denormalize_coarse(state), len(aa))]
    for i in range(steps):
        state = state + branch.coarse_velocity(aa, state, i / steps) / steps
        path.append(project(branch.denormalize_coarse(state), len(aa)))
    return torch.stack(path)


def distance_features(centers):
    if centers.ndim != 3 or centers.shape[-1] != 3 or centers.shape[1] < 5 or not torch.isfinite(centers).all():
        raise ValueError("Ensemble x at least5 centers x XYZ required")
    i, j = torch.triu_indices(centers.shape[1], centers.shape[1], offset=4, device=centers.device)
    return torch.linalg.vector_norm(centers[:, i] - centers[:, j], dim=-1)


def distance_variance(centers):
    if len(centers) < 2: raise ValueError("At least two noise draws required")
    return distance_features(centers).var(0, unbiased=False).mean()


@torch.no_grad()
def observable_moments(generated, reference):
    """Distance-observable bias and spread; never average Cartesian structures."""
    g = distance_features(generated.double()); r = distance_features(reference.double())
    bias = (g.mean(0) - r.mean(0)).square().mean()
    gv = g.var(0, unbiased=False).mean(); rv = r.var(0, unbiased=False).mean()
    return dict(distance_mean_bias_RMSE_A=float(bias.sqrt()), generated_distance_variance_A2=float(gv),
        reference_distance_variance_A2=float(rv), distance_variance_ratio=float(gv / rv) if rv > 1e-12 else None)


def parameters(branch):
    return [v for v in branch.parameters() if v.requires_grad]


def flatten_gradient(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    value = torch.cat([torch.zeros_like(v).flatten() if g is None else g.detach().flatten()
                       for v, g in zip(params, grads)]).double()
    if not torch.isfinite(value).all(): raise RuntimeError("Nonfinite diagnostic gradient")
    return value


def event_loss(branch, teacher, case, seed, mode):
    check_case(case, "train")
    if mode not in {"native", "parent_replay"}: raise ValueError("Declared diagnostic loss required")
    aa = encode_aa(case["sequence"]); target = branch.normalize_coarse(case["coarse"].detach())
    rng = torch.Generator().manual_seed(seed)
    noise = torch.randn(target.shape, generator=rng, dtype=target.dtype)
    time = torch.rand((), generator=rng, dtype=target.dtype)
    state = (1 - time) * noise + time * target
    with same_forward_kernel():
        velocity = branch.coarse_velocity(aa, state, time)
        if mode == "parent_replay":
            with torch.no_grad(): wanted = teacher.coarse_velocity(aa, state, time)
        else: wanted = target - noise
    digest = hashlib.sha256(noise.numpy().tobytes() + time.numpy().tobytes()).hexdigest()
    return (velocity - wanted).square().mean(), digest


def mean_event_gradient(branch, teacher, events, registry, mode):
    if not events: raise ValueError("Nonempty fixed training events required")
    params = parameters(branch)
    total = torch.zeros(sum(v.numel() for v in params), dtype=torch.float64)
    losses = []
    for event in events:
        case = registry[event["case_id"]]
        loss, digest = event_loss(branch, teacher, case, event["seed"], mode)
        if digest != event["draw_sha256"]: raise RuntimeError("Training Gaussian/time replay failed")
        total += flatten_gradient(loss, params) / len(events)
        losses.append(float(loss.detach()))
    if not torch.isfinite(total).all(): raise RuntimeError("Nonfinite diagnostic domain gradient")
    return total, dict(events=len(events), mean_loss=sum(losses) / len(losses), gradient_norm=float(total.norm()))


def gradient_projection(spread_gradient, loss_gradient):
    spread_gradient = spread_gradient.double(); loss_gradient = loss_gradient.double()
    sn = float(spread_gradient.norm()); ln = float(loss_gradient.norm())
    if sn <= 1e-12 or ln <= 1e-12:
        return dict(spread_gradient_norm=sn, loss_gradient_norm=ln, cosine=None,
            first_order_spread_change_per_unit_parameter_move=None)
    dot = float(torch.dot(spread_gradient, loss_gradient))
    return dict(spread_gradient_norm=sn, loss_gradient_norm=ln, cosine=dot / (sn * ln),
        first_order_spread_change_per_unit_parameter_move=-dot / ln)


@torch.no_grad()
def response_gain(branch, aa, noise, direction, epsilon, steps=16):
    if epsilon <= 0 or noise.shape != direction.shape: raise ValueError("Matching perturbation and positive epsilon required")
    direction = direction / direction.square().mean().sqrt().clamp_min(1e-12)
    plus = trajectory(branch, aa, noise + epsilon * direction, steps)[-1:]
    minus = trajectory(branch, aa, noise - epsilon * direction, steps)[-1:]
    delta = (distance_features(plus) - distance_features(minus)) / (2 * epsilon)
    return float(delta.square().mean().sqrt())


@torch.no_grad()
def bridge_endpoint(branch, aa, coarse, noise, time):
    if not 0 < time < 1: raise ValueError("Interior bridge time required")
    target = branch.normalize_coarse(coarse)
    state = (1 - time) * noise + time * target
    predicted = state + (1 - time) * branch.coarse_velocity(aa, state, time)
    return project(branch.denormalize_coarse(predicted), len(aa))
