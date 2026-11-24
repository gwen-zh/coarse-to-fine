"""Same fine flow, with an optional strictly positive early-time density.

uniform: t=U; early: density 2.5 on [0,.25], .5 on (.25,1].
The inverse CDF couples both arms to the same Gaussian and uniform draws.
Changing only a positive time density preserves the pointwise population-MSE
minimizer, but can change finite-capacity optimization and endpoint quality.
It neither guarantees state separation nor proves ensemble preservation.
"""
import hashlib

import torch


def transform_time(uniform, mode):
    if mode not in {"uniform", "early"}:
        raise ValueError("Declare uniform or early time sampling")
    if not torch.isfinite(uniform).all() or (uniform < 0).any() or (uniform > 1).any():
        raise ValueError("Uniform draws must lie in [0,1]")
    if mode == "uniform":
        return uniform
    return torch.where(uniform <= .625, uniform / 2.5, 2 * uniform - 1)


def fine_time_loss(model, aa, coarse, fine, generator, mode):
    model = model.fine_model
    f = model.normalize_fine(fine.detach())
    noise = torch.randn(f.shape, generator=generator, device=f.device, dtype=f.dtype)
    uniform = torch.rand((), generator=generator, device=f.device, dtype=f.dtype)
    t = transform_time(uniform, mode)
    state = (1 - t) * noise + t * f
    velocity = model.fine_velocity(aa, state, t, model.normalize_coarse(coarse.detach()))
    loss = (velocity - (f - noise)).square().mean()
    digest = hashlib.sha256(noise.detach().cpu().numpy().tobytes()
                            + uniform.detach().cpu().numpy().tobytes()).hexdigest()
    return loss, dict(flow=float(loss.detach()), time=float(t), uniform_time=float(uniform),
                      draw_sha256=digest)
