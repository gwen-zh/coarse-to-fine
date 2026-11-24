"""Coarse-only learnability with an independently frozen fine generator."""
import hashlib
import math

import torch
from torch import nn


class FixedFineCoarseFlow(nn.Module):
    def __init__(self, coarse_model, fine_model):
        super().__init__()
        if coarse_model is fine_model:
            raise ValueError("Separate coarse and fine parameter objects required")
        if coarse_model.coarse_dim != 3 or fine_model.coarse_dim != 3:
            raise ValueError("Physical centers required")
        if {id(v) for v in coarse_model.parameters()} & {id(v) for v in fine_model.parameters()}:
            raise ValueError("Coarse and fine cannot share trainable parameter objects")
        self.coarse_model = coarse_model.requires_grad_(True)
        # These modules have no role in coarse_velocity; keep them immutable.
        for name in ["fine_head", "fine_input", "coarse_to_fine"]:
            getattr(self.coarse_model, name).requires_grad_(False)
        self.fine_model = fine_model.eval().requires_grad_(False)
        self.coarse_dim = 3; self.fine_dim = fine_model.fine_dim
        self.config = dict(coarse_model=coarse_model.config, fine_generator="frozen_geometry_aware_candidate")

    def train(self, mode=True):
        super().train(mode)
        self.fine_model.eval()
        return self

    def trainable_parameters(self):
        return [v for v in self.coarse_model.parameters() if v.requires_grad]

    def sample_coarse(self, *args, **kwargs):
        return self.coarse_model.sample_coarse(*args, **kwargs)

    def sample_fine(self, *args, **kwargs):
        return self.fine_model.sample_fine(*args, **kwargs)


def coarse_flow_loss(model, aa, coarse, generator):
    branch = model.coarse_model
    target = branch.normalize_coarse(coarse.detach())
    noise = torch.randn(target.shape, generator=generator, device=target.device, dtype=target.dtype)
    time = torch.rand((), generator=generator, device=target.device, dtype=target.dtype)
    state = (1 - time) * noise + time * target
    velocity = branch.coarse_velocity(aa, state, time)
    loss = (velocity - (target - noise)).square().mean()
    digest = hashlib.sha256(noise.detach().cpu().numpy().tobytes() + time.detach().cpu().numpy().tobytes()).hexdigest()
    return loss, dict(draw_sha256=digest, time=float(time), flow=float(loss.detach()))


def event(step, targets=8, batch_size=4):
    if step < 0 or targets < 1 or batch_size < 1:
        raise ValueError("Nonnegative step, positive target count and batch required")
    target = step % targets
    first = (step // targets) * batch_size
    return target, list(range(first, first + batch_size))


def draw_seed(seed, case_id, phase, index):
    return int(hashlib.sha256(f"coarse-few-target-v1:{seed}:{case_id}:{phase}:{index}".encode()).hexdigest()[:8], 16)


def learning_rate(step, cycle=2048, maximum=3e-4, minimum=3e-5):
    if step < 0 or cycle < 2:
        raise ValueError("Nonnegative step and cycle of at least two updates required")
    return minimum + .5 * (maximum - minimum) * (1 + math.cos(math.pi * (step % cycle) / (cycle - 1)))
