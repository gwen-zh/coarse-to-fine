"""Controlled refinement experiments with a bitwise shared frozen coarse policy."""
import copy
import hashlib
import torch
from torch import nn

from depth_generator import DepthGenerator


def generation_seed(target, sample, seed=0):
    return (seed * 1000000 + sample * 10000 +
            int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)) % (2 ** 32)


class FrozenCoarseRefiner(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.refiner = copy.deepcopy(base).requires_grad_(True)
        self.base = base.requires_grad_(False).eval()
        self.sizes = base.sizes

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, features, sequence, lower, current, stage):
        if stage == 0:
            with torch.no_grad():
                return self.base(features, sequence, lower, current, stage)
        return self.refiner(features, sequence, lower, current, stage)

    generate = DepthGenerator.generate


@torch.no_grad()
def compensation_labels(bridge, latent, lower, stage):
    if bridge.mode != 'residual' or len(lower) != stage or stage < 1:
        raise ValueError('Expected a complete available residual prefix')
    prefix = torch.stack(lower, dim=-1)
    remaining = latent - bridge.prefix_latent(prefix)
    return bridge.nearest(remaining, getattr(bridge, f'centers_{stage}'))


@torch.no_grad()
def greedy_oracle_refinement(bridge, latent, coarse):
    """Reference-latent diagnostic, not a generative procedure or global bound."""
    lower = [coarse]
    for stage in range(1, len(bridge.sizes)):
        lower.append(compensation_labels(bridge, latent, lower, stage))
    return torch.stack(lower, dim=-1)
