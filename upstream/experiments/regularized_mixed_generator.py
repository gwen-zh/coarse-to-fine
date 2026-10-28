"""Smaller regularized generators on the unchanged two-level representation.

The metric arm changes only the fine output head: a bounded 8D point is scored
against frozen learned code vectors. This is an empirical geometry hypothesis,
not a guarantee that code distance equals decoded structural distance.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mixed_hierarchical_generator import MixedHierarchicalGenerator


def set_dropout(module, probability):
    if not 0 <= probability < 1:
        raise ValueError('Dropout probability outside [0,1)')
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = probability
        elif isinstance(child, nn.MultiheadAttention):
            child.dropout = probability


class MetricCodeHead(nn.Module):
    def __init__(self, width, codes, minimum_temperature=.02, maximum_temperature=.5):
        super().__init__()
        codes = torch.as_tensor(codes, dtype=torch.float32).detach().clone()
        if codes.ndim != 2 or not torch.isfinite(codes).all():
            raise ValueError('Finite code matrix required')
        self.register_buffer('codes', codes)
        self.point = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, codes.shape[1]), nn.Tanh())
        self.minimum_temperature = minimum_temperature
        self.maximum_temperature = maximum_temperature
        ratio = (.05 - minimum_temperature) / (maximum_temperature - minimum_temperature)
        self.temperature_logit = nn.Parameter(torch.tensor(math.log(ratio / (1 - ratio))))

    def temperature(self):
        return self.minimum_temperature + (self.maximum_temperature - self.minimum_temperature) * self.temperature_logit.sigmoid()

    def forward(self, hidden):
        point = self.point(hidden)
        distances = (point[..., None, :] - self.codes).square().mean(-1)
        return -distances / self.temperature()


class RegularizedMixedGenerator(MixedHierarchicalGenerator):
    def __init__(self, mean, std, condition_dim=1536, coarse_dim=16, width=128,
                 coarse_layers=2, fine_layers=2, fine_size=1024, dropout=.1,
                 fine_head='ce', codes=None):
        super().__init__(mean, std, condition_dim, coarse_dim, width,
                         coarse_layers, fine_layers, fine_size)
        if fine_head not in ['ce', 'metric']:
            raise ValueError('Unknown fine head')
        if fine_head == 'metric':
            if codes is None or len(codes) != fine_size:
                raise ValueError('Frozen fine codebook required for metric head')
            self.fine.head = MetricCodeHead(width, codes)
        set_dropout(self.coarse_flow, dropout)
        set_dropout(self.fine, dropout)
        self.config.update(dropout=dropout, fine_head=fine_head)


def fine_training_loss(generator, features, native_coarse, target, jitter_std=.05,
                       label_smoothing=.05, rate=None):
    """Keep conditions near the matching conformation, rather than resampling it."""
    if not 0 <= label_smoothing < 1 or jitter_std < 0:
        raise ValueError('Invalid regularization')
    condition = generator.normalize(native_coarse)
    if jitter_std:
        condition = generator.normalize(generator.denormalize(
            condition + jitter_std * torch.randn_like(condition)).clamp(-1, 1))
    rate = float(torch.rand((), device=features.device) * .9 + .1) if rate is None else float(rate)
    if not 0 < rate <= 1:
        raise ValueError('Mask rate outside (0,1]')
    mask = torch.rand(len(target), device=target.device) < rate
    if not mask.any():
        mask[torch.randint(len(mask), (1,), device=mask.device)] = True
    corrupted = torch.where(mask, generator.fine.fine_size, target)
    logits = generator.fine(features, condition, corrupted, rate)
    loss = F.cross_entropy(logits[mask], target[mask], label_smoothing=label_smoothing)
    with torch.no_grad():
        nll = F.cross_entropy(logits[mask], target[mask])
        accuracy = (logits[mask].argmax(-1) == target[mask]).float().mean()
    return loss, dict(smoothed_ce=loss.detach(), nll=nll, accuracy=accuracy)
