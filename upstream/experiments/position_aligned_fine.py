"""Fine training with train-only positional support and explicit scale alignment."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mixed_hierarchical_generator import ContinuousFineGenerator, pooled_sequence
from two_level_autoencoder import positions
from regularized_mixed_generator import set_dropout
from position_supported_fine import support_rows, sample_supported

ARMS = ['support_only', 'position_only', 'position_coarse']


class PositionedFineGenerator(ContinuousFineGenerator):
    def __init__(self, condition_dim=1536, coarse_dim=16, width=128, layers=2,
                 fine_size=1024, dropout=.1, arm='support_only'):
        if arm not in ARMS:
            raise ValueError('Unknown fine alignment arm')
        super().__init__(condition_dim, coarse_dim, width, layers, fine_size)
        self.arm = arm
        if arm != 'support_only':
            self.block_position = nn.Embedding(4, width)
            nn.init.zeros_(self.block_position.weight)
        if arm == 'position_coarse':
            # Exact warm-start equivalence; learn how much local coarse to add.
            self.local_coarse_gate = nn.Parameter(torch.zeros(()))
        set_dropout(self, dropout)

    def forward(self, features, coarse, fine, mask_rate):
        if self.arm == 'support_only':
            return super().forward(features, coarse, fine, mask_rate)
        n = len(features)
        m = (n + 3) // 4
        if coarse.shape != (m, self.coarse_dim) or not coarse.is_floating_point() or fine.shape != (n,):
            raise ValueError('Continuous coarse/fine length mismatch')
        if not torch.isfinite(coarse).all() or fine.min() < 0 or fine.max() > self.fine_size:
            raise ValueError('Invalid coarse or fine values')
        h = self.condition(features)
        projected = self.coarse_projection(coarse)
        c = pooled_sequence(h) + projected + self.type_embedding.weight[0]
        f = h + self.fine_embedding(fine) + self.type_embedding.weight[1]
        f = f + self.block_position(torch.arange(n, device=h.device) % 4)
        if self.arm == 'position_coarse':
            f = f + self.local_coarse_gate.tanh() * projected.repeat_interleave(4, 0)[:n]
        c = c + positions(m, h.device, h.dtype, h.shape[-1])
        f = f + positions(n, h.device, h.dtype, h.shape[-1])
        freq = torch.exp(torch.linspace(0, math.log(1000.), 8, device=h.device, dtype=h.dtype))
        phase = torch.as_tensor(mask_rate, device=h.device, dtype=h.dtype) * freq
        time = self.time(torch.cat([phase.sin(), phase.cos()]))
        output = self.layers((torch.cat([c, f]) + time)[None])[0]
        return self.head(output[m:])


class PositionedFineModel(nn.Module):
    def __init__(self, mean, std, counts, **config):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        counts = torch.as_tensor(counts, dtype=torch.long)
        if mean.ndim != 1 or mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Finite train-only normalization required')
        if counts.shape != (config.get('fine_size', 1024), 4):
            raise ValueError('Training support vocabulary mismatch')
        support_rows(counts, 4, counts.device)
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        self.register_buffer('position_counts', counts.clone())
        self.fine = PositionedFineGenerator(**config)
        self.config = dict(config)

    def normalize(self, coarse):
        return (coarse - self.mean) / self.std

    def denormalize(self, coarse):
        return coarse * self.std + self.mean

    @torch.no_grad()
    def sample(self, features, coarse, seed, steps=16):
        return sample_supported(self.fine, features, self.normalize(coarse),
                                self.position_counts, seed, steps)


def supported_cross_entropy(logits, target, support, smoothing=0.):
    if logits.shape != support.shape or target.shape != logits.shape[:-1] or not 0 <= smoothing < 1:
        raise ValueError('Invalid supported loss inputs')
    if not support.any(-1).all() or not support.gather(-1, target[..., None]).all():
        raise ValueError('Target outside TRAIN positional support; cannot silently omit it')
    logp = logits.masked_fill(~support, -torch.inf).log_softmax(-1)
    nll = -logp.gather(-1, target[..., None]).squeeze(-1)
    # Uniform smoothing only across allowed classes. No 0 * infinity terms.
    smooth = -logp.masked_fill(~support, 0.).sum(-1) / support.sum(-1)
    return ((1-smoothing)*nll + smoothing*smooth).mean(), nll.mean()


def denoise_loss(model, features, native_coarse, target, jitter_std=.05,
                 smoothing=.05, rate=None):
    if jitter_std < 0:
        raise ValueError('Nonnegative jitter required')
    coarse = model.normalize(native_coarse)
    if jitter_std:
        coarse = model.normalize(model.denormalize(coarse + jitter_std*torch.randn_like(coarse)).clamp(-1, 1))
    rate = float(torch.rand((), device=features.device)*.9+.1) if rate is None else float(rate)
    if not 0 < rate <= 1:
        raise ValueError('Mask rate outside (0,1]')
    mask = torch.rand(len(target), device=target.device) < rate
    if not mask.any():
        mask[torch.randint(len(mask), (1,), device=mask.device)] = True
    corrupted = torch.where(mask, model.fine.fine_size, target)
    logits = model.fine(features, coarse, corrupted, rate)
    support = support_rows(model.position_counts, len(target), target.device) > 0
    loss, nll = supported_cross_entropy(logits[mask], target[mask], support[mask], smoothing)
    return loss, dict(loss=loss.detach(), nll=nll.detach())
