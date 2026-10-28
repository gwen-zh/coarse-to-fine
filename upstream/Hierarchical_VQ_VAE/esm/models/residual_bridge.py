"""Inference bridge from residual/parallel prefix tokens to original fine IDs.

The frozen structure decoder consumes the resulting fine IDs. This module does
not predict new tokens and contains no trained conformation generator.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualTokenBridge(nn.Module):
    def __init__(self, centers, fine_codes, mode='residual'):
        super().__init__()
        if mode not in ('residual', 'parallel') or not centers:
            raise ValueError('Invalid quantization mode or empty dictionary list')
        width = fine_codes.shape[-1]
        for table in [*centers, fine_codes]:
            if table.ndim != 2 or table.shape[-1] != width or not table.is_floating_point() or not torch.isfinite(table).all():
                raise ValueError('Codebooks must be finite floating matrices of a common width')
        self.mode = mode
        self.sizes = tuple(len(c) for c in centers)
        for i, table in enumerate(centers):
            self.register_buffer(f'centers_{i}', table.detach().clone())
        self.register_buffer('fine_codes', fine_codes.detach().clone())

    @staticmethod
    def nearest(z, centers):
        distance = z.square().sum(-1, keepdim=True) - 2 * z @ centers.T + centers.square().sum(-1)
        return distance.argmin(-1)

    @torch.no_grad()
    def encode(self, latent):
        accumulated = torch.zeros_like(latent)
        tokens = []
        for level in range(len(self.sizes)):
            table = getattr(self, f'centers_{level}')
            source = latent - accumulated if self.mode == 'residual' else latent
            ids = self.nearest(source, table)
            tokens.append(ids)
            accumulated = accumulated + F.embedding(ids, table)
        return torch.stack(tokens, dim=-1)

    @torch.no_grad()
    def prefix_latent(self, tokens):
        if tokens.dtype != torch.long or tokens.ndim < 2:
            raise ValueError('Expected long tokens shaped [..., residues, prefix_levels]')
        levels = tokens.shape[-1]
        if not 1 <= levels <= len(self.sizes):
            raise ValueError('Invalid number of prefix levels')
        result = None
        for level in range(levels):
            ids = tokens[..., level]
            if torch.any(ids < 0) or torch.any(ids >= self.sizes[level]):
                raise ValueError('Token outside level dictionary')
            value = F.embedding(ids, getattr(self, f'centers_{level}'))
            result = value if result is None else result + value
        return result if self.mode == 'residual' else result / levels

    @torch.no_grad()
    def fine_tokens(self, tokens):
        return self.nearest(self.prefix_latent(tokens), self.fine_codes)


def load_residual_bridge(path, device='cpu'):
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('version') != 1:
        raise ValueError('Unsupported residual bridge bundle version')
    bridge = ResidualTokenBridge(bundle['centers'], bundle['fine_codes'], bundle['mode'])
    if list(bridge.sizes) != bundle['sizes']:
        raise ValueError('Dictionary metadata mismatch')
    return bridge.to(device).eval(), bundle['metadata']
