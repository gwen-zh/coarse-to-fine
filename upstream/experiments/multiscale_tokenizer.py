"""Two-scale reconstruction probe. Geometry is encoder-only side information.

The local and global arms have identical parameters and 5 extra bits/residue.
Their only architecture difference is the geometry message-passing mask.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def pair_geometry(bb):
    """SE(3)-invariant distance, local directions, relative frames, sequence offset."""
    ca = bb[:, 1]
    ex = F.normalize(bb[:, 2] - ca, dim=-1)
    ey = bb[:, 0] - ca
    ey = F.normalize(ey - (ey * ex).sum(-1, keepdim=True) * ex, dim=-1)
    frames = torch.stack((ex, ey, torch.cross(ex, ey, dim=-1)), dim=-1)
    displacement = ca[None] - ca[:, None]
    distance = displacement.norm(dim=-1)
    direction = torch.einsum('nia,nji->nja', frames, displacement)
    direction = direction / distance.clamp_min(1e-6)[..., None]
    rotation = torch.einsum('nia,mib->nmab', frames, frames).flatten(-2)
    index = torch.arange(len(bb), device=bb.device)
    offset = index[None] - index[:, None]
    features = torch.cat((torch.log1p(distance)[..., None] / 4., direction,
                          rotation, offset.clamp(-32, 32)[..., None] / 32.), -1)
    local = offset.abs() <= 4
    neighbors = distance.topk(min(16, len(bb)), largest=False).indices
    local.scatter_(1, neighbors, True)
    return features, local


class GeometryBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.pair_bias = nn.Sequential(nn.Linear(14, 32), nn.SiLU(), nn.Linear(32, heads))
        self.pair_value = nn.Linear(14, width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
                                nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, x, pair, allowed):
        n = len(x)
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, self.heads, -1).unbind(1)
        logits = torch.einsum('ihd,jhd->hij', q, k) / math.sqrt(self.width // self.heads)
        logits = logits + self.pair_bias(pair).permute(2, 0, 1)
        weights = logits.masked_fill(~allowed[None], -torch.inf).softmax(-1)
        messages = torch.einsum('hij,jhd->ihd', weights, v)
        geometric = self.pair_value(pair).reshape(n, n, self.heads, -1)
        messages = messages + torch.einsum('hij,ijhd->ihd', weights, geometric)
        x = x + self.out(messages.flatten(1))
        return x + self.ff(x)


class EMABottleneck(nn.Module):
    def __init__(self, size=32, dim=32, decay=.95):
        super().__init__()
        self.decay = decay
        self.register_buffer('codes', F.normalize(torch.randn(size, dim), dim=-1))
        self.register_buffer('counts', torch.ones(size))
        self.register_buffer('sums', self.codes.clone())
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def initialize(self, features):
        # Training-only spherical k-means initialization, deterministic seed upstream.
        z = F.normalize(features, dim=-1)
        centers = z[torch.linspace(0, len(z)-1, len(self.codes), device=z.device).long()].clone()
        for _ in range(10):
            ids = (z @ centers.T).argmax(-1)
            sums = torch.zeros_like(centers).index_add_(0, ids, z)
            count = torch.bincount(ids, minlength=len(centers))
            centers = torch.where((count > 0)[:, None], F.normalize(sums, dim=-1), centers)
        self.codes.copy_(centers); self.counts.fill_(1); self.sums.copy_(centers)
        self.initialized.fill_(True)

    def forward(self, z):
        z = F.normalize(z, dim=-1)
        codes = self.codes.detach().clone()
        ids = (z @ codes.T).argmax(-1)
        quantized = codes[ids]
        commitment = .25 * (z - quantized).square().sum(-1).mean()
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(ids, minlength=len(codes)).to(z.dtype)
                sums = torch.zeros_like(codes).index_add_(0, ids, z)
                self.counts.mul_(self.decay).add_(counts, alpha=1-self.decay)
                self.sums.mul_(self.decay).add_(sums, alpha=1-self.decay)
                self.codes.copy_(F.normalize(self.sums, dim=-1))
                dead = self.counts < .1
                if dead.any():
                    replacement = z[torch.arange(int(dead.sum()), device=z.device) % len(z)]
                    self.codes[dead] = replacement
                    self.sums[dead] = replacement
                    self.counts[dead] = 1
        return z + (quantized - z).detach(), ids, commitment


class TwoScaleTokenizer(nn.Module):
    def __init__(self, mode, width=128, decoder_dim=1280):
        super().__init__()
        if mode not in ('local', 'global'):
            raise ValueError(mode)
        self.mode = mode
        self.input = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, width))
        self.blocks = nn.ModuleList([GeometryBlock(width) for _ in range(2)])
        self.project = nn.Linear(width, 32)
        self.quantizer = EMABottleneck()
        self.fusion = nn.Sequential(nn.Linear(160, 128), nn.SiLU(), nn.Linear(128, decoder_dim))
        nn.init.zeros_(self.fusion[-1].weight); nn.init.zeros_(self.fusion[-1].bias)

    def encode(self, fine_codes, pair, local_mask):
        x = self.input(fine_codes)
        allowed = local_mask if self.mode == 'local' else torch.ones_like(local_mask)
        for block in self.blocks:
            x = block(x, pair, allowed)
        return self.project(x)

    def fuse(self, fine_codes, fine_embedding, coarse, embedding_scale):
        # Exact original reconstruction at initialization; bounded residual amplitude.
        correction = .1 * embedding_scale * torch.tanh(self.fusion(torch.cat((F.layer_norm(fine_codes, (128,)), coarse * math.sqrt(32)), -1)))
        return fine_embedding + correction

    def forward(self, fine_codes, fine_embedding, pair, mask, scale):
        quantized, ids, commitment = self.quantizer(self.encode(fine_codes, pair, mask))
        return self.fuse(fine_codes, fine_embedding, quantized, scale), ids, commitment


def reconstruction_loss(prediction, reference):
    # Rotation is the optimal proper Kabsch rotation; detach its SVD for stability.
    pc, rc = prediction[:, 1].mean(0), reference[:, 1].mean(0)
    with torch.no_grad():
        u, _, vh = torch.linalg.svd((prediction[:, 1]-pc).T @ (reference[:, 1]-rc))
        sign = torch.ones(3, device=prediction.device)
        sign[-1] = torch.linalg.det(u @ vh)
        rotation = (u * sign) @ vh
    aligned = (prediction-pc) @ rotation + rc
    coordinate = (aligned-reference).square().sum(-1).mean()
    pred_cn = (prediction[:-1, 2]-prediction[1:, 0]).norm(dim=-1)
    true_cn = (reference[:-1, 2]-reference[1:, 0]).norm(dim=-1)
    local = (pred_cn-true_cn).square().mean()
    return coordinate + local, {'coordinate': coordinate.detach(), 'cn': local.detach()}
