"""Bounded adjacent-block decoder context and separately normalized edge losses."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_latent_contract import compose_levels, remove_block_translation
from two_level_autoencoder import TwoLevelAutoencoder, pack, training_loss


class NeighborFineAutoencoder(TwoLevelAutoencoder):
    """Keep local encoders; decoder fine support expands to at most three blocks.

    Common parameters exactly copy a base model. Added columns start at zero;
    constructing this adapter does not consume the training augmentation RNG.
    Adds 5120 parameters (2.75%); this is not a capacity-matched information test.
    """
    def __init__(self, base):
        nn.Module.__init__(self)
        for name in ['coarse_dim', 'fine_dim', 'block_size']:
            setattr(self, name, getattr(base, name))
        for name in ['coarse_encoder', 'coarse_decoder', 'fine_encoder', 'fine_decoder']:
            setattr(self, name, copy.deepcopy(getattr(base, name)))
        first = self.fine_decoder[0]
        extra = 2 * self.block_size * (self.fine_dim + 1)
        with torch.random.fork_rng(devices=[]):
            expanded = nn.Linear(first.in_features + extra, first.out_features)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :first.in_features].copy_(first.weight)
            expanded.bias.copy_(first.bias)
        self.fine_decoder[0] = expanded
        self.original_fine_inputs = first.in_features

    def decode_with_anchors(self, centers, z_f, length):
        if z_f.shape != (length, self.fine_dim):
            raise ValueError('Fine latent shape')
        packed, mask = pack(z_f, self.block_size)
        if centers.shape != (len(mask), 3):
            raise ValueError('Anchor shape')
        previous = F.pad(centers[1:] - centers[:-1], (0, 0, 1, 0)) / 10.
        following = F.pad(centers[1:] - centers[:-1], (0, 0, 0, 1)) / 10.
        own = torch.cat([packed.flatten(1), previous, following, mask.to(z_f.dtype)], -1)
        blocks = torch.cat([packed.flatten(1), mask.to(z_f.dtype)], -1)
        left = F.pad(blocks[:-1], (0, 0, 1, 0))
        right = F.pad(blocks[1:], (0, 0, 0, 1))
        raw = self.fine_decoder(torch.cat([own, left, right], -1))
        raw = raw.reshape(-1, 3, 3)[:length] * 4.
        return dict(backbone=compose_levels(centers, raw, self.block_size), coarse=centers,
                    fine_residual=remove_block_translation(raw, self.block_size))


def boundary_terms(pred, true, block_size=4):
    ids = torch.arange(len(pred) - 1, device=pred.device)
    edge = ids // block_size != (ids + 1) // block_size
    if not edge.any():
        zero = pred.sum() * 0.
        return zero, zero
    def values(x):
        distance = (x[:-1, 2] - x[1:, 0]).norm(dim=-1)[edge]
        def cosine(a, b):
            return (F.normalize(a, dim=-1, eps=1e-6) * F.normalize(b, dim=-1, eps=1e-6)).sum(-1)
        first = cosine(x[:-1, 1] - x[:-1, 2], x[1:, 0] - x[:-1, 2])[edge]
        second = cosine(x[:-1, 2] - x[1:, 0], x[1:, 1] - x[1:, 0])[edge]
        return distance, torch.cat([first, second])
    a, b = values(pred), values(true)
    return (a[0] - b[0]).square().mean(), (a[1] - b[1]).square().mean()


def boundary_training_loss(output, reference, weights, enabled):
    loss, terms = training_loss(output, reference, weights)
    bond, angle = boundary_terms(output['backbone'], reference)
    terms = dict(**terms, boundary_bond=bond, boundary_angle=angle)
    if enabled:
        # Same coefficients as the global geometry terms; no weight sweep.
        loss = loss + weights['bond'] * bond + weights['angle'] * angle
    return loss, terms
