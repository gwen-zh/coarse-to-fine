"""Sequence-conditioned spatial hierarchy with optional continuous fine decoding.

This is a small controlled implementation, inspired by PAR's scale-wise
conditioning, not a reproduction or adaptation of PAR's released weights.
The inherited Transformer parameterizes two flow velocity fields:
p(c, f | AA) = p(c | AA) p(f | c, AA).
"""
import hashlib

import torch
from torch import nn

from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from hierarchical_ensemble_flow import component_seed, snap_fine
from hierarchical_latent_contract import pool_ca
from mixed_hierarchical_generator import HybridTokenizer


class MultiscaleStructureCodec(nn.Module):
    """Frozen structure maps; coarse is either learned16 or physical centers3."""

    def __init__(self, coarse_tokenizer, fine_tokenizer, coarse_kind):
        super().__init__()
        if coarse_kind not in {"latent16", "centers3"}:
            raise ValueError("Unknown coarse representation")
        if fine_tokenizer.mode != "local" or fine_tokenizer.block_size != 4:
            raise ValueError("Expected the frozen local-frame stride-four model")
        self.coarse_tokenizer = coarse_tokenizer
        self.fine_tokenizer = fine_tokenizer
        self.coarse_kind = coarse_kind
        self.coarse_dim = 16 if coarse_kind == "latent16" else 3
        self.eval().requires_grad_(False)

    @classmethod
    def from_checkpoints(cls, warmup, final, fine_checkpoint, coarse_kind):
        coarse = HybridTokenizer.from_checkpoints(warmup, final)
        saved = torch.load(fine_checkpoint, map_location="cpu", weights_only=False)
        fine = StableBlockFrameFineTokenizer(**saved["model_config"])
        fine.load_state_dict(saved["model"])
        return cls(coarse, fine, coarse_kind)

    def project_coarse(self, coarse, length):
        m = (length + 3) // 4
        if coarse.shape != (m, self.coarse_dim) or not torch.isfinite(coarse).all():
            raise ValueError("Invalid coarse shape or nonfinite values")
        if self.coarse_kind == "latent16":
            return coarse
        counts = coarse.new_tensor([min(4, length - 4 * i) for i in range(m)])
        return coarse - (coarse * counts[:, None]).sum(0) / length

    def centers(self, coarse, length):
        coarse = self.project_coarse(coarse, length)
        if self.coarse_kind == "centers3":
            return coarse
        return self.coarse_tokenizer.coarse_centers(coarse, length)

    def context(self, coarse, length):
        return self.fine_tokenizer.context(self.centers(coarse, length), length)

    @torch.no_grad()
    def encode_coarse(self, target):
        centers = pool_ca(target, 4)
        if self.coarse_kind == "centers3":
            return self.project_coarse(centers, len(target))
        return torch.tanh(self.coarse_tokenizer.model.coarse_encoder(centers / 10.))

    @torch.no_grad()
    def fine_target(self, target, coarse):
        # Re-express the SAME physical target in this condition's local frames.
        # Reusing an old fine label after changing the frames is incorrect.
        return self.fine_tokenizer.encode(target, self.context(coarse, len(target)))

    @torch.no_grad()
    def encode(self, target):
        coarse = self.encode_coarse(target)
        return coarse, self.fine_target(target, coarse)

    @torch.no_grad()
    def decode(self, coarse, fine, quantized=False):
        if fine.ndim != 2 or fine.shape[1] != self.fine_tokenizer.fine_dim:
            raise ValueError("Expected continuous per-residue fine vectors")
        if not torch.isfinite(fine).all():
            raise ValueError("Nonfinite fine values")
        if quantized:
            ids, _ = snap_fine(fine, self.fine_tokenizer.book.codes)
            fine = self.fine_tokenizer.book.codes[ids]
        return self.fine_tokenizer.decode_vectors(
            fine, self.context(coarse, len(fine)), len(fine)
        )


def paired_flow_loss(model, codec, aa, target, coarse, fine, rng,
                     scheduled_probability=0.):
    """PAR-style same-example endpoint context, with local labels re-expressed.

    Endpoint prediction is based on a corrupted version of this training
    structure. It is not an unconditional rollout and cannot prove exposure
    bias has been eliminated. No test/reference structures enter sampling.
    """
    if not 0. <= scheduled_probability <= 1.:
        raise ValueError("Scheduled-sampling probability outside [0, 1]")
    c = model.normalize_coarse(coarse.detach())
    noise_c = torch.randn(c.shape, generator=rng, device=c.device, dtype=c.dtype)
    noise_f = torch.randn(fine.shape, generator=rng, device=c.device, dtype=c.dtype)
    tc = torch.rand((), generator=rng, device=c.device)
    tf = torch.rand((), generator=rng, device=c.device)
    draw = torch.rand((), generator=rng, device=c.device)
    state_c = (1 - tc) * noise_c + tc * c
    velocity_c = model.coarse_velocity(aa, state_c, tc)
    scheduled = bool(draw < scheduled_probability)
    if scheduled:
        with torch.no_grad():
            endpoint = model.denormalize_coarse(state_c + (1 - tc) * velocity_c.detach())
            condition = codec.project_coarse(endpoint, len(aa))
            fine = codec.fine_target(target, condition)
    else:
        condition = coarse.detach()
    f = model.normalize_fine(fine.detach())
    state_f = (1 - tf) * noise_f + tf * f
    velocity_f = model.fine_velocity(aa, state_f, tf, model.normalize_coarse(condition))
    coarse_loss = (velocity_c - (c - noise_c)).square().mean()
    fine_loss = (velocity_f - (f - noise_f)).square().mean()
    digest = hashlib.sha256(
        noise_c.cpu().numpy().tobytes() + noise_f.cpu().numpy().tobytes()
        + torch.stack([tc, tf, draw]).cpu().numpy().tobytes()
    ).hexdigest()
    return coarse_loss + fine_loss, dict(
        coarse_flow=float(coarse_loss.detach()), fine_flow=float(fine_loss.detach()),
        scheduled=scheduled, draw_sha256=digest
    )


@torch.no_grad()
def sample_levels(model, codec, aa, seed, coarse_steps=16, fine_steps=16):
    """Only AA and random seeds are needed; no native structure argument."""
    c = model.sample_coarse(aa, component_seed(seed, "coarse"), coarse_steps)
    c = codec.project_coarse(c, len(aa))
    f = model.sample_fine(aa, c, component_seed(seed, "fine"), fine_steps)
    return c, f
