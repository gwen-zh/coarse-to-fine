"""Zero-initialized geometry corrections on a frozen coarse velocity field.

Both arms use identical MLP weights/counts and relative-vector aggregation.
Only three state features differ: rotation-invariant radial features versus
Cartesian relative coordinates. The radial CORRECTION is rotation equivariant;
the complete frozen-parent-plus-correction model is not claimed equivariant.
"""
import hashlib

import torch
from torch import nn

from coarse_few_target_flow import learning_rate
from coarse_full_cohort_flow import training_seed
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from joint_sequence_predictor import encode_aa
from multisequence_flow_data import check_case


class CoarseRelativeGeometry(nn.Module):
    def __init__(self, base, kind="radial", hidden=64):
        super().__init__()
        if kind not in {"cartesian", "radial"} or base.coarse_dim != 3 or hidden < 1:
            raise ValueError("Physical centers and a declared correction kind required")
        self.base = base.eval().requires_grad_(False)
        self.kind = kind; self.coarse_dim = 3; self.block_size = base.block_size
        width = base.config["width"]
        self.adapter = nn.Sequential(nn.Linear(3 * width + 4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.register_buffer("physical_scale", base.coarse_std.square().mean().sqrt().detach().clone())
        self.config = dict(base=base.config, kind=kind, hidden=hidden, gate="t_squared")

    @property
    def feature_bank(self): return self.base.feature_bank
    @feature_bank.setter
    def feature_bank(self, value): self.base.feature_bank = value

    def train(self, mode=True):
        super().train(mode); self.base.eval(); return self

    def normalize_coarse(self, values): return self.base.normalize_coarse(values)
    def denormalize_coarse(self, values): return self.base.denormalize_coarse(values)

    def correction(self, aa, state, time):
        self.base._coarse_state(state, len(aa))
        _, block = self.base.aa_conditions(aa)
        t = torch.as_tensor(time, dtype=state.dtype, device=state.device)
        clock = self.base.time_state(t, block)
        centers = self.base.denormalize_coarse(state)
        m = len(centers)
        if m < 2: raise ValueError("At least two blocks required")
        relative = (centers[None] - centers[:, None]) / self.physical_scale
        if self.kind == "radial":
            squared = relative.square().sum(-1)
            spatial = torch.stack([(squared + 1e-8).sqrt(), torch.log1p(squared), squared / (1 + squared)], -1)
        else:
            spatial = relative
        position = torch.arange(m, device=state.device, dtype=state.dtype)
        separation = (position[:, None] - position[None]).abs()[..., None] / max(1, m - 1)
        pair = torch.cat([block[:, None].expand(m, m, -1), block[None].expand(m, m, -1),
            clock.expand(m, m, -1), separation, spatial], -1)
        coefficients = self.adapter(pair).squeeze(-1).tanh()
        coefficients = coefficients * (1 - torch.eye(m, device=state.device, dtype=state.dtype))
        physical = self.physical_scale * (coefficients[..., None] * relative).sum(1) / (m - 1)
        counts = torch.minimum(torch.full((m,), 4., device=state.device, dtype=state.dtype),
            len(aa) - 4 * torch.arange(m, device=state.device, dtype=state.dtype))
        physical = physical - (physical * counts[:, None]).sum(0) / len(aa)
        return t.square() * physical / self.base.coarse_std

    def velocity_parts(self, aa, state, time):
        # Frozen parameters still allow derivatives through state for a full
        # differentiable rollout. Do not detach the parent velocity here.
        original = self.base.coarse_velocity(aa, state, time)
        delta = self.correction(aa, state, time)
        return original + delta, delta

    def coarse_velocity(self, aa, state, time):
        return self.velocity_parts(aa, state, time)[0]

    @torch.no_grad()
    def sample_coarse(self, aa, seed, steps=16):
        return HierarchicalEnsembleFlow.sample_coarse(self, aa, seed, steps)


class GeometryCorrectedEnsemble(nn.Module):
    def __init__(self, coarse, fine):
        super().__init__(); self.coarse_model = coarse
        self.fine_model = fine.eval().requires_grad_(False); self.fine_dim = fine.fine_dim
        if {id(v) for v in coarse.parameters()} & {id(v) for v in fine.parameters()}:
            raise ValueError("Fine and coarse must be independent parameter objects")
    def train(self, mode=True):
        super().train(mode); self.coarse_model.base.eval(); self.fine_model.eval(); return self
    def trainable_parameters(self): return list(self.coarse_model.adapter.parameters())
    def sample_coarse(self, *args, **kwargs): return self.coarse_model.sample_coarse(*args, **kwargs)
    def sample_fine(self, *args, **kwargs): return self.fine_model.sample_fine(*args, **kwargs)


def update(model, optimizer, batch, seed, step, p):
    if len(batch) != 4: raise ValueError("Exactly three PDB plus one native MD frame required")
    for case in batch: check_case(case, "train")
    model.train(); optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, p["updates"], p["lr"], p["min_lr"])
    for group in optimizer.param_groups: group["lr"] = lr
    losses = []; fm_values = []; penalties = []; ids = []; draws = []
    for slot, case in enumerate(batch):
        branch = model.coarse_model; target = branch.normalize_coarse(case["coarse"].detach())
        rng = torch.Generator().manual_seed(training_seed(seed, step, slot))
        noise = torch.randn(target.shape, generator=rng, dtype=target.dtype)
        t = torch.rand((), generator=rng, dtype=target.dtype)
        velocity, delta = branch.velocity_parts(encode_aa(case["sequence"]), (1-t)*noise+t*target, t)
        fm = (velocity-(target-noise)).square().mean(); penalty = delta.square().mean()
        loss = fm + p["correction_penalty"] * penalty
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite coarse correction loss")
        (loss / 4).backward(); losses.append(float(loss.detach())); fm_values.append(float(fm.detach())); penalties.append(float(penalty.detach()))
        ids.append(dict(protein=case["protein"], case_id=case["id"], domain="PDB" if slot < 3 else "MD"))
        draws.append(hashlib.sha256(noise.numpy().tobytes()+t.numpy().tobytes()).hexdigest())
    norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), p["gradient_clip"])
    if not torch.isfinite(norm): raise RuntimeError("Nonfinite correction gradient")
    optimizer.step()
    return dict(update=step+1, loss=sum(losses)/4, FM_loss=sum(fm_values)/4,
        correction_squared_velocity=sum(penalties)/4, PDB_FM_loss=sum(fm_values[:3])/3,
        MD_FM_loss=fm_values[3], gradient_norm=float(norm), batch=ids, base_draws=draws, lr=lr)
