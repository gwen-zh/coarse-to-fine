"""Matched polar/axial residuals for the native coarse velocity objective.

Only the residual has the stated transformation laws. The pretrained PAR field
is not assumed to be exactly equivariant. No targets or reference structures
are inputs to this module, and no chirality/geometry loss is introduced.
"""
import hashlib

import torch
from torch import nn

from par_coarse_capacity import learning_rate
from par_two_protein_fit import update as native_update


ARMS = ['native', 'polar', 'axial']
SEEDS = [2029, 2030]
UPDATES = 1024
DRAW_OFFSET = 6144
RESIDUAL_INIT_SEED = 20261124
WIDTH = 64
LENGTH_FLOOR_NM = .1


def evaluation_keys():
    keys = [int(hashlib.sha256(f'par-chiral-velocity-v1:noise:{i}'.encode()).hexdigest()[:8], 16)
            for i in range(64)]
    if len(set(keys)) != 64:
        raise RuntimeError('Evaluation keys collided')
    return keys


def chain_features(x):
    """Reflection-even scalars and polar / axial bases, without axis fallbacks.

    x is unpadded B x M x 3, in native nm. End edges are explicitly zero.
    Smooth norm regularization leaves degenerate-chain gradients finite.
    """
    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[1] < 3:
        raise ValueError('At least three unpadded coarse points required')
    centered = x - x.mean(1, keepdim=True)
    edges = x[:, 1:] - x[:, :-1]
    zero = torch.zeros_like(x[:, :1])
    left = torch.cat([zero, edges], 1)
    right = torch.cat([edges, zero], 1)
    vectors = torch.stack([left, right, centered], 2)
    norms = (vectors.square().sum(-1) + LENGTH_FLOOR_NM**2).sqrt()
    unit = vectors / norms[..., None]
    a, b, c = unit.unbind(2)
    invariants = torch.cat([norms.log1p(), torch.stack([(a*b).sum(-1), (a*c).sum(-1),
                                                      (b*c).sum(-1)], -1)], -1)
    axial = torch.stack([torch.cross(a, b, dim=-1), torch.cross(a, c, dim=-1),
                         torch.cross(b, c, dim=-1)], 2)
    return invariants, unit, axial


class VelocityResidual(nn.Module):
    """Equal-parameter residuals with polar versus axial vector bases.

    Scalar coefficients see AA-derived z, time and reflection-even geometry.
    Under row-vector x -> xR, polar bases transform with R, axial with det(R)R.
    AA-only z is held invariant in this coarse-only interface. This does not
    assert equivariance for a future structure-conditioned AR interface.
    """
    def __init__(self, kind, condition_dim=512, width=WIDTH):
        super().__init__()
        if kind not in ['polar', 'axial']:
            raise ValueError('Declared vector basis required')
        self.kind = kind
        self.condition_dim = condition_dim
        self.norm = nn.LayerNorm(condition_dim)
        self.hidden = nn.Linear(condition_dim + 15, width)
        self.output = nn.Linear(width, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostic = {}

    def forward(self, state, condition, time, self_condition=None, mask=None):
        if condition.shape != (*state.shape[:2], self.condition_dim):
            raise ValueError('AA-derived conditioning must match the coarse trace')
        if mask is not None and (mask.shape != state.shape[:2] or not bool(mask.all())):
            raise ValueError('This pilot accepts only fully valid unpadded traces')
        if not torch.isfinite(state).all() or not torch.isfinite(condition).all():
            raise ValueError('Finite coarse state and AA conditions required')
        present = self_condition is not None
        sc = self_condition if present else torch.zeros_like(state)
        if sc.shape != state.shape or not torch.isfinite(sc).all():
            raise ValueError('Invalid self-conditioning coordinates')
        inv_x, polar_x, axial_x = chain_features(state)
        inv_s, polar_s, axial_s = chain_features(sc)
        time = torch.as_tensor(time, dtype=state.dtype, device=state.device).reshape(-1)
        if time.numel() == 1:
            time = time.expand(len(state))
        if time.shape != (len(state),):
            raise ValueError('One time per sample required')
        times = torch.stack([time, 1-time], -1)[:, None].expand(-1, state.shape[1], -1)
        flag = state.new_full((*state.shape[:2], 1), float(present))
        scalars = torch.cat([self.norm(condition), inv_x, inv_s, times, flag], -1)
        weights = self.output(torch.nn.functional.silu(self.hidden(scalars)))
        basis = torch.cat([polar_x, polar_s] if self.kind == 'polar' else [axial_x, axial_s], 2)
        delta = (weights[..., None] * basis).sum(2)
        delta = delta - delta.mean(1, keepdim=True)
        self.last_diagnostic = dict(basis_RMS=float(basis.detach().square().mean().sqrt()),
                                    residual_RMS_nm=float(delta.detach().square().mean().sqrt()))
        return delta


class ResidualFlow(nn.Module):
    def __init__(self, base, residual):
        super().__init__()
        self.base = base
        self.residual = residual

    def forward(self, batch):
        result = dict(self.base(batch))
        delta = self.residual(batch['x_t'], batch['z_cond'], batch['t'],
                              batch.get('x_sc'), batch.get('mask'))
        if delta.shape != result['coors_pred'].shape:
            raise ValueError('Residual must match the native velocity shape')
        result['coors_pred'] = result['coors_pred'] + delta
        return result


def install_residual(model, view, kind):
    if kind not in ['polar', 'axial'] or hasattr(view, 'velocity_residual'):
        raise ValueError('Fresh matched residual arm required')
    width = model.cfg_exp.model.nn.token_dim
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RESIDUAL_INIT_SEED)
        residual = VelocityResidual(kind, condition_dim=width)
    residual = residual.to(next(model.parameters()).device)
    model.atom_diffusion.nn = ResidualFlow(model.atom_diffusion.nn, residual)
    view.add_module('velocity_residual', residual)
    return residual


def update(model, optimizer, cases, bank, seed, local_step):
    lr = learning_rate(local_step)
    row = native_update(model, optimizer, cases, bank, 'sequence_uniform', seed,
                        DRAW_OFFSET + local_step,
                        dict(lr=lr, min_lr=lr, updates=DRAW_OFFSET+UPDATES, gradient_clip=1.))
    row['draw_step'] = DRAW_OFFSET + local_step
    row['total_adapter_updates'] = 4096 + local_step + 1
    row['update'] = local_step + 1
    field = model.atom_diffusion.nn
    if isinstance(field, ResidualFlow):
        row['residual_diagnostic'] = dict(field.residual.last_diagnostic)
    return row


def paired_identity(row):
    return dict(lr=row['lr'], batch=row['batch'], draw_step=row['draw_step'])
