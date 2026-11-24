"""Reflection-sensitive supervision of clean coarse predictions, without STE.

Both objective arms have identical adapter+flow-LoRA parameters. Only this
training loss differs; generation continues to use the unchanged native field.
"""
import numpy as np
import torch

from par_coarse_backbone import ar_condition, flow_batch
from par_coarse_capacity import learning_rate
from par_two_protein_fit import matched_draw


ARMS = ['velocity_only', 'velocity_signed_volume']
UPDATES = 1024
DRAW_OFFSET = 5120
STRIDES = (1, 2, 4)
EDGE_FLOOR_NM = .1


def signed_features(points):
    """Dimensionless oriented volumes, invariant under SO(3), odd under reflection.

    Use consecutive edges at three fixed spacings. The 0.1-nm edge floor keeps
    nearly collapsed clean predictions numerically stable; it is applied to
    predictions and targets equally and is not an inference constraint.
    """
    if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 4:
        raise ValueError('Batch x complete coarse chain x xyz required')
    features = []
    for stride in STRIDES:
        count = points.shape[1] - 3 * stride
        if count <= 0:
            continue
        a = points[:, stride:stride+count] - points[:, :count]
        b = points[:, 2*stride:2*stride+count] - points[:, stride:stride+count]
        c = points[:, 3*stride:3*stride+count] - points[:, 2*stride:2*stride+count]
        numerator = (torch.cross(a, b, dim=-1) * c).sum(-1)
        denominator = (a.norm(dim=-1).clamp_min(EDGE_FLOOR_NM)
                       * b.norm(dim=-1).clamp_min(EDGE_FLOOR_NM)
                       * c.norm(dim=-1).clamp_min(EDGE_FLOOR_NM))
        features.append(numerator / denominator)
    return torch.cat(features, dim=-1)


def slot_losses(model, cases, bank, seed, draw_step, slot):
    case = cases[(draw_step + slot) % 2]
    if case['split'] != 'train':
        raise ValueError('Only the fixed training pair is authorized')
    device = next(model.parameters()).device
    state, truth, t, sc_coin, event = matched_draw(case['coarse'], seed, draw_step, slot, 'uniform')
    state, truth, t = state[None].to(device), truth[None].to(device), t.to(device)
    batch, condition = ar_condition(model, bank(case['sequence']).to(device), 90)
    sc = None
    if sc_coin:
        with torch.no_grad():
            initial = model.atom_diffusion.nn(flow_batch(batch, condition.detach(), state, t))['coors_pred']
            sc = (state + (1-t) * initial).detach()
    prediction = model.atom_diffusion.nn(flow_batch(batch, condition, state, t, sc))['coors_pred']
    velocity = (prediction - truth).square().mean()
    clean = state + (1-t) * prediction
    target = (state + (1-t) * truth).detach()
    volume = (signed_features(clean) - signed_features(target)).square().mean()
    if not torch.isfinite(velocity) or not torch.isfinite(volume):
        raise RuntimeError('Nonfinite objective component')
    return velocity, volume, dict(case_id=case['id'], protein=case['protein'],
                                  self_condition=sc_coin, time=float(t), **event)


def choose_weight(records):
    ratios = [r['velocity_gradient_norm'] / r['volume_gradient_norm'] for r in records]
    if not ratios or not np.isfinite(ratios).all() or min(ratios) <= 0:
        raise ValueError('Finite positive calibration gradient norms required')
    return float(np.clip(np.median(ratios), .01, 100.))


def calibrate(model, cases, bank, seed):
    """Eight fixed TRAIN batches, no parameter updates or quality selection."""
    parameters = [v for v in model.parameters() if v.requires_grad]
    records = []
    model.eval()
    for index in range(8):
        velocity, volume = 0., 0.
        events = []
        for slot in range(4):
            fm, signed, event = slot_losses(model, cases, bank, seed, 2100000+index, slot)
            velocity, volume = velocity + fm/4, volume + signed/4
            events.append(event)
        grad_v = torch.autograd.grad(velocity, parameters, retain_graph=True, allow_unused=True)
        grad_s = torch.autograd.grad(volume, parameters, allow_unused=True)
        nv = sum(float(v.detach().double().square().sum()) for v in grad_v if v is not None) ** .5
        ns = sum(float(v.detach().double().square().sum()) for v in grad_s if v is not None) ** .5
        if min(nv, ns) <= 1e-12 or not np.isfinite([nv, ns]).all():
            raise RuntimeError('Calibration objective does not reach trainable parameters')
        dot = sum(float((a.detach().double() * b.detach().double()).sum())
                  for a, b in zip(grad_v, grad_s) if a is not None and b is not None)
        records.append(dict(calibration_draw_step=2100000+index, velocity_loss=float(velocity.detach()),
                            volume_loss=float(volume.detach()), velocity_gradient_norm=nv,
                            volume_gradient_norm=ns, gradient_cosine=dot/(nv*ns), batch=events))
    return dict(weight=choose_weight(records), formula='clip(median(||gradFM||/||gradVolume||),0.01,100)',
                batches=8, parameter_updates=0, records=records)


def update(model, optimizer, cases, bank, arm, seed, step, weight):
    if arm not in ARMS:
        raise ValueError('Declared objective arm required')
    lr = learning_rate(step)
    for group in optimizer.param_groups:
        group['lr'] = lr
    active_weight = weight if arm == 'velocity_signed_volume' else 0.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    losses, events = [], []
    for slot in range(4):
        velocity, volume, event = slot_losses(model, cases, bank, seed, DRAW_OFFSET+step, slot)
        combined = velocity + active_weight*volume if active_weight else velocity
        (combined/4).backward()
        losses.append([float(velocity.detach()), float(volume.detach()), float(combined.detach())])
        events.append(event)
    norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite objective gradient')
    optimizer.step()
    means = np.mean(losses, axis=0)
    return dict(update=step+1, total_adapter_updates=4096+step+1, draw_step=DRAW_OFFSET+step,
                loss=float(means[2]), velocity_loss=float(means[0]), volume_loss=float(means[1]),
                signed_volume_weight=active_weight, lr=lr, gradient_norm=float(norm), batch=events)
