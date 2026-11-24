"""Matched two-protein overfit controls for sequence and time-distribution use."""
import hashlib
import math

import torch

from par_coarse_backbone import ar_condition, flow_batch
from par_coarse_transfer import conditioning_features, draw


ARMS = {
    'position_beta': ('position_only', 'beta'),
    'sequence_beta': ('sequence', 'beta'),
    'position_uniform': ('position_only', 'uniform'),
    'sequence_uniform': ('sequence', 'uniform'),
}


def sample_keys(seed, count):
    # Deliberately omit protein ID: the length-only negative control must
    # return the SAME finite samples for both same-length proteins.
    return [int(hashlib.sha256(f'par-two-fit-sample-v1:{seed}:{i}'.encode()).hexdigest()[:8], 16)
            for i in range(count)]


def matched_draw(target, seed, step, slot, rule):
    if rule not in ['beta', 'uniform']:
        raise ValueError('Declared time-distribution rule required')
    state, velocity, old_t, self_condition, digest = draw(target, seed, 1200000+step, slot)
    noise = state-old_t*velocity
    key = int(hashlib.sha256(f'par-two-fit-time-v1:{seed}:{step}:{slot}'.encode()).hexdigest()[:8], 16)
    coins = torch.rand(2, generator=torch.Generator().manual_seed(key))
    uniform, mixture = coins[0], coins[1]
    t = uniform if rule == 'uniform' or mixture < .02 else uniform.pow(1/1.9)
    return noise+t*velocity, velocity, t, self_condition, dict(
        noise_rotation_sha256=digest, uniform_time_draw=float(uniform), mixture_coin=float(mixture))


def update(model, optimizer, cases, bank, arm, seed, step, protocol):
    if arm not in ARMS or len(cases) != 2 or len(cases[0]['sequence']) != len(cases[1]['sequence']):
        raise ValueError('Two same-length training cases and a declared arm required')
    mode, rule = ARMS[arm]
    device = next(model.parameters()).device
    model.eval(); optimizer.zero_grad(set_to_none=True)
    lr = protocol['min_lr']+.5*(protocol['lr']-protocol['min_lr'])*(1+math.cos(math.pi*step/(protocol['updates']-1)))
    for group in optimizer.param_groups: group['lr'] = lr
    events = []; total = 0.
    for slot in range(4):
        case = cases[(step+slot) % 2]
        if case['split'] != 'train': raise ValueError('Non-training example entered overfit control')
        features = conditioning_features(bank(case['sequence']).to(device), mode)
        state, truth, t, self_condition, event = matched_draw(case['coarse'], seed, step, slot, rule)
        state = state[None].to(device); truth = truth[None].to(device); t = t.to(device)
        batch, z = ar_condition(model, features, len(case['sequence']))
        sc = None
        if self_condition:
            with torch.no_grad():
                v = model.atom_diffusion.nn(flow_batch(batch, z.detach(), state, t))['coors_pred']
                sc = (state+(1-t)*v).detach()
        prediction = model.atom_diffusion.nn(flow_batch(batch, z, state, t, sc))['coors_pred']
        loss = (prediction-truth).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite two-protein loss')
        (loss/4).backward(); total += float(loss.detach())/4
        events.append(dict(case_id=case['id'], protein=case['protein'], self_condition=self_condition,
                           time=float(t), **event))
    params = [v for v in model.parameters() if v.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, protocol['gradient_clip'])
    if not torch.isfinite(norm): raise RuntimeError('Invalid adapter gradient')
    optimizer.step()
    return dict(update=step+1, loss=total, lr=lr, gradient_norm=float(norm), batch=events)


def identity(row):
    return dict(lr=row['lr'], batch=[{k:v for k,v in event.items() if k != 'time'} for event in row['batch']])
