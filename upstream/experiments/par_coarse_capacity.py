"""Proper/improper shape diagnostics and a bounded native-flow LoRA control.

Reflection-permitting scores are descriptive diagnostics only. They must never
replace proper-rotation scores, change coordinates, or select generated samples.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn

from coarse_generation_diagnostic import block_counts, check_centers, weighted_rmsd_matrix
from coarse_lora import LoRALinear
from par_coarse_backbone import ar_condition, flow_batch
from par_two_protein_fit import sample_keys, update as original_update


ARMS = ['adapter_only', 'adapter_flow_lora']
PARENT_UPDATES = 4096
UPDATES = 1024


def evaluation_keys(seed):
    """32 historical noise draws plus 32 newly fixed draws; retain all 64."""
    fresh = [int(hashlib.sha256(f'par-capacity-new-noise-v1:{seed}:{i}'.encode()).hexdigest()[:8], 16)
             for i in range(32)]
    keys = sample_keys(seed, 32) + fresh
    if len(set(keys)) != 64:
        raise RuntimeError('Evaluation noise keys collided')
    return keys


def geometry_diagnostic(values, reference, length):
    values = check_centers(values, length)
    reference = check_centers(np.asarray(reference)[None], length)[0]
    weight = block_counts(length) / length
    x = values - np.einsum('smj,m->sj', values, weight)[:, None]
    y = reference - (reference * weight[:, None]).sum(0)
    u, singular, vh = np.linalg.svd(np.einsum('smi,m,mj->sij', x, weight, y))
    determinant = np.linalg.det(u @ vh)
    norm = np.einsum('smi,m,smi->s', x, weight, x) + (weight[:, None] * y * y).sum()
    improper = np.sqrt(np.maximum(norm - 2 * singular.sum(1), 0.))
    proper = weighted_rmsd_matrix(values, reference[None], length)[:, 0]
    return dict(proper_RMSD_A=proper, reflection_permitted_RMSD_A=improper,
                orthogonal_fit_determinant=determinant,
                mirror_like=(determinant < 0) & (proper > 2.) & (improper <= 2.),
                within_2A=proper <= 2.)


def summary(values, reference, length):
    result = geometry_diagnostic(values, reference, length)
    return dict(samples=len(values), mean_proper_RMSD_A=float(result['proper_RMSD_A'].mean()),
                median_proper_RMSD_A=float(np.median(result['proper_RMSD_A'])),
                p90_proper_RMSD_A=float(np.quantile(result['proper_RMSD_A'], .9)),
                mean_reflection_permitted_RMSD_A=float(result['reflection_permitted_RMSD_A'].mean()),
                negative_orthogonal_fit_count=int((result['orthogonal_fit_determinant'] < 0).sum()),
                mirror_like_count=int(result['mirror_like'].sum()),
                samples_within_2A=int(result['within_2A'].sum()))


def learning_rate(local_step):
    if not 0 <= local_step < UPDATES:
        raise ValueError('Outside the fixed capacity-control budget')
    return 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * local_step / (UPDATES - 1)))


def install_flow_lora(model, view, rank=8, last_blocks=2):
    """Add zero-output LoRA only to QKV and output in the last native flow blocks.

    The view stores only the low-rank parameters, not copies of frozen weights.
    Its existing adapter keys and optimizer group retain their original order.
    """
    blocks = model.atom_diffusion.nn.transformer_layers
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError('Invalid native flow adaptation depth')
    holders = nn.ModuleDict()
    names = []
    for index in range(len(blocks) - last_blocks, len(blocks)):
        attention = blocks[index].mhba.mha
        for name in ['to_qkv', 'to_out_node']:
            base = getattr(attention, name)
            if not isinstance(base, nn.Linear) or any(p.requires_grad for p in base.parameters()):
                raise ValueError('Expected an unwrapped frozen native projection')
            layer = LoRALinear(base, rank=rank)
            setattr(attention, name, layer)
            holder = nn.Module()
            holder.register_parameter('a', layer.a)
            holder.register_parameter('b', layer.b)
            holders[f'block{index}_{name}'] = holder
            names.append(f'atom_diffusion.nn.transformer_layers.{index}.mhba.mha.{name}')
    if hasattr(view, 'flow_lora'):
        raise ValueError('LoRA already installed')
    view.add_module('flow_lora', holders)
    return names


def update(model, optimizer, cases, bank, seed, local_step):
    lr = learning_rate(local_step)
    row = original_update(model, optimizer, cases, bank, 'sequence_uniform', seed,
                          PARENT_UPDATES + local_step,
                          dict(lr=lr, min_lr=lr, updates=PARENT_UPDATES + UPDATES, gradient_clip=1.))
    row['global_update'] = row['update']
    row['update'] = local_step + 1
    return row


@torch.no_grad()
def trajectory_sample(model, features, length, keys, steps=16):
    """Exact native coarse Euler loop with a copy of every state and clean guess.

    This routine receives no reference coordinates. Scoring happens afterwards.
    Arrays are saved in native nanometers; final outputs are centered Angstroms.
    """
    batch, condition = ar_condition(model, features, length, len(keys))
    device = condition.device
    noise = torch.stack([torch.randn(((length + 3) // 4, 3),
                                    generator=torch.Generator().manual_seed(int(key))) for key in keys])
    state = noise.to(device)
    state = state - state.mean(1, keepdim=True)
    states = [state.cpu().numpy().copy()]
    clean = []
    self_condition = None
    for i in range(steps):
        time = i / steps
        prediction = model.atom_diffusion.nn(flow_batch(batch, condition, state, time, self_condition))['coors_pred']
        self_condition = (state + (1 - time) * prediction).detach()
        clean.append(self_condition.cpu().numpy().copy())
        state = state + prediction / steps
        state = state - state.mean(1, keepdim=True)
        states.append(state.cpu().numpy().copy())
    result = state * 10.
    counts = result.new_tensor(block_counts(length))
    result = result - (counts[None, :, None] * result).sum(1, keepdim=True) / length
    return dict(coarse=result.cpu().numpy(), states_nm=np.stack(states), clean_predictions_nm=np.stack(clean),
                seeds=np.asarray(keys, dtype=np.uint32))
