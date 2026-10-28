"""LoRA adaptation of the released large model's last coarse-attention blocks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                          dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = 1.0

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def install_coarse_lora(model, last_blocks=8, rank=8):
    model.requires_grad_(False)
    blocks = model.net.transformer.blocks
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError('Invalid number of adapted blocks')
    names = []
    for index in range(len(blocks) - last_blocks, len(blocks)):
        attention = blocks[index].attn
        attention.layernorm_qkv[1] = LoRALinear(attention.layernorm_qkv[1], rank)
        attention.out_proj = LoRALinear(attention.out_proj, rank)
        names.extend([f'net.transformer.blocks.{index}.attn.layernorm_qkv.1',
                      f'net.transformer.blocks.{index}.attn.out_proj'])
    model.net.output_heads.coarse_structure_head.requires_grad_(True)
    return names


@torch.no_grad()
def initialize_new_coarse_semantics(model, new_centers, old_centers, initialization=None):
    """Soft old-code interpolation distinguishes new codes even with collisions."""
    if initialization is None:
        d = torch.cdist(new_centers.float(), old_centers.float()).square()
        temperature = d.min(-1).values.median().clamp_min(1e-4)
        probability = (-d / temperature).softmax(-1)
    else:
        temperature = float(initialization['temperature'])
        probability = torch.tensor(initialization['interpolation'], device=new_centers.device, dtype=torch.float32)
        if (probability.shape != (len(new_centers), len(old_centers)) or
                not torch.isfinite(probability).all() or torch.any(probability < 0) or
                not torch.allclose(probability.sum(-1), torch.ones(len(new_centers), device=new_centers.device))):
            raise ValueError('Invalid saved semantic interpolation')
    encoder = model.net.encoder
    old_embedding = encoder.MaskC_structure_tokens_embed.weight[:32].clone()
    encoder.MaskC_structure_tokens_embed.weight[:32].copy_(probability @ old_embedding)
    head = model.net.output_heads.coarse_structure_head[-1]
    weight, bias = head.weight[:32].clone(), head.bias[:32].clone()
    head.weight[:32].copy_(probability @ weight)
    head.bias[:32].copy_(probability @ bias)
    return dict(temperature=float(temperature), interpolation=probability.cpu().tolist())


def trainable_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if value.requires_grad}


@torch.no_grad()
def restore_trainable_state(model, state):
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        if name not in parameters or parameters[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid adapted parameter {name}')
    for name, value in state.items():
        parameters[name].copy_(value.to(parameters[name]))
