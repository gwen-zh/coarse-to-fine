"""Preserve published 32 -> 4096 token semantics while adapting both stages."""
import torch

from coarse_lora import install_coarse_lora


def install_native_lora(model, last_blocks=8, rank=8):
    names = install_coarse_lora(model, last_blocks, rank)
    model.net.output_heads.fine_structure_head.requires_grad_(True)
    return names


def choose_fine_prefix(native, sampled, probability, generator):
    if native.shape != sampled.shape or native.dtype != torch.long or sampled.dtype != torch.long:
        raise ValueError('Coarse prefix shape or dtype mismatch')
    if not 0 <= probability <= 1 or torch.any((sampled < 0) | (sampled >= 32)):
        raise ValueError('Invalid sampled-prefix policy')
    return sampled if float(torch.rand((), generator=generator)) < probability else native


def raw_fine_logits(model, sequence, coarse):
    n = coarse.shape[1] - 2
    return model.net(var_structure_tokens=coarse, sequence_tokens=sequence,
                     auxiliary_embeddings=None,
                     lengths=torch.tensor([n], device=coarse.device), scale=2).structure_logits[:, 1:-1, :4096]
