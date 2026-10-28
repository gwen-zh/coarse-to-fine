"""Load small medium-level tokenizer patches without modifying the other levels.

Patches that relabel tokens require corresponding generator adaptation. These
weights upgrade the tokenizer component, not the released diffusion checkpoint.
Apply before constructing an optimizer for further tokenizer training.
"""
from pathlib import Path

import torch


@torch.no_grad()
def apply_medium_patch(encoder, decoder, checkpoint):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("level") != "medium":
        raise ValueError("Unsupported tokenizer patch")
    source = [checkpoint["encoder_weight"], checkpoint["decoder_weight"]]
    destination = [encoder.medium_codebook.embeddings, decoder.medium_embed.weight]
    # Validate the entire patch before mutating either module.
    for value, parameter in zip(source, destination):
        if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
            raise ValueError("Tokenizer patch shape mismatch")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Tokenizer patch contains nonfinite values")
    converted = [value.to(device=parameter.device, dtype=parameter.dtype)
                 for value, parameter in zip(source, destination)]
    for value, parameter in zip(converted, destination):
        parameter.copy_(value)
    return {key: value for key, value in checkpoint.items()
            if key not in ("encoder_weight", "decoder_weight")}
