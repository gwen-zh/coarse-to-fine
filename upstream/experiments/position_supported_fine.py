"""Fine sampling restricted to code/within-block-position pairs seen in TRAIN.

Counts are structural tokenizer semantics, not a reference-geometry acceptance
rule. This does not ensure valid bonds or correct folds.
"""
import math
import torch


def support_rows(counts, length, device):
    counts = torch.as_tensor(counts, device=device)
    if counts.ndim != 2 or counts.shape[1] != 4 or (counts < 0).any():
        raise ValueError('Nonnegative Kx4 training counts required')
    if (counts.sum(0) == 0).any():
        raise ValueError('Every within-block position requires training support')
    return counts.T[torch.arange(length, device=device) % 4]


@torch.no_grad()
def sample_supported(fine_model, features, coarse, counts, seed, steps=16):
    if steps < 1:
        raise ValueError('Positive steps required')
    n = len(features)
    support = support_rows(counts, n, features.device) > 0
    if support.shape[1] != fine_model.fine_size:
        raise ValueError('Codebook vocabulary mismatch')
    steps = min(n, steps)
    rng = torch.Generator(device=features.device).manual_seed(seed)
    fine = torch.full((n,), fine_model.fine_size, dtype=torch.long, device=features.device)
    for i in range(steps):
        remaining = (fine == fine_model.fine_size).nonzero().flatten()
        logits = fine_model(features, coarse, fine, len(remaining) / n)[remaining]
        logits = logits.masked_fill(~support[remaining], -torch.inf)
        probs = logits.softmax(-1)
        proposal = torch.multinomial(probs, 1, generator=rng).flatten()
        confidence = probs.gather(1, proposal[:, None]).flatten()
        count = min(len(remaining), max(1, math.ceil((i+1)*n/steps)-(n-len(remaining))))
        chosen = confidence.topk(count).indices
        fine[remaining[chosen]] = proposal[chosen]
    if (fine == fine_model.fine_size).any() or not support.gather(1, fine[:, None]).all():
        raise RuntimeError('Unsupported/unfilled fine output')
    return fine


@torch.no_grad()
def sample_position_prior(counts, length, seed, device='cpu'):
    weights = support_rows(counts, length, device).float()
    rng = torch.Generator(device=device).manual_seed(seed)
    return torch.multinomial(weights / weights.sum(-1, keepdim=True), 1, generator=rng).flatten()
