"""Small sequence-conditioned generator for testing new depth-wise token semantics.

Context features come from the released sequence-only generator, never from the
reference-coarse feature cache. This is a downstream adapter, not a replacement
for or a speedup over the frozen upstream generator.
"""
import math
import torch
from torch import nn

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYXBZUO'


def sequence_ids(sequence):
    vocabulary = {letter: i + 1 for i, letter in enumerate(ALPHABET)}
    return torch.tensor([vocabulary.get(letter, vocabulary['X']) for letter in sequence], dtype=torch.long)


class DepthGenerator(nn.Module):
    def __init__(self, sizes=(32, 512, 4096), context_width=1536, width=128, layers=2):
        super().__init__()
        self.sizes = tuple(sizes)
        self.context = nn.Sequential(nn.LayerNorm(context_width), nn.Linear(context_width, width))
        self.sequence = nn.Embedding(len(ALPHABET) + 1, width)
        self.depth = nn.Embedding(len(sizes), width)
        self.tokens = nn.ModuleList(nn.Embedding(size + 1, width) for size in sizes)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.1,
                                            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.heads = nn.ModuleList(nn.Linear(width, size) for size in sizes)

    def forward(self, features, sequence, lower, current, stage):
        if len(lower) != stage or stage >= len(self.sizes):
            raise ValueError('Only the already available prefix may condition a stage')
        if features.shape[:2] != sequence.shape or current.shape != sequence.shape:
            raise ValueError('Feature/sequence/current-token lengths disagree')
        x = self.context(features) + self.sequence(sequence) + self.depth.weight[stage]
        for level, ids in enumerate(lower):
            x = x + self.tokens[level](ids)
        x = x + self.tokens[stage](current)
        n, width = x.shape[1:]
        positions = torch.arange(n, device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=x.dtype)
                              * (-math.log(10000.0) / width))
        pos = torch.zeros(n, width, device=x.device, dtype=x.dtype)
        pos[:, 0::2] = torch.sin(positions * frequency)
        pos[:, 1::2] = torch.cos(positions * frequency)
        return self.heads[stage](self.norm(self.transformer(x + pos[None])))

    @torch.no_grad()
    def refine_coarse(self, features, sequence, coarse):
        """Continue an externally sampled coarse plan without resampling it."""
        if len(self.sizes) < 2 or coarse.shape != sequence.shape:
            raise ValueError('Expected a coarse plan matching the sequence batch')
        if coarse.dtype != torch.long or torch.any((coarse < 0) | (coarse >= self.sizes[0])):
            raise ValueError('Coarse plan must contain valid unmasked integer tokens')
        self.eval()
        outputs = [coarse]
        for stage, size in enumerate(self.sizes[1:], start=1):
            current = torch.full_like(sequence, size)
            outputs.append(self(features, sequence, outputs, current, stage).argmax(-1))
        return torch.stack(outputs, dim=-1), len(self.sizes) - 1

    @torch.no_grad()
    def generate(self, features, sequence, seed, coarse_steps=5):
        """Only sequence-derived context is accepted; no native tokens are inputs."""
        self.eval()
        if coarse_steps < 1:
            raise ValueError('coarse_steps must be positive')
        rng = torch.Generator(device=features.device).manual_seed(seed)
        outputs = []
        calls = 0
        for stage, size in enumerate(self.sizes):
            current = torch.full(sequence.shape, size, dtype=torch.long, device=features.device)
            # Direct fine control has a single deterministic prediction, as do
            # the refinement stages; only the hierarchical coarse stage samples.
            if stage != 0 or len(self.sizes) == 1:
                logits = self(features, sequence, outputs, current, stage)
                outputs.append(logits.argmax(-1))
                calls += 1
                continue
            for step in range(coarse_steps):
                logits = self(features, sequence, outputs, current, stage)
                calls += 1
                probabilities = logits.softmax(-1)
                candidates = torch.multinomial(probabilities.reshape(-1, size), 1,
                                                generator=rng).reshape(sequence.shape)
                release = torch.rand(sequence.shape, device=features.device, generator=rng) < 1 / (coarse_steps - step)
                current = torch.where((current == size) & release, candidates, current)
            if torch.any(current == size):
                raise RuntimeError('Coarse sampling left masked residues')
            outputs.append(current)
        return torch.stack(outputs, dim=-1), calls
