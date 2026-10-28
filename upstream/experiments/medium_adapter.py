"""Small residual heads on frozen generator features, with matched direct control."""
import torch
from torch import nn


class MediumAdapter(nn.Module):
    def __init__(self, width=1536, vocabulary=512, rank=32):
        super().__init__()
        self.classifier = nn.Linear(width, vocabulary)
        self.embedding = nn.Parameter(torch.randn(vocabulary, rank) * 0.02)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h, permutation=None):
        logits = self.classifier(h)
        probabilities = logits.softmax(-1)
        if permutation is not None:
            probabilities = probabilities[:, permutation]
        residual = self.up(probabilities @ self.embedding)
        return h + residual, logits


class DirectAdapter(nn.Module):
    def __init__(self, width=1536, rank=278):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h, permutation=None):
        return h + self.up(torch.nn.functional.gelu(self.down(h))), None


def matched_direct_rank(width, parameter_count):
    return max(1, round((parameter_count - width) / (2 * width + 1)))
