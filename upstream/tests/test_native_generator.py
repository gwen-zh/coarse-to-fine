import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from native_generator import choose_fine_prefix, raw_fine_logits


class NativeGeneratorTests(unittest.TestCase):
    def test_generated_prefix_policy_uses_available_prediction(self):
        native, predicted = torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long)
        rng = torch.Generator().manual_seed(0)
        self.assertTrue(torch.equal(choose_fine_prefix(native, predicted, 1, rng), predicted))
        self.assertTrue(torch.equal(choose_fine_prefix(native, predicted, 0, rng), native))
        with self.assertRaises(ValueError):
            choose_fine_prefix(native, torch.full_like(predicted, 32), 1, rng)

    def test_fine_head_uses_coarse_input_and_preserves_autograd(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(37, 4)
                self.head = nn.Linear(4, 4101)
            def forward(self, var_structure_tokens, sequence_tokens, auxiliary_embeddings, lengths, scale):
                self.last = dict(scale=scale, lengths=lengths, tokens=var_structure_tokens)
                return SimpleNamespace(structure_logits=self.head(self.embed(var_structure_tokens)))
        model = SimpleNamespace(net=Net())
        coarse = torch.tensor([[34, 2, 3, 33]])
        logits = raw_fine_logits(model, torch.zeros_like(coarse), coarse)
        self.assertEqual(logits.shape, (1, 2, 4096))
        self.assertEqual(model.net.last['scale'], 2)
        logits.sum().backward()
        self.assertGreater(model.net.embed.weight.grad[2].abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
