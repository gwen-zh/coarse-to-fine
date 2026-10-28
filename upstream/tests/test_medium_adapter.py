import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from medium_adapter import MediumAdapter, DirectAdapter, matched_direct_rank


class MediumAdapterTests(unittest.TestCase):
    def test_identity_initialization_and_learning(self):
        torch.manual_seed(2)
        h = torch.randn(8, 12)
        model = MediumAdapter(width=12, vocabulary=7, rank=3)
        modified, logits = model(h)
        self.assertTrue(torch.equal(h, modified))
        loss = modified.square().mean() + torch.nn.functional.cross_entropy(logits, torch.arange(8) % 7)
        loss.backward()
        self.assertGreater(model.up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(), 0)
        # The correction uses predicted medium probabilities, no target labels.
        with torch.no_grad():
            model.up.weight.fill_(0.1)
        normal, _ = model(h)
        shuffled, _ = model(h, permutation=torch.arange(6, -1, -1))
        self.assertGreater((normal - shuffled).abs().sum().item(), 0)

    def test_direct_parameter_budget(self):
        medium = MediumAdapter()
        count = sum(p.numel() for p in medium.parameters())
        direct = DirectAdapter(rank=matched_direct_rank(1536, count))
        self.assertLess(abs(sum(p.numel() for p in direct.parameters()) - count) / count, 0.002)
        h = torch.randn(2, 1536)
        amended, labels = direct(h)
        self.assertTrue(torch.equal(h, amended))
        self.assertIsNone(labels)


if __name__ == '__main__':
    unittest.main()
