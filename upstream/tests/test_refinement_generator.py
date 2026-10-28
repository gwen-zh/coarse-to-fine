import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from depth_generator import DepthGenerator, sequence_ids
from refinement_generator import FrozenCoarseRefiner, compensation_labels, greedy_oracle_refinement

spec = importlib.util.spec_from_file_location('bridge_refinement_test', ROOT / 'Hierarchical_VQ_VAE/esm/models/residual_bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RefinementTests(unittest.TestCase):
    def test_relabel_depends_on_actual_prefix(self):
        bridge = module.ResidualTokenBridge([torch.tensor([[0.], [10.]]),
                                             torch.tensor([[2.], [12.]]), torch.tensor([[0.], [1.]])],
                                            torch.tensor([[10.], [12.], [13.]]))
        z = torch.tensor([[[12.]]])
        true_coarse = torch.tensor([[1]])
        wrong_coarse = torch.tensor([[0]])
        self.assertEqual(compensation_labels(bridge, z, [true_coarse], 1).item(), 0)
        self.assertEqual(compensation_labels(bridge, z, [wrong_coarse], 1).item(), 1)
        oracle = greedy_oracle_refinement(bridge, z, wrong_coarse)
        self.assertEqual(oracle.tolist(), [[[0, 1, 0]]])
        self.assertTrue(torch.equal(oracle[..., 0], wrong_coarse))
        self.assertTrue(torch.equal(bridge.prefix_latent(oracle), z))

    def test_refiner_training_cannot_change_coarse_policy(self):
        torch.manual_seed(0)
        base = DepthGenerator(sizes=(3, 5, 7), context_width=8, width=16, layers=1)
        model = FrozenCoarseRefiner(base)
        h = torch.randn(1, 4, 8)
        seq = sequence_ids('ACDE')[None]
        before, _ = model.generate(h, seq, seed=9, coarse_steps=2)
        model.train()
        self.assertFalse(model.base.training)
        current = torch.full_like(seq, 5)
        logits = model(h, seq, [before[..., 0]], current, 1)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([0, 1, 2, 3])).backward()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertGreater(model.refiner.heads[1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            for p in model.refiner.parameters():
                p.add_(torch.randn_like(p) * 0.2)
        after, _ = model.generate(h, seq, seed=9, coarse_steps=2)
        self.assertTrue(torch.equal(before[..., 0], after[..., 0]))


if __name__ == '__main__':
    unittest.main()
