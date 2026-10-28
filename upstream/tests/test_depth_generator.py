import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from depth_generator import DepthGenerator, sequence_ids


class DepthGeneratorTests(unittest.TestCase):
    def test_external_coarse_is_preserved_and_not_resampled(self):
        model = DepthGenerator(sizes=(3, 5, 7), context_width=8, width=16, layers=1)
        seq = sequence_ids('ACDE')[None]
        h = torch.randn(1, 4, 8)
        coarse = torch.tensor([[0, 2, 1, 0]])
        original = coarse.clone()
        stages = []
        hook = model.register_forward_pre_hook(lambda m, inputs: stages.append(inputs[-1]))
        result, calls = model.refine_coarse(h, seq, coarse)
        hook.remove()
        self.assertEqual(stages, [1, 2])
        self.assertEqual(calls, 2)
        self.assertTrue(torch.equal(result[..., 0], original))
        self.assertTrue(torch.equal(coarse, original))
        with self.assertRaises(ValueError):
            model.refine_coarse(h, seq, torch.full_like(coarse, 3))
        with self.assertRaises(ValueError):
            model.refine_coarse(h, seq, coarse.float())

    def test_sampling_uses_predicted_prefix_and_is_reproducible(self):
        torch.manual_seed(1)
        model = DepthGenerator(sizes=(3, 5, 7), context_width=8, width=16, layers=1)
        seq = sequence_ids('ACDEFG')[None]
        h = torch.randn(1, 6, 8)
        records = []
        hook = model.register_forward_pre_hook(lambda m, inputs: records.append(
            (inputs[-1], [x.detach().clone() for x in inputs[2]])))
        tokens, calls = model.generate(h, seq, seed=4, coarse_steps=3)
        hook.remove()
        self.assertEqual(calls, 5)
        self.assertEqual(tokens.shape, (1, 6, 3))
        for level, size in enumerate(model.sizes):
            self.assertTrue(((tokens[..., level] >= 0) & (tokens[..., level] < size)).all())
        for stage, lower in records:
            for i, values in enumerate(lower):
                self.assertTrue(torch.equal(values, tokens[..., i]))
        repeated, _ = model.generate(h, seq, seed=4, coarse_steps=3)
        self.assertTrue(torch.equal(tokens, repeated))
        with self.assertRaises(ValueError):
            model(h, seq, [tokens[..., 0]], tokens[..., 0], 0)

    def test_direct_control_and_training_gradients(self):
        model = DepthGenerator(sizes=(7,), context_width=8, width=16, layers=1)
        seq = sequence_ids('ACDX')[None]
        h = torch.randn(1, 4, 8)
        current = torch.full_like(seq, 7)
        logits = model(h, seq, [], current, 0)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([0, 1, 2, 3])).backward()
        self.assertGreater(model.context[1].weight.grad.abs().sum().item(), 0)
        a, calls = model.generate(h, seq, seed=1)
        b, _ = model.generate(h, seq, seed=2)
        self.assertEqual(calls, 1)
        self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
