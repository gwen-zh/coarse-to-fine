import sys
from pathlib import Path
import tempfile
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from coarse_lora import LoRALinear, trainable_state, restore_trainable_state, initialize_new_coarse_semantics


class CoarseLoraTests(unittest.TestCase):
    def test_saved_semantic_initialization_replays_without_recomputing_distances(self):
        original = SimpleNamespace(net=SimpleNamespace(
            encoder=SimpleNamespace(MaskC_structure_tokens_embed=nn.Embedding(37, 8)),
            output_heads=SimpleNamespace(coarse_structure_head=nn.Sequential(nn.Linear(8, 37)))))
        replay = copy.deepcopy(original)
        centers, old = torch.randn(32, 4), torch.randn(32, 4)
        saved = initialize_new_coarse_semantics(original, centers, old)
        initialize_new_coarse_semantics(replay, centers + 1000, old, initialization=saved)
        self.assertTrue(torch.equal(original.net.encoder.MaskC_structure_tokens_embed.weight,
                                    replay.net.encoder.MaskC_structure_tokens_embed.weight))
        self.assertTrue(torch.equal(original.net.output_heads.coarse_structure_head[-1].weight,
                                    replay.net.output_heads.coarse_structure_head[-1].weight))
        with self.assertRaises(ValueError):
            initialize_new_coarse_semantics(replay, centers, old,
                initialization=dict(temperature=1, interpolation=[[1.0]]))

    def test_zero_delta_initialization_and_frozen_base(self):
        torch.manual_seed(0)
        base = nn.Linear(5, 7)
        model = LoRALinear(base, rank=2)
        x = torch.randn(3, 5)
        self.assertTrue(torch.equal(model(x), base(x)))
        model(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(model.b.grad.abs().sum().item(), 0)
        state = trainable_state(model)
        self.assertEqual(set(state), {'a', 'b'})
        with torch.no_grad(): model.b.fill_(1)
        restore_trainable_state(model, state)
        self.assertTrue(torch.equal(model(x), base(x)))

    def test_restore_validates_before_mutation(self):
        model = LoRALinear(nn.Linear(3, 4), rank=2)
        original = model.a.clone()
        with self.assertRaises(ValueError):
            restore_trainable_state(model, {'a':torch.ones_like(model.a), 'b':torch.zeros(1)})
        self.assertTrue(torch.equal(model.a, original))


if __name__ == '__main__':
    unittest.main()
