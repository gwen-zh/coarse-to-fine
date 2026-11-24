from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from par_coarse_capacity import geometry_diagnostic, evaluation_keys, install_flow_lora, learning_rate
from par_two_protein_fit import sample_keys
from par_coarse_capacity import update


class CapacityTests(unittest.TestCase):
    def test_proper_rotation_and_translation_preserve_shape_but_mirror_is_detected(self):
        rng = np.random.default_rng(41)
        reference = rng.normal(size=(23, 3)) * 10
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, -1] *= np.linalg.det(q)
        values = np.stack([reference @ q + 37, reference @ np.diag([-1, 1, 1]) + 19])
        r = geometry_diagnostic(values, reference, 90)
        self.assertLess(r['proper_RMSD_A'][0], 2e-6)
        self.assertGreater(r['proper_RMSD_A'][1], 2)
        self.assertLess(r['reflection_permitted_RMSD_A'][1], 2e-6)
        self.assertEqual(r['mirror_like'].tolist(), [False, True])

    def test_nonmirror_shape_error_is_not_hidden_by_reflection(self):
        rng = np.random.default_rng(9)
        reference = rng.normal(size=(23, 3)) * 10
        r = geometry_diagnostic(reference[None] * 2, reference, 90)
        self.assertGreater(r['reflection_permitted_RMSD_A'][0], 2)
        self.assertFalse(r['mirror_like'][0])

    def test_historical_noise_prefix_and_fresh_half_are_fixed_and_disjoint(self):
        keys = evaluation_keys(2026)
        self.assertEqual(keys[:32], sample_keys(2026, 32))
        self.assertEqual(len(set(keys)), 64)
        self.assertNotEqual(keys, evaluation_keys(2027))
        self.assertEqual(keys, evaluation_keys(2026))

    def test_lora_preserves_initial_projection_and_only_adapters_receive_gradient(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mhba = nn.Module()
                self.mhba.mha = nn.Module()
                self.mhba.mha.to_qkv = nn.Linear(8, 24).requires_grad_(False)
                self.mhba.mha.to_out_node = nn.Linear(8, 8).requires_grad_(False)
        model = nn.Module()
        model.atom_diffusion = nn.Module()
        model.atom_diffusion.nn = nn.Module()
        model.atom_diffusion.nn.transformer_layers = nn.ModuleList([Layer(), Layer(), Layer()])
        x = torch.randn(2, 4, 8)
        base = model.atom_diffusion.nn.transformer_layers[-1].mhba.mha.to_qkv
        before = base(x).detach()
        view = nn.Module()
        names = install_flow_lora(model, view, rank=2, last_blocks=2)
        layer = model.atom_diffusion.nn.transformer_layers[-1].mhba.mha.to_qkv
        torch.testing.assert_close(before, layer(x), atol=0, rtol=0)
        self.assertEqual(len(names), 4)
        layer(x).square().mean().backward()
        self.assertGreater(float(layer.b.grad.norm()), 0)
        self.assertIsNone(base.weight.grad)
        self.assertTrue(all('base' not in name for name in view.state_dict()))
        self.assertIsInstance(model.atom_diffusion.nn.transformer_layers[0].mhba.mha.to_qkv, nn.Linear)

    def test_lr_budget_is_fixed_and_global_offset_cannot_change_schedule(self):
        self.assertEqual(learning_rate(0), 1e-4)
        self.assertEqual(learning_rate(1023), 1e-5)
        self.assertGreater(learning_rate(255), learning_rate(767))
        with self.assertRaises(ValueError):
            learning_rate(1024)

    def test_update_advances_noise_draws_and_retains_local_checkpoint_indices(self):
        def fake_update(model, optimizer, cases, bank, arm, seed, step, protocol):
            self.assertEqual(arm, 'sequence_uniform')
            self.assertEqual(protocol['lr'], protocol['min_lr'])
            return dict(update=step+1)
        with patch('par_coarse_capacity.original_update', side_effect=fake_update):
            for local, global_update in [(0, 4097), (1023, 5120)]:
                row = update(None, None, None, None, 2026, local)
                self.assertEqual(row['update'], local+1)
                self.assertEqual(row['global_update'], global_update)


if __name__ == '__main__':
    unittest.main()
