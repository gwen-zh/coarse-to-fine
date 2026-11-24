from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_few_target_flow import FixedFineCoarseFlow, coarse_flow_loss, event, learning_rate
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from joint_sequence_predictor import encode_aa
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow


class CoarseFewTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)
        args = (torch.zeros(3), torch.ones(3), torch.zeros(8), torch.ones(8))
        kw = dict(width=32, layers=1, coarse_dim=3, feature_dim=64)
        self.coarse = SequenceFeatureEnsembleFlow(*args, **kw)
        self.fine = ExplicitCoarseFineFlow(*args, **kw, context_kind="local_geometry")
        self.aa = encode_aa("ACDEFGHIKLMNPQRST")
        self.c = torch.randn(5, 3).cumsum(0)

    def test_coarse_update_keeps_fine_sampler_bitwise_fixed(self):
        model = FixedFineCoarseFlow(self.coarse, self.fine)
        before_fine = model.sample_fine(self.aa, self.c, 33, 3)
        before_coarse = model.sample_coarse(self.aa, 47, 3)
        frozen = {k: v.clone() for k, v in self.fine.state_dict().items()}
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=.01)
        model.train(); self.assertFalse(model.fine_model.training)
        opt.zero_grad()
        loss, _ = coarse_flow_loss(model, self.aa, self.c, torch.Generator().manual_seed(64))
        loss.backward(); opt.step()
        self.assertFalse(torch.equal(before_coarse, model.sample_coarse(self.aa, 47, 3)))
        torch.testing.assert_close(before_fine, model.sample_fine(self.aa, self.c, 33, 3), atol=0, rtol=0)
        self.assertTrue(all(v.grad is None for v in self.fine.parameters()))
        for k, value in self.fine.state_dict().items(): torch.testing.assert_close(value, frozen[k], atol=0, rtol=0)

    def test_aliasing_is_rejected_and_only_coarse_parameters_train(self):
        with self.assertRaises(ValueError): FixedFineCoarseFlow(self.coarse, self.coarse)
        model = FixedFineCoarseFlow(self.coarse, self.fine)
        trainable = {id(v) for v in model.trainable_parameters()}
        self.assertTrue(trainable)
        self.assertFalse(trainable & {id(v) for v in self.fine.parameters()})
        for name in ["fine_input", "fine_head", "coarse_to_fine"]:
            self.assertTrue(all(not p.requires_grad for p in getattr(self.coarse, name).parameters()))

    def test_local_noise_replays_loss_gradients_without_changing_global_rng(self):
        model = FixedFineCoarseFlow(self.coarse, self.fine)
        before = torch.get_rng_state().clone()
        loss1, a = coarse_flow_loss(model, self.aa, self.c, torch.Generator().manual_seed(19))
        loss2, b = coarse_flow_loss(model, self.aa, self.c, torch.Generator().manual_seed(19))
        self.assertEqual(a, b); torch.testing.assert_close(loss1, loss2, atol=0, rtol=0)
        ga = torch.autograd.grad(loss1, self.coarse.coarse_head[-1].weight)[0]
        gb = torch.autograd.grad(loss2, self.coarse.coarse_head[-1].weight)[0]
        torch.testing.assert_close(ga, gb, atol=0, rtol=0)
        self.assertGreater(float(ga.norm()), 0.)
        torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)

    def test_all_targets_receive_exact_declared_exposures(self):
        for total, expected in [(2048, 1024), (8192, 4096)]:
            events = [[] for _ in range(8)]
            for step in range(total):
                target, counters = event(step); events[target] += counters
            self.assertEqual(events, [list(range(expected)) for _ in range(8)])
        self.assertAlmostEqual(learning_rate(0), 3e-4)
        self.assertAlmostEqual(learning_rate(2047), 3e-5)
        self.assertEqual(learning_rate(2048), learning_rate(0))
        self.assertEqual(learning_rate(8191), learning_rate(2047))


if __name__ == "__main__":
    unittest.main()
