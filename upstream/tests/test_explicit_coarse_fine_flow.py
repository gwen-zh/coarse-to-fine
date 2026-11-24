from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from explicit_coarse_fine_flow import ExplicitCoarseFineFlow
from joint_sequence_predictor import encode_aa
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow


class ExplicitConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.aa = encode_aa("ACDEFGHIKLMNPQRST")
        self.c = torch.randn(5, 3).cumsum(0)
        self.f = torch.randn(17, 8)
        self.args = (torch.zeros(3), torch.ones(3), torch.zeros(8), torch.ones(8))
        self.kw = dict(width=32, layers=1, coarse_dim=3, feature_dim=64)
        self.base = SequenceFeatureEnsembleFlow(*self.args, **self.kw).eval()

    def adapted(self, arm):
        torch.manual_seed(42)
        model = ExplicitCoarseFineFlow(*self.args, **self.kw, context_kind=arm).eval()
        missing, extra = model.load_state_dict(self.base.state_dict(), strict=False)
        self.assertFalse(extra)
        self.assertTrue(all(k.startswith("context_adapter.") for k in missing))
        return model

    def test_both_arms_replay_original_epoch_zero_velocity_and_sampling(self):
        expected = self.base.fine_velocity(self.aa, self.f, .2, self.c)
        sample = self.base.sample_fine(self.aa, self.c, 15, 3)
        for arm in ["raw_context", "local_geometry"]:
            model = self.adapted(arm)
            torch.testing.assert_close(model.fine_velocity(self.aa, self.f, .2, self.c), expected, atol=0, rtol=0)
            torch.testing.assert_close(model.sample_fine(self.aa, self.c, 15, 3), sample, atol=0, rtol=0)

    def test_geometry_is_rigid_invariant_and_cache_updates_for_changed_condition(self):
        model = self.adapted("local_geometry")
        native = model.context_features(self.c, len(self.aa))
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = model.context_features(self.c @ rotation + 5., len(self.aa))
        torch.testing.assert_close(native, transformed, atol=2e-5, rtol=2e-5)
        changed = self.c.clone()
        changed[2, 0] += 1.
        self.assertFalse(torch.equal(model.context_features(changed, len(self.aa)), native))

    def test_capacity_and_initial_adapter_weights_match_and_gradient_reaches_adapter(self):
        a, b = self.adapted("raw_context"), self.adapted("local_geometry")
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        for key, value in a.context_adapter.state_dict().items():
            torch.testing.assert_close(value, b.context_adapter.state_dict()[key], atol=0, rtol=0)
        for model in [a, b]:
            loss = (model.fine_velocity(self.aa, self.f, .3, self.c) - self.f).square().mean()
            loss.backward()
            grad = model.context_adapter[-1].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()), 0.)

    def test_grad_enabled_geometry_is_not_served_from_detached_cache(self):
        model = self.adapted("local_geometry")
        model.context_features(self.c, len(self.aa))
        c = self.c.clone().requires_grad_(True)
        features = model.context_features(c, len(self.aa))
        features.square().sum().backward()
        self.assertIsNotNone(c.grad)
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater(float(c.grad.norm()), 0.)


if __name__ == "__main__":
    unittest.main()
