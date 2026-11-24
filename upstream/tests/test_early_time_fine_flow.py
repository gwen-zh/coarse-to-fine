from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from early_time_fine_flow import fine_time_loss, transform_time
from fine_geometry_flow import fine_flow_loss
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from joint_sequence_predictor import encode_aa


class EarlyTimeFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(12)
        self.model = HierarchicalEnsembleFlow(torch.zeros(3), torch.ones(3),
            torch.zeros(8), torch.ones(8), width=32, layers=1, coarse_dim=3).eval()
        self.adapter = SimpleNamespace(fine_model=self.model)
        self.aa = encode_aa("ACDEFGHIKLMNPQRST")
        self.c, self.f = torch.randn(5, 3), torch.randn(17, 8)

    def test_uniform_loss_and_all_parameter_gradients_replay_original(self):
        loss, old = fine_flow_loss(self.adapter, self.aa, self.c, self.f,
                                   torch.Generator().manual_seed(321))
        loss.backward()
        gradients = {k: p.grad.clone() for k, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        replay, new = fine_time_loss(self.adapter, self.aa, self.c, self.f,
                                    torch.Generator().manual_seed(321), "uniform")
        torch.testing.assert_close(loss, replay, atol=0, rtol=0)
        self.assertEqual(old["draw_sha256"], new["draw_sha256"])
        replay.backward()
        for k, p in self.model.named_parameters():
            if k in gradients:
                torch.testing.assert_close(p.grad, gradients[k], atol=0, rtol=0)

    def test_early_inverse_cdf_has_full_support_and_declared_mass(self):
        u = torch.linspace(0, 1, 10001)
        t = transform_time(u, "early")
        self.assertEqual(float(t[0]), 0.); self.assertEqual(float(t[-1]), 1.)
        self.assertTrue((t[1:] > t[:-1]).all())
        self.assertAlmostEqual(float((t <= .25).float().mean()), .625, places=3)
        torch.testing.assert_close(transform_time(torch.tensor([.1, .625, .9]), "early"),
                                   torch.tensor([.04, .25, .8]))

    def test_arms_share_gaussian_uniform_draws_but_change_time(self):
        a, x = fine_time_loss(self.adapter, self.aa, self.c, self.f,
            torch.Generator().manual_seed(322), "uniform")
        b, y = fine_time_loss(self.adapter, self.aa, self.c, self.f,
            torch.Generator().manual_seed(322), "early")
        self.assertEqual(x["draw_sha256"], y["draw_sha256"])
        self.assertEqual(x["uniform_time"], y["uniform_time"])
        self.assertNotEqual(x["time"], y["time"])
        self.assertTrue(torch.isfinite(a) and torch.isfinite(b))


if __name__ == "__main__":
    unittest.main()
