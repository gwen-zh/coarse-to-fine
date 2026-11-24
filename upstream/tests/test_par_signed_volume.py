from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from par_signed_volume import signed_features, choose_weight


class SignedVolumeTests(unittest.TestCase):
    def test_proper_rotation_translation_invariance_and_reflection_oddness(self):
        torch.manual_seed(11)
        points = torch.randn(2, 23, 3)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        rotation[:, -1] *= torch.linalg.det(rotation)
        initial = signed_features(points)
        self.assertEqual(initial.shape, (2, 48))
        torch.testing.assert_close(signed_features(points @ rotation + 17), initial, atol=4e-6, rtol=4e-5)
        mirror = torch.diag(torch.tensor([-1., 1., 1.]))
        torch.testing.assert_close(signed_features(points @ mirror), -initial, atol=1e-7, rtol=1e-7)

    def test_identical_pairwise_distances_can_have_different_signed_supervision(self):
        points = torch.randn(2, 23, 3)
        reflected = points @ torch.diag(torch.tensor([-1., 1., 1.]))
        torch.testing.assert_close(torch.cdist(points, points), torch.cdist(reflected, reflected), atol=0, rtol=0)
        loss = (signed_features(reflected) - signed_features(points)).square().mean()
        self.assertGreater(float(loss), .001)

    def test_degenerate_inputs_and_ordinary_gradients_are_finite(self):
        target = torch.randn(1, 23, 3)
        for points in [torch.zeros(1, 23, 3), torch.randn(1, 23, 3)]:
            points.requires_grad_()
            loss = (signed_features(points) - signed_features(target)).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(float(points.grad.norm()), 0)

    def test_weight_rule_is_median_gradient_match_with_fixed_bounds(self):
        rows = [dict(velocity_gradient_norm=v, volume_gradient_norm=1.) for v in [2., 9., 3.]]
        self.assertEqual(choose_weight(rows), 3.)
        self.assertEqual(choose_weight([dict(velocity_gradient_norm=1000., volume_gradient_norm=1.)]), 100.)
        self.assertEqual(choose_weight([dict(velocity_gradient_norm=.001, volume_gradient_norm=1.)]), .01)


if __name__ == '__main__':
    unittest.main()
