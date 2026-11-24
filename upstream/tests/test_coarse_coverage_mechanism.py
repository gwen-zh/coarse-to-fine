import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_coverage_mechanism import (bridge_endpoint, distance_variance, event_loss,
    flatten_gradient, gradient_projection, observable_moments, project, response_gain, trajectory)
from coarse_few_target_flow import coarse_flow_loss
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from joint_sequence_predictor import encode_aa


class RadialField(nn.Module):
    def __init__(self, value=-.5):
        super().__init__(); self.rate = nn.Parameter(torch.tensor(value))
    def coarse_velocity(self, aa, state, time): return self.rate * state
    def denormalize_coarse(self, state): return state
    def normalize_coarse(self, state): return state


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.aa = encode_aa("ACDEFGHIKLMNPQRSTVWYA")
        self.noise = torch.randn(4, 6, 3, generator=torch.Generator().manual_seed(81))

    def test_rigid_body_motion_does_not_count_as_structural_spread(self):
        x = self.noise[:1].repeat(3, 1, 1)
        rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        x[1] = x[1] @ rot + 7.; x[2] = x[2] @ rot.T - 3.
        self.assertLess(float(distance_variance(x)), 1e-12)
        self.assertGreater(float(distance_variance(self.noise)), .01)

    def test_trajectory_matches_original_sampler_and_handles_partial_block(self):
        torch.manual_seed(51)
        m = HierarchicalEnsembleFlow(torch.zeros(3), torch.ones(3), torch.zeros(8), torch.ones(8),
            width=32, layers=1, coarse_dim=3).eval()
        noise = torch.randn((6, 3), generator=torch.Generator().manual_seed(15))
        with torch.no_grad(): a = trajectory(m, self.aa, noise)[-1]; b = project(m.sample_coarse(self.aa, 15, 16), len(self.aa))
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        counts = torch.tensor([4., 4., 4., 4., 4., 1.])
        torch.testing.assert_close((a * counts[:, None]).sum(0), torch.zeros(3), atol=1e-6, rtol=0)

    def test_local_projection_identifies_single_target_contraction(self):
        m = RadialField(); initial = m.rate.detach().clone()
        endpoints = torch.stack([trajectory(m, self.aa, x)[-1] for x in self.noise])
        spread = distance_variance(endpoints); gs = flatten_gradient(spread, [m.rate])
        loss = ((m.rate + 1) * self.noise).square().mean()
        gl = flatten_gradient(loss, [m.rate]); result = gradient_projection(gs, gl)
        self.assertGreater(result["cosine"], .9999)
        self.assertLess(result["first_order_spread_change_per_unit_parameter_move"], 0.)
        self.assertTrue(torch.equal(m.rate, initial)); self.assertIsNone(m.rate.grad)

    def test_noise_response_scales_with_linear_field_and_is_stable_in_epsilon(self):
        a = RadialField(0.); b = RadialField(-.5)
        x = self.noise[0]; direction = self.noise[1]
        ga = response_gain(a, self.aa, x, direction, .01)
        gb = response_gain(b, self.aa, x, direction, .01)
        ratio = (1 - .5 / 16) ** 16
        self.assertAlmostEqual(gb / ga, ratio, places=4)
        self.assertAlmostEqual(gb, response_gain(b, self.aa, x, direction, .02), places=3)

    def test_zero_gradient_is_recorded_as_undefined_cosine(self):
        result = gradient_projection(torch.ones(3), torch.zeros(3))
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["first_order_spread_change_per_unit_parameter_move"])

    def test_distance_moments_decompose_cross_ensemble_error(self):
        g=self.noise[:2].double();r=self.noise[2:].double()
        from coarse_coverage_mechanism import distance_features
        gf=distance_features(g);rf=distance_features(r)
        cross=(gf[:,None]-rf[None]).square().mean()
        moments=observable_moments(g,r)
        decomposition=moments["distance_mean_bias_RMSE_A"]**2+moments["generated_distance_variance_A2"]+moments["reference_distance_variance_A2"]
        self.assertAlmostEqual(float(cross),decomposition,places=10)
        self.assertIsNone(observable_moments(g,r[:1])["distance_variance_ratio"])

    def test_spread_parameter_gradient_matches_finite_difference(self):
        m=RadialField();original=m.rate.detach().clone()
        def score():return distance_variance(torch.stack([trajectory(m,self.aa,z)[-1] for z in self.noise]))
        analytic=float(flatten_gradient(score(),[m.rate])[0]);epsilon=.001
        with torch.no_grad():
            m.rate.copy_(original+epsilon);plus=float(score())
            m.rate.copy_(original-epsilon);minus=float(score())
            m.rate.copy_(original)
        self.assertAlmostEqual(analytic,(plus-minus)/(2*epsilon),places=3)
        self.assertIsNone(m.rate.grad)

    def test_bridge_endpoint_is_exact_for_oracle_velocity(self):
        target = project(self.noise[0], len(self.aa)); noise = self.noise[1]
        class Oracle(RadialField):
            def coarse_velocity(self, aa, state, time): return target - noise
        for time in [.1, .5, .9]:
            torch.testing.assert_close(bridge_endpoint(Oracle(), self.aa, target, noise, time), target, atol=1e-6, rtol=1e-6)

    def test_training_gradient_rejects_development_and_replays_native_draws(self):
        m = RadialField(); teacher = copy.deepcopy(m).requires_grad_(False)
        case = dict(split="train", sequence="ACDEFGHIKLMNPQRSTVWYA", target=torch.zeros(21, 3, 3), coarse=self.noise[0])
        loss, digest = event_loss(m, teacher, case, 71, "native")
        class Wrapper:
            coarse_model = m
        reference, info = coarse_flow_loss(Wrapper(), self.aa, case["coarse"], torch.Generator().manual_seed(71))
        torch.testing.assert_close(loss, reference, atol=0, rtol=0); self.assertEqual(digest, info["draw_sha256"])
        case["split"] = "validation"
        with self.assertRaises(ValueError): event_loss(m, teacher, case, 71, "native")


if __name__ == "__main__": unittest.main()
