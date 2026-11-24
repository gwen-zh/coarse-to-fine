from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from multiscale_flow_solver import integrate_velocity, sample_with_coarse_solver
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from joint_sequence_predictor import encode_aa
from sequence_multiscale_ensemble import sample_levels


class CenterOnlyCodec:
    def project_coarse(self, value, length):
        counts = value.new_tensor([min(4, length - 4 * i) for i in range(len(value))])
        return value - (value * counts[:, None]).sum(0) / length


class SolverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_velocity_call_budget(self):
        for method in ["euler", "heun"]:
            calls = []
            def velocity(value, time):
                calls.append(time)
                return torch.ones_like(value)
            result = integrate_velocity(torch.zeros(3), velocity, 16, method)
            self.assertEqual(len(calls), 16)
            torch.testing.assert_close(result, torch.ones(3))

    def test_second_order_accuracy_at_equal_budget(self):
        truth = torch.exp(torch.tensor(1., dtype=torch.float64))
        initial = torch.tensor([1.], dtype=torch.float64)
        a = integrate_velocity(initial, lambda x, t: x, 16, "euler")
        b = integrate_velocity(initial, lambda x, t: x, 16, "heun")
        self.assertLess(float((b - truth).abs()), float((a - truth).abs()))

    def test_gaussian_flow_has_known_endpoint_variance(self):
        # Exact conditional flow for independent N(0,1) noise -> N(0,sigma^2).
        sigma = .25
        def velocity(x, t):
            denominator = (1 - t) ** 2 + t ** 2 * sigma ** 2
            return x * (t * sigma ** 2 - (1 - t)) / denominator
        initial = torch.tensor([-1., 1.], dtype=torch.float64)
        a = integrate_velocity(initial, velocity, 16, "euler")
        b = integrate_velocity(initial, velocity, 16, "heun")
        self.assertLess(float(a[1]), sigma)
        self.assertLess(abs(float(b[1]) - sigma), abs(float(a[1]) - sigma))

    def test_euler_replays_existing_sampler_without_global_rng_changes(self):
        torch.manual_seed(11)
        model = HierarchicalEnsembleFlow(torch.zeros(3), torch.ones(3),
            torch.zeros(8), torch.ones(8), width=32, layers=1, coarse_dim=3).eval()
        codec = CenterOnlyCodec()
        aa = encode_aa("ACDEFGHIKLMNPQRST")
        rng = torch.get_rng_state().clone()
        a = sample_levels(model, codec, aa, 63, 16, 16)
        b = sample_with_coarse_solver(model, codec, aa, 63, 16, "euler", 16)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_invalid_budgets_and_nonfinite_states_fail(self):
        for method, nfe in [("heun", 3), ("euler", 0), ("unknown", 16)]:
            with self.assertRaises(ValueError):
                integrate_velocity(torch.ones(1), lambda x, t: x, nfe, method)
        with self.assertRaises(RuntimeError):
            integrate_velocity(torch.ones(1), lambda x, t: x * float("nan"), 2)


if __name__ == "__main__":
    unittest.main()
