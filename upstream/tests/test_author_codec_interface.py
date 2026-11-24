import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from block_local_frames import FrameGeometryError
from hierarchical_latent_contract import pool_ca
from run_author_codec_interface import evaluate_values, project_all, project_one


class IdentityCodec:
    def encode(self, x):
        self.x = x - x[:, 1].mean(0)
        return pool_ca(self.x), torch.zeros((len(x), 8))

    def decode(self, coarse, fine, quantized):
        assert not quantized
        return self.x.clone()


class InterfaceTests(unittest.TestCase):
    def test_projection_restores_input_origin_including_partial_block(self):
        torch.manual_seed(81)
        x = torch.randn(13, 3, 3) + torch.tensor([11., -27., 54.])
        y, c, f, drift = project_one(IdentityCodec(), x)
        np.testing.assert_allclose(y, x.numpy(), atol=4e-6, rtol=1e-6)
        self.assertEqual(c.shape, (4, 3)); self.assertEqual(f.shape, (13, 8))
        self.assertLess(drift, 5e-6)

    def test_frame_failure_is_retained_and_disables_group_metrics(self):
        class Failure(IdentityCodec):
            calls = 0
            def encode(self, x):
                self.calls += 1
                if self.calls == 2:
                    raise FrameGeometryError("injected invalid trace")
                return super().encode(x)
        x = np.random.default_rng(81).normal(size=(3, 13, 3, 3)).astype(np.float32)
        projected, audit = project_all(Failure(), x)
        self.assertEqual(len(projected["backbone"]), 3)
        np.testing.assert_array_equal(projected["valid"], [True, False, True])
        self.assertTrue(np.isnan(projected["backbone"][1]).all())
        self.assertEqual(audit["failures"][0]["sample"], 1)
        result = evaluate_values(projected["backbone"], None, audit)
        self.assertFalse(result["metrics_available"])
        self.assertNotIn("mean_sample_nearest_reference_ca_rmsd_A", result)

    def test_interface_errors_stop_instead_of_silently_falling_back(self):
        class WrongShape(IdentityCodec):
            def encode(self, x):
                return torch.zeros((len(x), 16)), torch.zeros((len(x), 8))
        x = torch.zeros(13, 3, 3)
        with self.assertRaisesRegex(ValueError, "interface"):
            project_one(WrongShape(), x)
        x[1, 1, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Finite"):
            project_one(IdentityCodec(), x)


if __name__ == "__main__":
    unittest.main()
