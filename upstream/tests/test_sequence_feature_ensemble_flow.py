from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from joint_sequence_predictor import encode_aa
from sequence_feature_ensemble_flow import SequenceFeatureEnsembleFlow


class SequenceFeatureFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sequence = "ACDEFGHIKLMNPQRST"
        self.aa = encode_aa(self.sequence)

    def make(self, mode):
        torch.manual_seed(17)
        return SequenceFeatureEnsembleFlow(torch.zeros(3), torch.ones(3),
            torch.zeros(8), torch.ones(8), width=32, layers=1, coarse_dim=3,
            feature_mode=mode, feature_dim=64).eval()

    def test_parameter_counts_and_initial_weights_match(self):
        a, b = self.make("AA_lookup"), self.make("ESM3_context")
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]))

    def test_contextual_condition_is_explicit_and_frozen(self):
        model = self.make("ESM3_context")
        h = torch.randn(len(self.aa), 64, requires_grad=True)
        model.attach_feature_bank({self.sequence: h})
        c, f = torch.randn(5, 3), torch.randn(17, 8)
        loss, _ = model.flow_loss(self.aa, c, f, coarse_jitter_std=0., generator=torch.Generator().manual_seed(19))
        loss.backward()
        self.assertIsNone(h.grad)
        for module in [model.sequence_projection, model.transformer, model.coarse_head, model.fine_head]:
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0.)

    def test_missing_or_misaligned_features_fail(self):
        model = self.make("ESM3_context")
        with self.assertRaises(ValueError):
            model.aa_conditions(self.aa)
        with self.assertRaises(ValueError):
            model.attach_feature_bank({self.sequence: torch.zeros(16, 64)})
        model.attach_feature_bank({self.sequence: torch.randn(17, 64)})
        with self.assertRaises(ValueError):
            model.aa_conditions(encode_aa(self.sequence[::-1]))

    def test_feature_intervention_changes_predictions_and_seeds_replay(self):
        model = self.make("ESM3_context")
        model.attach_feature_bank({self.sequence: torch.randn(17, 64)})
        a = model.sample_latents(self.aa, 31, 3, 3)
        b = model.sample_latents(self.aa, 31, 3, 3)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        model.attach_feature_bank({self.sequence: torch.randn(17, 64)})
        c = model.sample_latents(self.aa, 31, 3, 3)
        self.assertFalse(torch.equal(a[0], c[0]))
        self.assertFalse(torch.equal(a[1], c[1]))


if __name__ == "__main__":
    unittest.main()
