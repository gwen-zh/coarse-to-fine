import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from hierarchical_latent_contract import pool_ca
from hierarchical_rigid_vq import HierarchicalRigidVQ
from joint_sequence_predictor import encode_aa
from mixed_hierarchical_generator import HybridTokenizer
from sequence_multiscale_ensemble import (
    MultiscaleStructureCodec, paired_flow_loss, sample_levels,
)


class SequenceMultiscaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61)
        self.aa = encode_aa("ACDEFGHIKLMNPQRST")
        self.target = torch.randn(17, 3, 3)
        self.target -= self.target[:, 1].mean(0)
        self.coarse = HybridTokenizer(HierarchicalRigidVQ(width=32, layers=1))
        self.fine = StableBlockFrameFineTokenizer("local", fine_size=16)
        self.fine.book.initialize(torch.randn(64, 8), 2)

    def make(self, kind):
        codec = MultiscaleStructureCodec(self.coarse, self.fine, kind)
        model = HierarchicalEnsembleFlow(
            torch.zeros(codec.coarse_dim), torch.ones(codec.coarse_dim),
            torch.zeros(8), torch.ones(8), width=32, layers=2,
            coarse_dim=codec.coarse_dim,
        ).eval()
        return model, codec

    def test_both_coarse_shapes_and_both_decode_outlets(self):
        for kind in ("latent16", "centers3"):
            _, codec = self.make(kind)
            c, f = codec.encode(self.target)
            self.assertEqual(c.shape, (5, codec.coarse_dim))
            for quantized in (False, True):
                bb = codec.decode(c, f, quantized)
                self.assertEqual(bb.shape, self.target.shape)
                torch.testing.assert_close(pool_ca(bb), codec.centers(c, 17), atol=2e-5, rtol=1e-5)

    def test_ragged_block_centers_have_residue_weighted_zero_mean(self):
        _, codec = self.make("centers3")
        c = codec.project_coarse(torch.randn(5, 3) + 9., 17)
        weights = torch.tensor([4., 4., 4., 4., 1.])
        torch.testing.assert_close((c * weights[:, None]).sum(0), torch.zeros(3), atol=3e-5, rtol=0)

    def test_reframing_uses_same_physical_target(self):
        _, codec = self.make("centers3")
        c, f = codec.encode(self.target)
        rotated = c[:, [1, 2, 0]]
        changed = codec.fine_target(self.target, rotated)
        expected = codec.fine_tokenizer.encode(self.target, codec.context(rotated, 17))
        torch.testing.assert_close(changed, expected)
        self.assertFalse(torch.equal(f, changed))

    def test_sampling_is_seeded_stochastic_and_fine_depends_on_coarse(self):
        model, codec = self.make("centers3")
        c, f = sample_levels(model, codec, self.aa, 8, 3, 3)
        c2, f2 = sample_levels(model, codec, self.aa, 8, 3, 3)
        torch.testing.assert_close(c, c2, atol=0, rtol=0)
        torch.testing.assert_close(f, f2, atol=0, rtol=0)
        other_c, other_f = sample_levels(model, codec, self.aa, 9, 3, 3)
        self.assertFalse(torch.equal(c, other_c))
        self.assertFalse(torch.equal(f, other_f))
        a = model.sample_fine(self.aa, c, 31, 3)
        b = model.sample_fine(self.aa, other_c, 31, 3)
        self.assertFalse(torch.equal(a, b))

    def test_loss_gradients_and_frozen_structure_for_both_context_modes(self):
        for probability in (0., 1.):
            model, codec = self.make("centers3")
            frozen = copy.deepcopy(codec.state_dict())
            c, f = codec.encode(self.target)
            loss, terms = paired_flow_loss(
                model, codec, self.aa, self.target, c, f,
                torch.Generator().manual_seed(9), probability,
            )
            loss.backward()
            self.assertEqual(terms["scheduled"], bool(probability))
            for module in (model.coarse_head, model.fine_head, model.aa_embedding):
                self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None), 0.)
            self.assertTrue(all(p.grad is None for p in codec.parameters()))
            for key, value in frozen.items():
                self.assertTrue(torch.equal(value, codec.state_dict()[key]))

    def test_context_modes_pair_the_noise_draws(self):
        model, codec = self.make("centers3")
        c, f = codec.encode(self.target)
        records = [paired_flow_loss(model, codec, self.aa, self.target, c, f,
                   torch.Generator().manual_seed(11), p)[1] for p in (0., 1.)]
        self.assertEqual(records[0]["draw_sha256"], records[1]["draw_sha256"])


if __name__ == "__main__":
    unittest.main()
