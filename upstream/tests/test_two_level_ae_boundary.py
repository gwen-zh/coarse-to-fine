import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from two_level_autoencoder import TwoLevelAutoencoder, training_loss
from two_level_ae_boundary import NeighborFineAutoencoder, boundary_terms, boundary_training_loss
from hierarchical_latent_contract import pool_ca, remove_block_translation


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.base = TwoLevelAutoencoder().eval()
        self.model = NeighborFineAutoencoder(self.base).eval()
        self.x = torch.randn(25, 3, 3)
        self.weights = dict(coarse=1., fine=1., full=.1, bond=.1, angle=.05, clash=.01, latent=1e-4)

    def test_initial_function_and_common_parameters_preserved(self):
        torch.testing.assert_close(self.base(self.x)['backbone'], self.model(self.x)['backbone'], atol=2e-5, rtol=2e-6)
        self.assertEqual(sum(p.numel() for p in self.model.parameters())-sum(p.numel() for p in self.base.parameters()), 5120)
        for name in ['coarse_encoder', 'coarse_decoder', 'fine_encoder']:
            for a, b in zip(getattr(self.base, name).parameters(), getattr(self.model, name).parameters()):
                self.assertTrue(torch.equal(a, b))
        self.assertFalse(self.model.fine_decoder[0].weight[:, self.model.original_fine_inputs:].any())

    def test_construction_does_not_change_augmentation_rng(self):
        before = torch.get_rng_state().clone()
        NeighborFineAutoencoder(self.base)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_decoder_influence_bounded_to_immediate_neighbors_and_centroids_fixed(self):
        with torch.no_grad():
            self.model.fine_decoder[0].weight[:, self.model.original_fine_inputs:] = .03
        c, f = self.model.encode(self.x)
        changed = f.clone(); changed[12:16] += .5
        a, b = self.model.decode(c, f, len(f)), self.model.decode(c, changed, len(f))
        outside = torch.ones(len(f), dtype=torch.bool); outside[8:20] = False
        torch.testing.assert_close(a['backbone'][outside], b['backbone'][outside], atol=0, rtol=0)
        self.assertGreater(float((a['backbone'][8:12]-b['backbone'][8:12]).abs().max()), 1e-6)
        self.assertGreater(float((a['backbone'][16:20]-b['backbone'][16:20]).abs().max()), 1e-6)
        torch.testing.assert_close(pool_ca(a['backbone']), pool_ca(b['backbone']), atol=2e-5, rtol=1e-6)

    def test_fine_encoder_still_cannot_read_block_translation(self):
        changed = self.x.clone(); changed[4:8] += 7.
        _, a = self.model.encode(self.x); _, b = self.model.encode(changed)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_boundary_loss_is_zero_for_target_and_ignores_non_boundary_atoms(self):
        for value in boundary_terms(self.x, self.x): self.assertEqual(float(value), 0.)
        changed = self.x.clone(); changed[1, 0] += 5.
        for value in boundary_terms(changed, self.x): self.assertEqual(float(value), 0.)
        changed = self.x.clone(); changed[4, 0] += 5.
        self.assertGreater(float(sum(boundary_terms(changed, self.x))), 0.)

    def test_boundary_gradients_only_touch_declared_boundary_atoms(self):
        pred = (self.x + .1*torch.randn_like(self.x)).requires_grad_()
        sum(boundary_terms(pred, self.x)).backward()
        mask = torch.zeros_like(pred, dtype=torch.bool)
        for i in range(3, len(pred)-1, 4):
            mask[i, 1:] = True; mask[i+1, :2] = True
        self.assertEqual(float(pred.grad[~mask].abs().max()), 0.)
        self.assertGreater(float(pred.grad[mask].abs().sum()), 0.)

    def test_short_chains_no_missing_neighbor_or_empty_mean(self):
        for n in [1, 4, 5, 9]:
            x = self.x[:n]; output = self.model(x)
            torch.testing.assert_close(pool_ca(output['backbone']), output['coarse'], atol=2e-5, rtol=1e-6)
            loss, terms = boundary_training_loss(output, x, self.weights, True)
            self.assertTrue(torch.isfinite(loss)); loss.backward()
            if n <= 4: self.assertEqual(float(terms['boundary_bond']), 0.)

    def test_loss_adds_only_frozen_boundary_terms(self):
        output = self.model(self.x)
        base, _ = training_loss(output, self.x, self.weights)
        disabled, _ = boundary_training_loss(output, self.x, self.weights, False)
        enabled, terms = boundary_training_loss(output, self.x, self.weights, True)
        torch.testing.assert_close(base, disabled, atol=0, rtol=0)
        torch.testing.assert_close(enabled, base+.1*terms['boundary_bond']+.05*terms['boundary_angle'], atol=0, rtol=0)

    def test_new_context_columns_receive_real_loss_gradient(self):
        loss, _ = boundary_training_loss(self.model(self.x), self.x, self.weights, True)
        loss.backward()
        gradient = self.model.fine_decoder[0].weight.grad[:, self.model.original_fine_inputs:]
        self.assertTrue(torch.isfinite(gradient).all()); self.assertGreater(float(gradient.norm()), 0.)
        for module in [self.model.coarse_encoder, self.model.coarse_decoder, self.model.fine_encoder, self.model.fine_decoder]:
            self.assertGreater(sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None), 0.)

    def test_coarse_loss_still_has_zero_fine_gradient(self):
        pool_ca(self.model(self.x)['backbone']).square().mean().backward()
        for module in [self.model.fine_encoder, self.model.fine_decoder]:
            self.assertLess(max([float(p.grad.abs().max()) for p in module.parameters() if p.grad is not None], default=0), 1e-6)


if __name__ == '__main__': unittest.main()
