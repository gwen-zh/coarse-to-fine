import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from regularized_mixed_generator import MetricCodeHead, RegularizedMixedGenerator, fine_training_loss
from hierarchical_pipeline_io import save_checkpoint, restore_checkpoint
import numpy as np
import tempfile


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.codes = torch.tanh(torch.randn(8, 8))
        self.h = torch.randn(13, 12)
        self.c = torch.tanh(torch.randn(4, 16))
        self.f = torch.randint(8, (13,))

    def model(self, head='metric'):
        return RegularizedMixedGenerator(torch.zeros(16), torch.ones(16), condition_dim=12,
                                          width=32, coarse_layers=1, fine_layers=1,
                                          fine_size=8, fine_head=head, codes=self.codes)

    def test_metric_head_respects_code_permutation(self):
        a = MetricCodeHead(32, self.codes)
        b = copy.deepcopy(a)
        permutation = torch.tensor([3, 1, 6, 2, 5, 4, 0, 7])
        b.codes.copy_(a.codes[permutation])
        h = torch.randn(13, 32)
        torch.testing.assert_close(b(h), a(h)[:, permutation])

    def test_metric_head_prefers_near_code(self):
        head = MetricCodeHead(32, torch.stack([torch.zeros(8), torch.ones(8)]))
        torch.nn.init.zeros_(head.point[1].weight)
        torch.nn.init.zeros_(head.point[1].bias)
        logits = head(torch.randn(13, 32))
        self.assertTrue((logits[:, 0] > logits[:, 1]).all())
        self.assertFalse(head.codes.requires_grad)

    def test_fine_arms_share_common_initial_parameters(self):
        torch.manual_seed(25)
        a = self.model('ce')
        torch.manual_seed(25)
        b = self.model('metric')
        for name, value in a.state_dict().items():
            if not name.startswith('fine.head.'):
                self.assertTrue(torch.equal(value, b.state_dict()[name]), name)

    def test_training_loss_reaches_sequence_coarse_and_metric_head(self):
        model = self.model().train()
        before = model.fine.head.codes.clone()
        loss, _ = fine_training_loss(model, self.h, self.c, self.f, rate=1.)
        loss.backward()
        for parameter in [model.fine.condition[1].weight, model.fine.coarse_projection[0].weight,
                          model.fine.head.point[1].weight]:
            self.assertGreater(float(parameter.grad.norm()), 0.)
        self.assertTrue(torch.equal(before, model.fine.head.codes))

    def test_evaluation_sampling_is_seeded_without_structure(self):
        for arm in ['ce', 'metric']:
            model = self.model(arm).eval()
            a = model.sample(self.h, 17, 3, 3)
            b = model.sample(self.h, 17, 3, 3)
            for x, y in zip(a, b):
                self.assertTrue(torch.equal(x, y))
            self.assertEqual(a[0].shape, (4, 16))
            self.assertEqual(a[1].shape, (13,))
            self.assertTrue((a[1] < 8).all())

    def test_dropout_disabled_at_evaluation(self):
        model = self.model()
        self.assertTrue(any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()))
        model.eval()
        self.assertTrue(torch.equal(model.fine(self.h, self.c, self.f, .5),
                                    model.fine(self.h, self.c, self.f, .5)))

    def test_checkpoint_restores_next_dropout_mask_noise_and_adam_update(self):
        model = self.model().train()
        optimizer = torch.optim.AdamW(model.fine.parameters(), lr=3e-4)
        rng = np.random.default_rng(27)
        def step(m, opt):
            opt.zero_grad(set_to_none=True)
            loss, _ = fine_training_loss(m, self.h, self.c, self.f)
            loss.backward()
            opt.step()
            return float(loss.detach())
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_checkpoint(path, model, optimizer, rng)
            expected = step(model, optimizer)
            expected_state = {k: v.clone() for k, v in model.state_dict().items()}
            expected_adam = copy.deepcopy(optimizer.state_dict())
            expected_rng = torch.get_rng_state().clone()
            restored = self.model().train()
            opt2 = torch.optim.AdamW(restored.fine.parameters(), lr=3e-4)
            rng2 = np.random.default_rng(88)
            restore_checkpoint(path, restored, opt2, rng2, torch.device('cpu'))
            self.assertEqual(expected, step(restored, opt2))
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for name, value in expected_state.items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)
            for key, values in expected_adam['state'].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, opt2.state_dict()['state'][key][name]))


if __name__ == '__main__':
    unittest.main()
