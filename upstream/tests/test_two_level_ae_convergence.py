import copy
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from continue_two_level_autoencoder import boundary_quality, initialize, restore, snapshot, update, verify_parent_model


class Tests(unittest.TestCase):
    def test_resume_mid_epoch_is_identical_including_adam_and_randomness(self):
        p = dict(threads=1, lr=3e-4)
        weights = dict(coarse=1., fine=1., full=.1, bond=.1, angle=.05, clash=.01, latent=1e-4)
        generator = torch.Generator().manual_seed(99)
        train = [dict(bb=torch.randn(9, 3, 3, generator=generator)) for _ in range(3)]
        state = initialize(17, p)
        for _ in range(4):
            update(state, train, weights)
        saved = snapshot(state)
        expected_trace = [update(state, train, weights) for _ in range(5)]
        expected = snapshot(state)
        resumed = initialize(800, p)
        restore(resumed, saved)
        actual_trace = [update(resumed, train, weights) for _ in range(5)]
        actual = snapshot(resumed)
        self.assertEqual(expected_trace, actual_trace)
        verify_parent_model(resumed['model'], expected['model'])
        self.assertTrue(torch.equal(expected['torch_rng'], actual['torch_rng']))
        self.assertEqual(expected['numpy_rng'], actual['numpy_rng'])
        self.assertEqual(expected['order'], actual['order'])
        for key in expected['optimizer']['state']:
            for term, value in expected['optimizer']['state'][key].items():
                self.assertTrue(torch.equal(value, actual['optimizer']['state'][key][term]))

    def test_boundary_counts_partition_cn_including_ragged_last_block(self):
        x = torch.zeros(9, 3, 3)
        x[1:, 0, 0] = 1.3
        x[4, 0, 0] = 2.
        x[8, 0, 0] = .5
        x[2, 0, 0] = 2.
        result = boundary_quality(x)
        self.assertEqual(result['within_bonds'], 6)
        self.assertEqual(result['boundary_bonds'], 2)
        self.assertEqual(result['within_bad'], 1)
        self.assertEqual(result['boundary_bad'], 2)
        self.assertAlmostEqual(result['within_rate'], 1/6)
        self.assertEqual(result['boundary_rate'], 1.)
        self.assertIsNone(boundary_quality(x[:1])['boundary_rate'])

    def test_replay_rejects_a_changed_parameter(self):
        state = initialize(17, dict(threads=1, lr=3e-4))
        saved = copy.deepcopy(state['model'].state_dict())
        saved[next(iter(saved))].flatten()[0] += .01
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            verify_parent_model(state['model'], saved)


if __name__ == '__main__':
    unittest.main()
