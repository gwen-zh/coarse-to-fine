from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from multisequence_flow_data import balanced_batch, fit_train_normalization, slot_seed, physical_histogram_edges


def example(name, length=12, split="train", value=0.):
    return dict(id=name, sequence="A" * length, target=torch.zeros(length, 3, 3),
                coarse=torch.full(((length+3)//4, 3), value),
                fine_continuous=torch.full((length, 8), value), split=split)


class MultiFlowDataTests(unittest.TestCase):
    def test_replay_only_replaces_the_fourth_slot(self):
        groups = {"a": [example("a1"), example("a2")], "b": [example("b1", 16)]}
        pdb = [example("pdb")]
        a = balanced_batch(groups, pdb, np.random.default_rng(17), False)
        b = balanced_batch(groups, pdb, np.random.default_rng(17), True)
        self.assertEqual([x["id"] for x in a[:3]], [x["id"] for x in b[:3]])
        self.assertEqual(b[-1]["id"], "pdb")

    def test_development_cannot_fit_normalization_or_train(self):
        bad = {"bad": [example("val", split="validation")]}
        with self.assertRaises(ValueError):
            fit_train_normalization(bad)
        with self.assertRaises(ValueError):
            balanced_batch(bad, [example("pdb")], np.random.default_rng(1), False)

    def test_normalization_weights_proteins_equally(self):
        stats = fit_train_normalization({"a": [example("a", value=0.)] * 9,
                                         "b": [example("b", value=2.)]})
        np.testing.assert_allclose(stats["coarse_mean"], [1., 1., 1.])
        np.testing.assert_allclose(stats["coarse_std"], [1., 1., 1.])

    def test_noise_seed_is_independent_of_variable_shape_other_slots(self):
        self.assertEqual(slot_seed(2026, 17, 0), slot_seed(2026, 17, 0))
        self.assertNotEqual(slot_seed(2026, 17, 0), slot_seed(2026, 17, 1))
        d, r = physical_histogram_edges(90)
        self.assertEqual(d.shape[0], len(np.triu_indices(90, 4)[0]))
        self.assertTrue(np.all(np.diff(d[0]) > 0))
        self.assertTrue(np.isneginf(r[0, 0]) and np.isposinf(r[0, -1]))


if __name__ == "__main__":
    unittest.main()
