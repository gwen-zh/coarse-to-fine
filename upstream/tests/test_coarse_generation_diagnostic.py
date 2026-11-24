from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_generation_diagnostic import block_counts, pool_backbones, scalar_features, weighted_rmsd_matrix
from structural_metrics import aligned_rmsd


class CoarseDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_partial_block_is_counted_and_matches_atom_pooling(self):
        bb = self.rng.normal(size=(2, 21, 3, 3))
        self.assertEqual(block_counts(21).tolist(), [4., 4., 4., 4., 4., 1.])
        centers = pool_backbones(bb)
        np.testing.assert_allclose(centers[:, -1], bb[:, -1, 1])
        np.testing.assert_allclose(centers[:, 0], bb[:, :4, 1].mean(1))
        for bad in [0, -1, True, 3.5]:
            with self.assertRaises(ValueError): block_counts(bad)

    def test_weighted_alignment_equals_explicit_repeated_centers(self):
        a = self.rng.normal(size=(3, 6, 3)); b = self.rng.normal(size=(2, 6, 3))
        counts = block_counts(21).astype(int)
        matrix = weighted_rmsd_matrix(a, b, 21)
        expected = [[aligned_rmsd(np.repeat(x, counts, axis=0), np.repeat(y, counts, axis=0)) for y in b] for x in a]
        np.testing.assert_allclose(matrix, expected, atol=1e-12)

    def test_bounds_survive_rigid_motion_but_not_a_reflection(self):
        a = self.rng.normal(size=(2, 6, 3))
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        distance = weighted_rmsd_matrix(a, a @ rotation + 12, 21)
        np.testing.assert_allclose(np.diag(distance), 0., atol=1e-6)
        self.assertGreater(weighted_rmsd_matrix(a[:1], a[:1] * [-1, 1, 1], 21)[0, 0], .1)
        f = scalar_features(a, 21); g = scalar_features(a @ rotation + 12, 21)
        for key in f: np.testing.assert_allclose(f[key], g[key], atol=1e-12)

    def test_jensen_bound_holds_for_different_local_residuals(self):
        # Include a partial block and many independent within-block details.
        first = self.rng.normal(size=(5, 21, 3, 3))
        second = self.rng.normal(size=(4, 21, 3, 3))
        lower = weighted_rmsd_matrix(pool_backbones(first), pool_backbones(second), 21)
        actual = np.array([[aligned_rmsd(a[:, 1], b[:, 1]) for b in second] for a in first])
        self.assertTrue((lower <= actual + 1e-12).all())
        self.assertTrue((lower.min(1) <= actual.min(1) + 1e-12).all())

    def test_adjacency_bound_accounts_for_partial_block(self):
        ca = np.zeros((21, 3)); ca[:, 0] = 3.8 * np.arange(21)
        bb = np.repeat(ca[None, :, None, :], 3, axis=2)
        centers = pool_backbones(bb)
        f = scalar_features(centers, 21)
        self.assertAlmostEqual(f["adjacent_span_bound_A"][-1], 4.1 * 2.5)
        self.assertFalse(f["conditional_span_flag"].any())
        self.assertTrue(scalar_features(centers * 2, 21)["conditional_span_flag"].all())


if __name__ == "__main__":
    unittest.main()
