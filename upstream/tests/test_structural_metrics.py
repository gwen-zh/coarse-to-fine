from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from structural_metrics import aligned_rmsd, summarize_backbone, ensemble_summary


class MetricTests(unittest.TestCase):
    def test_alignment_removes_rotation_but_not_reflection(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], float)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertLess(aligned_rmsd(x @ rotation + 10, x), 1e-10)
        self.assertGreater(aligned_rmsd(x * [-1, 1, 1], x), 0.1)

    def test_cn_error_detected_without_ca_error(self):
        xyz = np.full((4, 37, 3), np.nan)
        ca = np.column_stack([np.arange(4) * 3.8, np.zeros((4, 2))])
        xyz[:, 1] = ca
        xyz[:, 0] = ca - [1.2, 0, 0]
        xyz[:, 2] = ca + [1.3, 0, 0]
        report = summarize_backbone(xyz, [ca])
        self.assertTrue(report["upstream_ca_clash_valid"])
        self.assertEqual(report["backbone_cn_outliers"], 0)
        self.assertLess(report["closest_available_reference_ca_rmsd"], 1e-10)
        xyz[1, 0] = xyz[0, 2]
        report = summarize_backbone(xyz)
        self.assertTrue(report["upstream_ca_clash_valid"])
        self.assertEqual(report["backbone_cn_outliers"], 1)

    def test_ensemble_reports_coverage_and_diversity(self):
        x = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], float)
        result = ensemble_summary([x, 2 * x], [x, 2 * x])
        self.assertGreater(result["mean_pairwise_ca_rmsd"], 0)
        self.assertTrue(all(r < 1e-10 for r in result["reference_best_ca_rmsd"]))


if __name__ == "__main__":
    unittest.main()
