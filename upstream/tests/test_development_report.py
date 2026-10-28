from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from summarize_development import summarize


class ReportTests(unittest.TestCase):
    def test_paired_changes_include_initial_sampling_compute(self):
        def target(name, invalid, calls, diversity=2):
            return dict(target=name, samples=2, proxy_invalid_samples=invalid,
                        upstream_clash_invalid_samples=invalid, cn_outlier_samples=0,
                        network_calls=calls, mean_pairwise_ca_rmsd=diversity)

        baseline = dict(targets=[target("a", 2, 14), target("b", 0, 14)])
        feedback = dict(targets=[])
        for policy in ["geometry", "random", "global"]:
            for name in ["a", "b"]:
                feedback["targets"].append(dict(policy=policy, **target(name, 0, 28, 1)))
        rows = {r["policy"]: r for r in summarize(baseline, feedback)}
        self.assertEqual(rows["baseline"]["total_network_calls"], 28)
        self.assertEqual(rows["geometry"]["total_network_calls"], 84)
        self.assertEqual(rows["geometry"]["ca_validity_gain_mean"], 0.5)
        self.assertEqual(rows["geometry"]["median_diversity_ratio"], 0.5)
        direct = dict(targets=[target("a", 1, 42), target("b", 0, 42)])
        extended = {r["policy"]: r for r in summarize(baseline, feedback, direct)}
        self.assertEqual(extended["direct19"]["total_network_calls"], 84)
        self.assertEqual(extended["direct19"]["additional_network_calls"], 0)
        feedback["targets"].pop()
        with self.assertRaises(ValueError):
            summarize(baseline, feedback)


if __name__ == "__main__":
    unittest.main()
