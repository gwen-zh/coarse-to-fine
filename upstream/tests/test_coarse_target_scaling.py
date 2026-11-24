from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_target_scaling import initial_metric_roundoff, plan, round_learning_rate, select_nested_ids, training_event


class CoarseScalingTests(unittest.TestCase):
    def test_common_targets_get_identical_counter_and_lr_sequences(self):
        sequences = {}
        for arm in ["eight", "thirtytwo"]:
            p = plan(arm); rows = [[] for _ in range(p["targets"])]
            for step in range(p["updates"]):
                target, counters = training_event(step, p["targets"])
                rows[target].append((counters, round_learning_rate(step, p["targets"])))
            for row in rows:
                self.assertEqual(len(row), 256)
                self.assertEqual([c for counters, _ in row for c in counters], list(range(1024)))
                self.assertAlmostEqual(row[0][1], 3e-4)
                self.assertAlmostEqual(row[-1][1], 3e-5)
            sequences[arm] = rows
        self.assertEqual(sequences["eight"], sequences["thirtytwo"][:8])

    def test_lr_is_constant_inside_round_and_stops_at_budget(self):
        for n in [8, 32]:
            for round_index in [0, 63, 127, 255]:
                rates = [round_learning_rate(round_index * n + i, n) for i in range(n)]
                self.assertEqual(len(set(rates)), 1)
            with self.assertRaises(ValueError): round_learning_rate(n * 256, n)

    def test_selection_is_order_independent_and_keeps_original_eight(self):
        ids = [f"PDB:target{i:03d}" for i in range(90)]
        first = ids[10:18]
        a = select_nested_ids(ids, first); b = select_nested_ids(reversed(ids), first)
        self.assertEqual(a, b); self.assertEqual(a[:8], first)
        self.assertEqual(len(a), 32); self.assertEqual(len(set(a)), 32)
        with self.assertRaises(ValueError): select_nested_ids(ids + [ids[0]], first)
        with self.assertRaises(ValueError): select_nested_ids(ids[:20], first)

    def test_initial_parity_accepts_float_roundoff_but_rejects_changed_records(self):
        a = [{"id": "caseA", "nfe": 16, "metrics": {"ca": 2., "failure": False}}]
        b = [{"id": "caseA", "nfe": 16, "metrics": {"ca": 2.000001, "failure": False}}]
        self.assertLess(initial_metric_roundoff(a, b), 2e-6)
        b[0]["metrics"]["ca"] = 2.01
        with self.assertRaises(ValueError): initial_metric_roundoff(a, b)
        b[0]["metrics"]["ca"] = 2.
        b[0]["nfe"] = 17
        with self.assertRaises(ValueError): initial_metric_roundoff(a, b)


if __name__ == "__main__":
    unittest.main()
