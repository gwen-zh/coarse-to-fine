import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_few_target_flow import FixedFineCoarseFlow, learning_rate
from coarse_full_cohort_flow import batch_rng, sample_training_batch, update
from coarse_full_cohort_evaluation import GEOMETRY_KEYS, SUMMARY_KEYS, draw_fine_backbone, summarize_rows
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint


def case(name, protein, offset, split="train"):
    return dict(id=name, protein=protein, sequence="ACDEFGHIKLMNP", split=split,
        target=torch.zeros(13, 3, 3), coarse=torch.full((4, 3), float(offset)))


def groups():
    return dict(PDB_train=[case("p1", "p1", .1), case("p2", "p2", .3)],
        train={"md1": [case("md1:f1", "md1", 1.), case("md1:f2", "md1", 2.)],
               "md2": [case("md2:f1", "md2", 3.)]})


def model():
    torch.manual_seed(52)
    args = (torch.zeros(3), torch.ones(3), torch.zeros(8), torch.ones(8))
    a = HierarchicalEnsembleFlow(*args, width=32, layers=1, coarse_dim=3)
    b = HierarchicalEnsembleFlow(*args, width=32, layers=1, coarse_dim=3)
    return FixedFineCoarseFlow(a, b)


class FullCoarseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p = dict(cycle_updates=4, lr=3e-4, min_lr=3e-5, gradient_clip=1.)

    def test_real_frames_are_selected_without_averaging_same_sequence_targets(self):
        g = groups(); rng = batch_rng(2026); seen = set()
        originals = {c["id"]: c["coarse"].clone() for cases in g["train"].values() for c in cases}
        for _ in range(80):
            batch = sample_training_batch(g, rng)
            self.assertTrue(all(c["id"].startswith("p") for c in batch[:3]))
            self.assertTrue(batch[-1]["id"].startswith("md")); seen.add(batch[-1]["id"])
            torch.testing.assert_close(batch[-1]["coarse"], originals[batch[-1]["id"]], atol=0, rtol=0)
        self.assertEqual(seen, set(originals))
        self.assertTrue(torch.equal(originals["md1:f1"], torch.ones(4, 3)))
        self.assertTrue(torch.equal(originals["md1:f2"], torch.full((4, 3), 2.)))

    def test_checkpoint_resume_and_trace_truncation_replay_actual_update(self):
        g = groups(); a = model(); rng = batch_rng(2026)
        opt = torch.optim.AdamW(a.trainable_parameters(), lr=3e-4, weight_decay=1e-4)
        rows = []
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); writer = CompactTrainingTrace(4)
            for step in range(4):
                rows.append(update(a, opt, sample_training_batch(g, rng), 2026, step, self.p))
                if step == 1: save_checkpoint(folder / "last.pt", a, opt, rng, updates=2)
                if step == 2: writer(folder / "trace.json", rows)  # one uncheckpointed row
            b = model(); resumed_rng = batch_rng(0)
            resumed_opt = torch.optim.AdamW(b.trainable_parameters(), lr=3e-4, weight_decay=1e-4)
            saved = restore_checkpoint(folder / "last.pt", b, resumed_opt, resumed_rng, torch.device("cpu"))
            self.assertEqual(saved["updates"], 2); restore_trace_prefix(folder, 2)
            replayed = [json.loads(s) for s in (folder / "trace.jsonl").read_text().splitlines()]
            resumed_writer = CompactTrainingTrace(4, already_written=2)
            for step in range(2, 4): replayed.append(update(b, resumed_opt, sample_training_batch(g, resumed_rng), 2026, step, self.p))
            resumed_writer(folder / "trace.json", replayed)
            self.assertEqual(rows, replayed)
            for key, value in a.state_dict().items(): torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(len(json.loads((folder / "trace.json").read_text())), 4)

    def test_update_keeps_fine_fixed_and_rejects_development(self):
        a = model(); fine = copy.deepcopy(a.fine_model.state_dict())
        opt = torch.optim.AdamW(a.trainable_parameters(), lr=3e-4)
        batch = sample_training_batch(groups(), batch_rng(2026))
        row = update(a, opt, batch, 2026, 0, self.p)
        self.assertEqual([v["domain"] for v in row["batch"]], ["PDB", "PDB", "PDB", "MD"])
        for key, value in a.fine_model.state_dict().items(): torch.testing.assert_close(value, fine[key], atol=0, rtol=0)
        self.assertFalse(a.fine_model.training)
        batch[-1] = case("dev", "dev", 4., split="validation")
        with self.assertRaises(ValueError): update(a, opt, batch, 2026, 1, self.p)

    def test_lr_cycle_is_budget_independent(self):
        self.assertAlmostEqual(learning_rate(0, 32768), 3e-4)
        self.assertAlmostEqual(learning_rate(32767, 32768), 3e-5)
        self.assertEqual(learning_rate(32768, 32768), learning_rate(0, 32768))
        self.assertEqual(learning_rate(131071, 32768), learning_rate(32767, 32768))

    def test_invalid_fine_is_retained_without_decoder_or_fallback(self):
        class InvalidFine:
            fine_dim = 8
            def sample_fine(self, aa, coarse, seed, steps):
                return torch.full((len(aa), 8), float("nan"))
        class NoDecoder:
            def decode(self, *args): raise AssertionError("Invalid fine must not reach decoder")
        bb, failure = draw_fine_backbone(InvalidFine(), NoDecoder(), torch.zeros(13, dtype=torch.long), torch.ones(4, 3), 42, 16)
        self.assertEqual(failure["code"], "fine_nonfinite")
        self.assertFalse(torch.from_numpy(bb).isfinite().any())

    def test_one_failure_prevents_reporting_success_only_full_metrics(self):
        coarse = {k: 1. for k in ["mean_sample_nearest_reference_center_rmsd_A", "mean_reference_nearest_sample_center_rmsd_A",
            "generated_conditional_span_flag_rate", "js_pairwise_center_distance_nats", "spread_ratio"]}
        good = {k: 1. for k in SUMMARY_KEYS}
        good.update(metrics_available=True, failure_count=0, geometry={k: 1. for k in GEOMETRY_KEYS},
            mean_pairwise_ca_rmsd_A=1., reference_mean_pairwise_ca_rmsd_A=1., coarse_metrics=coarse)
        bad = copy.deepcopy(good); bad.update(metrics_available=False, failure_count=1, geometry=None)
        summary = summarize_rows([good, bad])
        self.assertFalse(summary["metrics_available"])
        self.assertEqual(summary["failed_samples"], 1)
        self.assertIsNone(summary["mean_sample_nearest_reference_ca_rmsd_A"])
        self.assertIsNone(summary["cn_rate"])
        self.assertEqual(summary["coarse_mean_sample_nearest_reference_center_rmsd_A"], 1.)


if __name__ == "__main__":
    unittest.main()
