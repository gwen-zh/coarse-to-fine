import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from coarse_coverage_repair import integrate, replay_loss, update
from coarse_few_target_flow import coarse_flow_loss
from coarse_full_cohort_flow import batch_rng, identity_record, sample_training_batch, update as ordinary_update
from joint_sequence_predictor import encode_aa
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint
from test_coarse_full_cohort_flow import case, groups, model


class CoverageRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p = dict(cycle_updates=8, lr=3e-4, min_lr=3e-5, gradient_clip=1.)

    def test_euler16_is_exact_original_sampler(self):
        m = model().eval(); aa = encode_aa("ACDEFGHIKLMNP")
        for seed in [17, 912]:
            torch.testing.assert_close(integrate(m.coarse_model, aa, seed, 16),
                m.sample_coarse(aa, seed, 16), atol=0, rtol=0)

    def test_midpoint_has_exact_declared_call_count_and_second_order_accuracy(self):
        class Field:
            block_size = 4; coarse_dim = 3
            def __init__(self): self.calls = 0
            def coarse_velocity(self, aa, state, time):
                self.calls += 1; return state
            def denormalize_coarse(self, state): return state
        aa = torch.zeros(13, dtype=torch.long)
        f = Field(); mid = integrate(f, aa, 12, 8, "midpoint"); self.assertEqual(f.calls, 16)
        f = Field(); eu = integrate(f, aa, 12, 16, "euler"); self.assertEqual(f.calls, 16)
        initial = torch.randn((4, 3), generator=torch.Generator().manual_seed(12))
        exact = initial * torch.exp(torch.tensor(1.))
        self.assertLess(float((mid - exact).square().mean()), float((eu - exact).square().mean()))

    def test_parent_replay_starts_at_zero_and_has_no_teacher_gradient(self):
        m = model().eval(); teacher = copy.deepcopy(m.coarse_model).eval().requires_grad_(False)
        aa = encode_aa("ACDEFGHIKLMNP"); target = torch.ones(4, 3)
        loss, _ = replay_loss(m.coarse_model, teacher, aa, target, torch.Generator().manual_seed(9), "PDB")
        self.assertEqual(float(loss.detach()), 0.); loss.backward()
        self.assertTrue(all(v.grad is None for v in teacher.parameters()))
        self.assertTrue(all(v.grad is None or torch.count_nonzero(v.grad) == 0 for v in m.parameters()))

    def test_md_loss_and_draws_equal_existing_actual_frame_flow(self):
        m = model().eval(); teacher = copy.deepcopy(m.coarse_model).eval().requires_grad_(False)
        aa = encode_aa("ACDEFGHIKLMNP"); target = torch.ones(4, 3)
        a, x = replay_loss(m.coarse_model, teacher, aa, target, torch.Generator().manual_seed(14), "MD")
        b, y = coarse_flow_loss(m, aa, target, torch.Generator().manual_seed(14))
        torch.testing.assert_close(a, b, atol=0, rtol=0); self.assertEqual(x["draw_sha256"], y["draw_sha256"])

    def test_control_and_replay_use_same_cases_draws_lr_but_different_pdb_targets(self):
        a = model(); b = copy.deepcopy(a); teacher = copy.deepcopy(a.coarse_model).eval().requires_grad_(False)
        ta = copy.deepcopy(teacher.state_dict()); fine = copy.deepcopy(a.fine_model.state_dict())
        oa = torch.optim.AdamW(a.trainable_parameters(), lr=3e-4)
        ob = torch.optim.AdamW(b.trainable_parameters(), lr=3e-4)
        batch = sample_training_batch(groups(), batch_rng(2026))
        r = update(a, teacher, oa, batch, 2026, 0, self.p)
        c = ordinary_update(b, ob, batch, 2026, 0, self.p)
        self.assertEqual(identity_record(r), identity_record(c))
        self.assertEqual(r["PDB_replay_loss"], 0.); self.assertGreater(c["PDB_loss"], 0.)
        for k, v in a.fine_model.state_dict().items(): torch.testing.assert_close(v, fine[k], atol=0, rtol=0)
        for k, v in teacher.state_dict().items(): torch.testing.assert_close(v, ta[k], atol=0, rtol=0)
        self.assertFalse(a.fine_model.training)

    def test_update_rejects_any_development_case_before_changing_parameters(self):
        m = model(); teacher = copy.deepcopy(m.coarse_model).eval().requires_grad_(False)
        original = copy.deepcopy(m.state_dict()); opt = torch.optim.AdamW(m.trainable_parameters(), lr=3e-4)
        batch = sample_training_batch(groups(), batch_rng(2026)); batch[0] = case("dev", "dev", 1., "validation")
        with self.assertRaises(ValueError): update(m, teacher, opt, batch, 2026, 0, self.p)
        for k, v in m.state_dict().items(): torch.testing.assert_close(v, original[k], atol=0, rtol=0)

    def test_replay_resume_reproduces_optimizer_rng_and_uncheckpointed_trace_tail(self):
        a = model(); teacher = copy.deepcopy(a.coarse_model).eval().requires_grad_(False)
        opt = torch.optim.AdamW(a.trainable_parameters(), lr=3e-4)
        rng = batch_rng(2026); rows = []
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); writer = CompactTrainingTrace(4)
            for step in range(4):
                rows.append(update(a, teacher, opt, sample_training_batch(groups(), rng), 2026, step, self.p))
                if step == 1: save_checkpoint(folder / "last.pt", a, opt, rng, updates=2)
                if step == 2: writer(folder / "trace.json", rows)
            b = model(); replay_rng = batch_rng(0); replay_opt = torch.optim.AdamW(b.trainable_parameters(), lr=3e-4)
            restore_checkpoint(folder / "last.pt", b, replay_opt, replay_rng, torch.device("cpu"))
            restore_trace_prefix(folder, 2)
            replayed = [json.loads(s) for s in (folder / "trace.jsonl").read_text().splitlines()]
            for step in range(2, 4):
                replayed.append(update(b, teacher, replay_opt, sample_training_batch(groups(), replay_rng), 2026, step, self.p))
            self.assertEqual(rows, replayed)
            for k, v in a.state_dict().items(): torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)

    def test_rehearsal_restores_global_fastpath_setting(self):
        m = model().eval(); teacher = copy.deepcopy(m.coarse_model).eval().requires_grad_(False)
        previous = torch.backends.mha.get_fastpath_enabled()
        replay_loss(m.coarse_model, teacher, encode_aa("ACDEFGHIKLMNP"), torch.ones(4, 3), torch.Generator().manual_seed(9), "PDB")
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), previous)


if __name__ == "__main__": unittest.main()
