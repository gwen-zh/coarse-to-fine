"""CPU checks; no ESM model downloads or GPU required."""
import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hierarchical_ConfGen"))
from slm.geometry_feedback import assess_ca, repair_mask, geometry_feedback, assess_backbone, backbone_improves


def line(n=6):
    return np.column_stack((np.arange(n) * 3.8, np.zeros((n, 2))))


def backbone(n=6):
    ca = line(n)
    return np.stack([ca - [1.2, 0, 0], ca, ca + [1.3, 0, 0]], axis=1)


class GeometryTests(unittest.TestCase):
    def test_backbone_detects_cn_error_and_rejects_tradeoff(self):
        good = backbone()
        broken = good.copy()
        broken[1, 0] = broken[0, 2]
        current = assess_backbone(broken)
        self.assertEqual(current.peptide_bonds, 1)
        self.assertEqual(assess_ca(broken[:, 1]).bad.sum(), 0)
        self.assertTrue(backbone_improves(assess_backbone(good), current))
        clash = good.copy()
        clash[3, 1] = clash[0, 1]
        self.assertFalse(backbone_improves(assess_backbone(clash), current))

    def test_backbone_missing_atom_has_finite_severity(self):
        xyz = backbone()
        xyz[1, 0] = np.nan
        report = assess_backbone(xyz)
        self.assertEqual(report.peptide_bonds, 1)
        self.assertTrue(np.isfinite(report.peptide_severity))

    def test_backbone_feedback_uses_full_coordinates(self):
        good, broken = backbone(), backbone()
        broken[1, 0] = broken[0, 2]

        class Model:
            coarse_mask_index = 32

            def ddpm_sample(self, input_prior, **kwargs):
                coarse = input_prior.clone()
                coarse[coarse == 32] = 1
                return torch.ones_like(coarse), coarse

        x = torch.zeros(1, 8, dtype=torch.long)
        _, _, trace = geometry_feedback(Model(), x, x, x, torch.tensor([6]),
            lambda tokens, b: good if tokens[0] else broken, rounds=1, radius=0,
            assessment_fn=assess_backbone, acceptance_fn=backbone_improves)
        self.assertTrue(trace[1]["accepted"])
        self.assertEqual(trace[1]["peptide_bonds"], 0)

    def test_clean_and_rigid_transform(self):
        ca = line()
        self.assertEqual(assess_ca(ca).score, (0, 0, 0))
        rotated = ca[:, [1, 0, 2]] + [13, -3, 6]
        self.assertEqual(assess_ca(rotated).score, (0, 0, 0))

    def test_break_missing_and_overlap(self):
        ca = line()
        ca[3] = ca[0]
        report = assess_ca(ca)
        self.assertEqual(report.clashes, 1)
        self.assertTrue(report.bad[0] and report.bad[3])
        ca[2] = np.nan
        self.assertEqual(assess_ca(ca).missing, 1)
        self.assertTrue(assess_ca(ca).bad[2])

    def test_mask_boundaries(self):
        np.testing.assert_array_equal(repair_mask([True, False, False, False], 1),
                                      [True, True, False, False])

    def test_accept_reject_and_fixed_tokens(self):
        fine = torch.zeros(1, 8, dtype=torch.long)
        coarse = torch.arange(8)[None]
        seq = fine.clone()
        original = coarse.clone()

        class Model:
            coarse_mask_index = 32

            def ddpm_sample(self, input_prior, **kwargs):
                self.prior = input_prior.clone()
                result = input_prior.clone()
                result[result == 32] = 9
                return torch.ones_like(result), result

        bad = line()
        bad[-1, 0] += 10
        model = Model()
        decoded = lambda tokens, b: line() if tokens[0] else bad
        out, new_coarse, trace = geometry_feedback(
            model, seq, fine, coarse, torch.tensor([6]), decoded, radius=0)
        self.assertTrue(trace[1]["accepted"])
        self.assertEqual(len(trace), 2)  # Early stop once no violations remain.
        self.assertTrue(torch.equal(new_coarse[:, 1:5], original[:, 1:5]))
        self.assertTrue(torch.equal(coarse, original))  # Inputs remain intact.
        self.assertTrue((out == 1).all())
        out, new_coarse, trace = geometry_feedback(
            model, seq, fine, coarse, torch.tensor([6]), lambda t, b: bad, radius=0)
        self.assertFalse(trace[1]["accepted"])
        self.assertTrue(torch.equal(out, fine))
        self.assertTrue(torch.equal(new_coarse, coarse))

    def test_random_control_same_mask_count(self):
        bad = line()
        bad[-1, 0] += 10
        counts = []

        class Model:
            coarse_mask_index = 32

            def ddpm_sample(self, input_prior, **kwargs):
                counts.append(int((input_prior == 32).sum()))
                result = input_prior.clone()
                result[result == 32] = 1
                return torch.zeros_like(result), result

        x = torch.zeros(1, 8, dtype=torch.long)
        for policy in ["geometry", "random"]:
            geometry_feedback(Model(), x, x, x, torch.tensor([6]),
                              lambda t, b: bad, rounds=1, radius=0, policy=policy)
        self.assertEqual(counts, [2, 2])

    def test_fixed_budget_runs_even_for_clean_chain(self):
        calls = []

        class Model:
            coarse_mask_index = 32

            def ddpm_sample(self, input_prior, **kwargs):
                calls.append(input_prior.clone())
                return input_prior.clone(), input_prior.clone()

        x = torch.zeros(1, 8, dtype=torch.long)
        _, _, trace = geometry_feedback(
            Model(), x, x, x, torch.tensor([6]), lambda t, b: line(),
            rounds=2, fixed_budget=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sum(t.get("network_calls", 0) for t in trace), 14)
        self.assertFalse(any(t.get("accepted", False) for t in trace))

    def test_random_mask_keeps_proposal_rng_aligned(self):
        draws = []
        bad = line()
        bad[-1, 0] += 10

        class Model:
            coarse_mask_index = 32

            def ddpm_sample(self, input_prior, **kwargs):
                draws.append(float(torch.rand(())))
                result = input_prior.clone()
                result[result == 32] = 1
                return result.clone(), result

        x = torch.zeros(1, 8, dtype=torch.long)
        for policy in ["geometry", "random", "global"]:
            torch.manual_seed(123)
            geometry_feedback(Model(), x, x, x, torch.tensor([6]), lambda t, b: bad,
                              rounds=1, policy=policy)
        self.assertEqual(draws[0], draws[1])
        self.assertEqual(draws[0], draws[2])

    def test_real_sampler_return_contract(self):
        # Execute the actual sampler with a tiny fake network to check the token
        # conventions and return contract without importing training dependencies.
        path = ROOT / "hierarchical_ConfGen/slm/models/model.py"
        module = ast.parse(path.read_text())
        helper = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                      and n.name == "add_eos_bos_tokens")
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef)
                   and any(getattr(m, "name", "") == "ddpm_sample" for m in n.body))
        method = next(m for m in cls.body if getattr(m, "name", "") == "ddpm_sample")
        namespace = dict(torch=torch, tqdm=lambda x, **kw: x)
        exec(compile(ast.Module(body=[helper, method], type_ignores=[]), str(path), "exec"), namespace)

        class Dummy:
            device = "cpu"
            noise_removal = True
            net = sigma_embedder = torch.nn.Identity()

            class noise:
                @staticmethod
                def eval():
                    pass

                def __new__(cls, t):
                    return t, t

            def _ddpm_update(self, x, *args, **kw):
                return torch.where(x == 32, 7, x)

            def _model_wrapper_inference(self, xt_var, scale, **kwargs):
                ids = xt_var.clone() if scale == 0 else torch.full_like(xt_var, 100)
                return torch.nn.functional.one_hot(ids, num_classes=4101).float(), None

        sample = namespace["ddpm_sample"]
        prior = torch.tensor([[34, 3, 32, 4, 33]])
        kw = dict(sequence_tokens=prior, input_prior=prior, lengths=torch.tensor([3]), num_steps=2)
        fine, coarse = sample(Dummy(), **kw, return_coarse=True)
        self.assertEqual(coarse.tolist(), [[34, 3, 7, 4, 33]])
        self.assertEqual(fine.tolist(), [[4098, 100, 100, 100, 4097]])
        self.assertTrue(torch.equal(sample(Dummy(), **kw), fine))


if __name__ == "__main__":
    unittest.main()
