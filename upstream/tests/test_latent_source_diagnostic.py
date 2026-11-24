from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow, component_seed
from joint_sequence_predictor import encode_aa
from latent_source_diagnostic import training_cases, diagnostic_seed, sample_fine, fine_support, decode_sources


def case(name, split="train"):
    return dict(id=name, protein=name.split(":")[0], split=split, sequence="A"*13,
                target=torch.zeros(13,3,3), coarse=torch.zeros(4,3),
                fine_continuous=torch.zeros(13,8))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_training_selection_never_reads_validation_or_reused_diagnostics(self):
        groups=dict(PDB_train=[case("p")],train={"m":[case(f"m:{i}") for i in range(9)]},
                    validation=None,reused_diagnostic=None)
        rows=training_cases(groups,["p"],4)
        self.assertEqual(len(rows),8)
        self.assertEqual([r["case"]["id"] for r in rows[4:]],["m:0","m:2","m:5","m:8"])
        self.assertEqual([r["draw"] for r in rows[:4]],[0,1,2,3])

    def test_accidental_validation_or_duplicate_probe_is_rejected(self):
        groups=dict(PDB_train=[case("p",split="validation")],train={})
        with self.assertRaises(ValueError): training_cases(groups,["p"])
        groups["PDB_train"][0]["split"]="train"
        with self.assertRaises(ValueError): training_cases(groups,["p","p"])

    def test_fine16_exactly_replays_release_sampler_and_preserves_rng(self):
        torch.manual_seed(42)
        model=HierarchicalEnsembleFlow(torch.zeros(3),torch.ones(3),torch.zeros(8),
            torch.ones(8),width=32,layers=1,coarse_dim=3).eval()
        aa=encode_aa("ACDEFGHIKLMNP")
        coarse=torch.randn(4,3)
        state=torch.get_rng_state().clone()
        a=sample_fine(model,aa,coarse,513,16)
        b=model.sample_fine(aa,coarse,component_seed(513,"fine"),16)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),state))

    def test_support_detects_encoder_range_and_nearest_code(self):
        f=torch.tensor([[2.,0.],[0.,0.]])
        codes=torch.tensor([[0.,0.],[1.,0.]])
        result=fine_support(f,codes)
        self.assertEqual(result["outside_encoder_range_fraction"],.25)
        self.assertAlmostEqual(result["nearest_code_rmse"],2**-.5/2,places=6)
        with self.assertRaises(ValueError):fine_support(f*float("nan"),codes)

    def test_sample_seeds_are_reproducible_and_case_specific(self):
        self.assertEqual(diagnostic_seed(2026,"a",0),diagnostic_seed(2026,"a",0))
        self.assertNotEqual(diagnostic_seed(2026,"a",0),diagnostic_seed(2026,"a",1))
        self.assertNotEqual(diagnostic_seed(2026,"a",0),diagnostic_seed(2026,"b",0))

    def test_coarse_swap_reencodes_native_and_cycle_reads_decoded_sample(self):
        target=torch.full((13,3,3),3.)
        calls=[]
        class Codec:
            fine_tokenizer=SimpleNamespace(book=SimpleNamespace(codes=torch.zeros(2,8)))
            def encode(self,x):return torch.zeros(4,3),torch.full((13,8),.7)
            def project_coarse(self,c,n):return c
            def fine_target(self,x,c):
                calls.append((x.clone(),c.clone()))
                return torch.full((13,8),.8 if torch.equal(x,target) else .9)
            def decode(self,c,f,quantized=False):
                return c.repeat_interleave(4,0)[:13,None,:]+f[:,:1,None].expand(13,3,3)
        model=SimpleNamespace(sample_coarse=lambda *args:torch.ones(4,3))
        with patch("latent_source_diagnostic.sample_fine",return_value=torch.full((13,8),.2)):
            decoded,_=decode_sources(model,Codec(),encode_aa("A"*13),target,11)
        torch.testing.assert_close(calls[0][0],target)
        torch.testing.assert_close(calls[0][1],torch.ones(4,3))
        torch.testing.assert_close(calls[1][0],decoded["true_coarse_generated_fine16"])
        self.assertFalse(torch.equal(calls[1][0],target))
        self.assertAlmostEqual(float(decoded["generated_coarse_reencoded_native_fine"][0,0,0]),1.8,places=6)


if __name__=="__main__":unittest.main()
