from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from fine_coarse_use import CELLS, crossed_inputs, response, validate_cells


class FineCoarseUseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.c=torch.randn(4,4,3);self.other_c=torch.randn(4,4,3)
        self.f=torch.randn(4,13,8);self.other_f=torch.randn(4,13,8)

    def test_crosses_preserve_exact_rows_columns_and_partial_block_pairing(self):
        inputs=crossed_inputs(self.c,self.f,self.other_c,self.other_f)
        self.assertEqual(list(inputs),CELLS)
        self.assertIs(inputs['ownC_ownF'][0],inputs['ownC_priorF'][0])
        self.assertIs(inputs['priorC_ownF'][0],inputs['priorC_priorF'][0])
        self.assertIs(inputs['ownC_ownF'][1],inputs['priorC_ownF'][1])
        self.assertIs(inputs['ownC_priorF'][1],inputs['priorC_priorF'][1])

    def test_mismatched_samples_and_nonfinite_latents_are_not_filtered(self):
        with self.assertRaises(ValueError):crossed_inputs(self.c,self.f,self.other_c[:3],self.other_f)
        bad=self.other_f.clone();bad[2,7,3]=float('nan')
        with self.assertRaisesRegex(ValueError,'Finite'):crossed_inputs(self.c,self.f,self.other_c,bad)

    def test_response_has_zero_control_and_uses_original_normalization(self):
        self.assertEqual(response(self.f,self.f,torch.ones(8))['fine_change_RMS'],0.)
        result=response(self.f,self.f+2,torch.full((8,),2.))
        self.assertAlmostEqual(result['fine_change_RMS'],2.,places=6)
        self.assertAlmostEqual(result['train_normalized_fine_change_RMS'],1.,places=6)

    def test_report_pairing_rejects_duplicate_missing_or_mixed_cells(self):
        records=[dict(seed=2026,protein='p',gauge='raw',cell=c) for c in CELLS]
        self.assertEqual(set(validate_cells(records)),set(CELLS))
        with self.assertRaises(ValueError):validate_cells(records[:3])
        records[3]=dict(records[2])
        with self.assertRaises(ValueError):validate_cells(records)
        records=[dict(seed=2026,protein='p',gauge='raw',cell=c) for c in CELLS]
        records[3]['protein']='other'
        with self.assertRaises(ValueError):validate_cells(records)


if __name__=='__main__':unittest.main()
