import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from likelihood_geometry import sample_tokens,fixed_log_score,spearman,protein_interval


class LikelihoodGeometryTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_scoring_always_uses_fixed_logits(self):
        logits=torch.tensor([[2.,0.,-1.],[0.,2.,1.]])
        samples=sample_tokens(logits,2.,12,8)
        self.assertTrue(torch.equal(samples,sample_tokens(logits,2.,12,8)))
        score=fixed_log_score(logits,samples)
        expected=torch.tensor([sum(logits.log_softmax(-1)[i,t] for i,t in enumerate(row))/2 for row in samples])
        self.assertTrue(torch.allclose(score,expected))
        self.assertTrue((fixed_log_score(logits,logits.argmax(-1))>=score).all())

    def test_undefined_correlations_are_not_silently_zero(self):
        self.assertIsNone(spearman([1,2,3],[0,0,0]))
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1.)
        result=protein_interval([None,.2,.4])
        self.assertEqual(result['total'],3)
        self.assertEqual(result['defined'],2)
        self.assertAlmostEqual(result['mean'],.3)


if __name__=='__main__':unittest.main()
