import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from gradient_diagnostics import compare_gradients


class GradientDiagnosticsTests(unittest.TestCase):
    def test_distinguishes_weak_conflicting_gradient_from_zero(self):
        ce=torch.tensor([3.,4.]);kl=torch.zeros(2)
        result=compare_gradients(ce,kl,-.01*ce)
        self.assertAlmostEqual(result['geometry_over_base'],.01,places=7)
        self.assertAlmostEqual(result['geometry_cosine_base'],-1.)
        self.assertIsNone(result['geometry_cosine_kl'])
        zero=compare_gradients(ce,kl,kl)
        self.assertEqual(zero['geometry_over_base'],0.)
        self.assertIsNone(zero['geometry_cosine_base'])

    def test_combines_ce_and_kl_before_norm_and_rejects_nonfinite(self):
        ce=torch.tensor([1.,0.]);kl=torch.tensor([-1.,0.]);geometry=torch.tensor([0.,2.])
        result=compare_gradients(ce,kl,geometry)
        self.assertEqual(result['norms']['base'],0.)
        self.assertIsNone(result['geometry_over_base'])
        self.assertIsNone(result['geometry_cosine_base'])
        with self.assertRaises(ValueError):compare_gradients(ce,kl,geometry*float('nan'))


if __name__=='__main__':unittest.main()
