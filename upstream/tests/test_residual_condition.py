import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from residual_condition import residual_target,MidPredictor


class ResidualConditionTests(unittest.TestCase):
    def test_target_tracks_actual_coarse_and_parallel_is_unchanged(self):
        z=torch.tensor([[3.,4.],[5.,6.]])
        codes=torch.tensor([[1.,0.],[0.,2.]])
        native=torch.tensor([0,1]);predicted=torch.tensor([1,0])
        self.assertTrue(torch.equal(residual_target(z,predicted,codes,'residual'),z-codes[predicted]))
        self.assertFalse(torch.equal(residual_target(z,native,codes,'residual'),residual_target(z,predicted,codes,'residual')))
        self.assertTrue(torch.equal(residual_target(z,predicted,codes,'parallel'),z))
        with self.assertRaises(ValueError):residual_target(z,torch.tensor([0,2]),codes,'residual')

    def test_mid_predictor_requires_only_available_context(self):
        model=MidPredictor(width=4,size=7)
        context=torch.randn(5,4)
        logits=model(context)
        self.assertEqual(logits.shape,(5,7))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,2,3,4])).backward()
        self.assertGreater(model.net[-1].weight.grad.abs().sum(),0)


if __name__=='__main__':unittest.main()
