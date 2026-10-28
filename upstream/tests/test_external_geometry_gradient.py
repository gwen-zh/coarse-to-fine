import sys
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from geometry_worker_client import backward_geometry_objective
from geometry_objective import hard_token_embedding


class ExternalGeometryGradientTests(unittest.TestCase):
    def test_external_gradient_matches_direct_frozen_decoder_chain(self):
        for weight in [0.,1.]:
            torch.manual_seed(8)
            parameter=torch.nn.Parameter(torch.randn(2,3))
            features=torch.randn(4,2);codes=torch.randn(3,5)
            logits=features@parameter
            embedding,_,_=hard_token_embedding(logits,codes)
            geometry=embedding.square().mean()
            ce=F.cross_entropy(logits,torch.tensor([0,1,2,0]))
            expected=torch.autograd.grad(ce+weight*geometry,parameter)[0]
            logits=features@parameter
            detached=logits.detach().requires_grad_()
            embedding,_,_=hard_token_embedding(detached,codes)
            external=torch.autograd.grad(embedding.square().mean(),detached)[0]
            backward_geometry_objective(F.cross_entropy(logits,torch.tensor([0,1,2,0])),logits,external,weight)
            self.assertTrue(torch.allclose(parameter.grad,expected,atol=1e-7))
            self.assertIsNone(codes.grad)

    def test_nonfinite_external_gradient_rejected_even_in_control(self):
        logits=torch.randn(2,3,requires_grad=True)
        with self.assertRaises(ValueError):
            backward_geometry_objective(logits.square().mean(),logits,torch.full_like(logits,float('nan')),0.)


if __name__=='__main__':unittest.main()
