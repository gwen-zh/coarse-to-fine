import sys
from pathlib import Path
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from geometry_objective import hard_token_embedding,backbone_penalties,mix_logit_gradients


class GeometryObjectiveTests(unittest.TestCase):
    def test_forward_is_exact_hard_lookup_and_gradient_reaches_logits(self):
        logits=torch.tensor([[2.,1.,0.],[0.,2.,1.]],requires_grad=True)
        codes=torch.tensor([[1.,0.],[0.,2.],[-1.,1.]])
        embedding,ids,_=hard_token_embedding(logits,codes)
        self.assertTrue(torch.equal(embedding,codes[ids]))
        embedding.square().sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(),0)

    def test_cn_penalty_has_zero_safe_band_and_correct_gradient_direction(self):
        bb=torch.tensor([[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[1.4,0.,0.],[4.,1.,0.],[4.,0.,0.]]],requires_grad=True)
        cn,clash=backbone_penalties(bb)
        self.assertEqual(cn.item(),0.)
        self.assertEqual(clash.item(),0.)
        bad=bb.detach().clone();bad[1,0,0]=2.4;bad.requires_grad_()
        cn,_=backbone_penalties(bad);cn.backward()
        self.assertGreater(bad.grad[1,0,0].item(),0)
        self.assertLess(bad.grad[0,2,0].item(),0)

    def test_gradient_shuffle_preserves_norm_and_kl_only_fixed_point(self):
        gradient=torch.randn(4,7);p=torch.randn(4,7).softmax(-1);order=torch.randperm(7)
        shuffled=mix_logit_gradients(gradient,p,p,permutation=order)
        self.assertTrue(torch.allclose(shuffled.norm(),gradient.norm()))
        self.assertTrue(torch.equal(mix_logit_gradients(gradient,p,p,geometry_weight=0),torch.zeros_like(gradient)))


if __name__=='__main__':unittest.main()
