import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from half_fine_tokenizer import HalfFineTokenizer,retained_positions,sanitize_fine,hidden_embedding_loss
from multiscale_tokenizer import pair_geometry

class HalfFineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43);torch.set_num_threads(1)
        self.z=torch.randn(21,128);self.e=torch.randn(21,40)
        self.keep=retained_positions(21);self.mean=torch.randn(40);self.scale=torch.ones(40)
        self.coarse=torch.randn(21,32)
    def test_fixed_bit_budget(self):
        for n in (20,21):
            k=retained_positions(n);self.assertEqual(int(k.sum()),(n+1)//2)
            self.assertEqual(5*n+12*int(k.sum()),11*n+(6 if n%2 else 0))
    def test_hidden_fine_no_leak_with_active_fusion(self):
        model=HalfFineTokenizer('global',40).eval()
        torch.nn.init.normal_(model.fusion.output.weight,std=.02)
        z,e=self.z.clone(),self.e.clone();z[~self.keep]=float('nan');e[~self.keep]=float('nan')
        a=model.fusion(self.z,self.e,self.keep,self.coarse,self.scale,self.mean)
        b=model.fusion(z,e,self.keep,self.coarse,self.scale,self.mean)
        self.assertTrue(torch.isfinite(b).all());torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_hidden_fine_zero_fusion_gradient(self):
        model=HalfFineTokenizer('local',40).eval();torch.nn.init.normal_(model.fusion.output.weight,std=.02)
        z=self.z.clone().requires_grad_();e=self.e.clone().requires_grad_()
        model.fusion(z,e,self.keep,self.coarse,self.scale,self.mean).square().mean().backward()
        self.assertEqual(float(z.grad[~self.keep].abs().sum()),0.)
        self.assertEqual(float(e.grad[~self.keep].abs().sum()),0.)
        self.assertGreater(float(z.grad[self.keep].abs().sum()),0.)
    def test_initial_half_baseline_and_capacity_match(self):
        pair,mask=pair_geometry(torch.randn(21,3,3));counts=[]
        for mode in ('none','local','global'):
            model=HalfFineTokenizer(mode,40).eval()
            output,ids,_=model(self.z,self.e,pair,mask,self.scale,self.mean)
            _,base=sanitize_fine(self.z,self.e,self.keep,self.mean)
            self.assertTrue(torch.equal(output,base))
            if mode=='none':self.assertIsNone(ids)
            else:counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertEqual(counts[0],counts[1])
    def test_no_coarse_ignores_teacher_geometry(self):
        model=HalfFineTokenizer('none',40).eval();torch.nn.init.normal_(model.fusion.output.weight,std=.02)
        pair,mask=pair_geometry(torch.randn(21,3,3))
        a,_,_=model(self.z,self.e,pair,mask,self.scale,self.mean)
        z=self.z.clone();e=self.e.clone();z[~self.keep]=10000;e[~self.keep]=-10000
        b,_,_=model(z,e,torch.full_like(pair,float('nan')),~mask,self.scale,self.mean)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_distillation_only_hidden_positions(self):
        pred=self.e.clone();pred[self.keep]+=100
        self.assertEqual(float(hidden_embedding_loss(pred,self.e,self.keep,self.scale)),0.)

if __name__=='__main__':unittest.main()
