import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from mixed_hierarchical_generator import HybridTokenizer,MixedHierarchicalGenerator,pooled_sequence
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_latent_contract import pool_ca


class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(19)
  self.model=MixedHierarchicalGenerator(torch.zeros(16),torch.ones(16),condition_dim=12,width=32,coarse_layers=1,fine_layers=1,fine_size=8).eval();self.h=torch.randn(13,12);self.c=torch.randn(4,16);self.f=torch.randint(8,(13,))
 def test_ragged_sequence_pool_has_correct_denominator(self):
  x=torch.arange(5.).reshape(5,1);torch.testing.assert_close(pooled_sequence(x),torch.tensor([[1.5],[4.]]))
 def test_normalization_roundtrip(self):
  model=MixedHierarchicalGenerator(torch.arange(16.)/10,torch.arange(16.)/10+.1,condition_dim=12,width=32,coarse_layers=1,fine_layers=1,fine_size=8)
  torch.testing.assert_close(model.denormalize(model.normalize(self.c)),self.c)
 def test_continuous_coarse_and_fine_inputs_both_influence_predictions(self):
  a=self.model.fine(self.h,self.c,self.f,.5)
  self.assertFalse(torch.equal(a,self.model.fine(self.h,self.c+.2*torch.randn_like(self.c),self.f,.5)))
  self.assertFalse(torch.equal(a,self.model.fine(self.h,self.c,torch.full_like(self.f,8),.5)))
 def test_fine_loss_reaches_condition_and_coarse_projection(self):
  loss=self.model.fine.denoise_loss(self.h,self.c,self.f,rate=1.);loss.backward()
  self.assertGreater(float(self.model.fine.condition[1].weight.grad.norm()),0);self.assertGreater(float(self.model.fine.coarse_projection[0].weight.grad.norm()),0)
 def test_sampling_requires_only_sequence_features_and_seed(self):
  a=self.model.sample(self.h,17,coarse_steps=3,fine_steps=3);b=self.model.sample(self.h,17,coarse_steps=3,fine_steps=3)
  for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
  self.assertEqual(a[0].shape,(4,16));self.assertTrue((a[1]<8).all());self.assertTrue(torch.isfinite(a[0]).all())
 def test_short_and_ragged_sampling(self):
  for n in [1,4,5,13]:
   c,f=self.model.sample(self.h[:n],19,coarse_steps=2,fine_steps=2);self.assertEqual(c.shape,((n+3)//4,16));self.assertEqual(f.shape,(n,))
 def test_hybrid_decode_preserves_rigid_geometry_and_coarse_centers(self):
  base=HierarchicalRigidVQ(width=32,layers=1,coarse_size=4,fine_size=8);base.fine_book.initialize(torch.randn(32,8),2);tok=HybridTokenizer(base)
  bb=tok.decode(self.c,self.f);torch.testing.assert_close(pool_ca(bb),tok.coarse_centers(self.c,len(self.f)),atol=2e-5,rtol=1e-6)
  local=bb.new_tensor([[.5256,1.3612,0.],[0,0,0],[-1.5251,0,0]])
  torch.testing.assert_close((bb[:,0]-bb[:,1]).norm(dim=-1),(local[0]-local[1]).norm().expand(len(bb)),atol=1e-5,rtol=1e-5)
 def test_flow_gradient_through_frozen_coarse_decoder(self):
  base=HierarchicalRigidVQ(width=32,layers=1,coarse_size=4,fine_size=8);tok=HybridTokenizer(base)
  opt=torch.optim.Adam(self.model.coarse_flow.parameters(),lr=1e-3)
  for _ in range(2):
   opt.zero_grad();loss,_=self.model.coarse_loss(self.h,self.c,tok);loss.backward();opt.step()
  self.assertGreater(float(self.model.coarse_flow.condition[1].weight.grad.norm()),0)
  self.assertTrue(all(p.grad is None and not p.requires_grad for p in tok.parameters()))
 def test_invalid_continuous_shapes_rejected(self):
  with self.assertRaises(ValueError):self.model.fine(self.h,self.c[:3],self.f,.5)
  with self.assertRaises(ValueError):self.model.fine(self.h,self.c.long(),self.f,.5)


if __name__=='__main__':unittest.main()
