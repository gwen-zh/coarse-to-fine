import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from hierarchical_token_generator import HierarchicalTokenGenerator


class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(19);self.model=HierarchicalTokenGenerator(condition_dim=12,width=32,layers=1,coarse_size=4,fine_size=8).eval();self.h=torch.randn(13,12);self.c=torch.randint(4,(4,));self.f=torch.randint(8,(13,))
 def test_actual_fine_input_changes_predictions(self):
  a=self.model(self.h,self.c,self.f,.5,'fine')[1];b=self.model(self.h,self.c,torch.full_like(self.f,8),.5,'fine')[1]
  self.assertFalse(torch.equal(a,b))
 def test_both_losses_reach_sequence_condition_and_heads(self):
  for stage,head in [('coarse',self.model.coarse_head),('fine',self.model.fine_head)]:
   self.model.zero_grad(set_to_none=True);loss,record=self.model.denoise_loss(self.h,self.c,self.f,stage,rate=1.);loss.backward()
   self.assertGreater(sum(float(v.grad.norm()) for v in head.parameters() if v.grad is not None),0)
   self.assertGreater(float(self.model.condition[1].weight.grad.norm()),0);self.assertEqual(record['masked'],len(self.c) if stage=='coarse' else len(self.f))
 def test_sampling_uses_no_reference_fine_during_coarse(self):
  seen=[];original=self.model.forward
  def forward(features,c,f,t,stage):
   if stage=='coarse':self.assertTrue((f==8).all())
   seen.append(stage);return original(features,c,f,t,stage)
  self.model.forward=forward
  c,f=self.model.sample(self.h,19,steps=3)
  self.assertTrue((c<4).all());self.assertTrue((f<8).all());self.assertEqual(seen,['coarse']*3+['fine']*3)
 def test_same_seed_replays_and_ragged_lengths(self):
  for n in [1,4,5,13]:
   a=self.model.sample(self.h[:n],3,steps=4);b=self.model.sample(self.h[:n],3,steps=4)
   for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
   self.assertEqual(a[0].shape,((n+3)//4,));self.assertEqual(a[1].shape,(n,))
 def test_generated_coarse_condition_is_supported(self):
  sampled=self.model.sample_coarse(self.h,17,steps=2);loss,_=self.model.denoise_loss(self.h,self.c,self.f,'fine',coarse_condition=sampled)
  self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
