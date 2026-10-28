import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from two_level_autoencoder import TwoLevelAutoencoder,training_loss
from hierarchical_latent_contract import pool_ca,remove_block_translation

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(17);self.model=TwoLevelAutoencoder().eval();self.bb=torch.randn(17,3,3)
 def test_distinct_shapes_and_no_reference_decode(self):
  c,f=self.model.encode(self.bb);self.assertEqual(c.shape,(5,8));self.assertEqual(f.shape,(17,4))
  output=self.model.decode(c,f,17);self.assertEqual(output['backbone'].shape,self.bb.shape);torch.testing.assert_close(pool_ca(output['backbone']),output['coarse'],atol=2e-6,rtol=2e-6)
 def test_translation_invariance(self):
  a=self.model(self.bb);b=self.model(self.bb+3)
  torch.testing.assert_close(a['backbone'],b['backbone'],atol=1e-5,rtol=1e-5)
 def test_coarse_only_sees_centroids(self):
  change=remove_block_translation(torch.randn_like(self.bb));a,_=self.model.encode(self.bb);b,_=self.model.encode(self.bb+change)
  torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
 def test_fine_encoder_cannot_see_block_translation(self):
  moved=self.bb.clone();moved[4:8]+=torch.tensor([5.,2.,-1.]);_,a=self.model.encode(self.bb);_,b=self.model.encode(moved)
  torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
 def test_fine_decoder_is_block_local(self):
  c,f=self.model.encode(self.bb);g=f.clone();g[4:8]+=.5;a=self.model.decode(c,f,17);b=self.model.decode(c,g,17);mask=torch.ones(17,dtype=torch.bool);mask[4:8]=False
  torch.testing.assert_close(a['backbone'][mask],b['backbone'][mask]);torch.testing.assert_close(pool_ca(a['backbone']),pool_ca(b['backbone']),atol=2e-6,rtol=2e-6)
 def test_gradients_reach_all_four_learned_modules(self):
  o=self.model(self.bb);weights=dict(coarse=1.,fine=1.,full=.1,bond=.1,angle=.05,clash=.01,latent=1e-4);loss,_=training_loss(o,self.bb,weights);loss.backward()
  for module in [self.model.coarse_encoder,self.model.coarse_decoder,self.model.fine_encoder,self.model.fine_decoder]:
   grad=sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None);self.assertGreater(float(grad),0);self.assertTrue(torch.isfinite(grad))
 def test_singleton_and_ragged_end(self):
  for n in [1,4,5,9]:
   o=self.model(torch.randn(n,3,3));self.assertEqual(o['backbone'].shape,(n,3,3));self.assertTrue(torch.isfinite(o['backbone']).all())
 def test_coarse_loss_cannot_train_fine_modules(self):
  o=self.model(self.bb);pool_ca(o['backbone']).square().mean().backward()
  for module in [self.model.fine_encoder,self.model.fine_decoder]:
   norms=[float(p.grad.norm()) for p in module.parameters() if p.grad is not None];self.assertLess(max(norms,default=0),1e-6)

if __name__=='__main__':unittest.main()
