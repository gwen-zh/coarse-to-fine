import sys,unittest,copy
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from hierarchical_rigid_vq import HierarchicalRigidVQ,EMAVectorQuantizer,canonical_backbone,orthogonal_frame,rigid_backbone,BB_LOCAL,tokenizer_loss
from hierarchical_latent_contract import pool_ca


class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(1);torch.manual_seed(19);self.model=HierarchicalRigidVQ(width=32,layers=1,coarse_size=4,fine_size=8).eval();self.x=torch.randn(17,3,3)
 def test_frames_proper_even_degenerate(self):
  for x,y in [(torch.zeros(4,3),torch.zeros(4,3)),(torch.ones(4,3),torch.ones(4,3)),(torch.randn(4,3),torch.randn(4,3))]:
   r=orthogonal_frame(x,y);torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand(4,3,3),atol=1e-5,rtol=1e-5);torch.testing.assert_close(torch.linalg.det(r),torch.ones(4),atol=1e-5,rtol=1e-5)
 def test_rigid_bonds_and_angle_survive_centroid_projection(self):
  o=self.model(self.x);bb=o['backbone'];local=torch.tensor(BB_LOCAL)
  for a,b in [(0,1),(1,2),(0,2)]:torch.testing.assert_close((bb[:,a]-bb[:,b]).norm(dim=-1),(local[a]-local[b]).norm().expand(len(bb)),atol=1e-5,rtol=1e-5)
  torch.testing.assert_close(pool_ca(bb),o['coarse'],atol=2e-5,rtol=1e-6)
 def test_canonicalization_removes_proper_rotation_translation(self):
  q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q)
  torch.testing.assert_close(canonical_backbone(self.x),canonical_backbone(self.x@q+7.),atol=5e-6,rtol=5e-6)
  a,b=self.model.encode_latents(self.x)[:2],self.model.encode_latents(self.x@q+7.)[:2]
  for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=5e-6,rtol=5e-6)
 def test_no_reference_decode_and_distinct_scales(self):
  c,f,_=self.model.encode_latents(self.x);self.assertEqual(c.shape,(5,16));self.assertEqual(f.shape,(17,8));self.assertEqual(self.model.decode(c,f,17)['backbone'].shape,(17,3,3))
 def test_fine_changes_cannot_move_coarse_centers(self):
  c,f,_=self.model.encode_latents(self.x);a=self.model.decode(c,f,17);b=self.model.decode(c,f+.2,17)
  torch.testing.assert_close(pool_ca(a['backbone']),pool_ca(b['backbone']),atol=2e-5,rtol=1e-6)
 def test_codebook_frozen_in_eval_and_hard_forward_has_encoder_gradients(self):
  book=EMAVectorQuantizer(8,3);book.initialize(torch.randn(64,3),2);book.eval();before=copy.deepcopy(book.state_dict());z=torch.randn(11,3,requires_grad=True);q,ids,loss=book(z)
  torch.testing.assert_close(q,book.codes[ids],atol=1e-7,rtol=1e-6);(q.square().mean()+loss).backward();self.assertGreater(float(z.grad.norm()),0)
  for k,v in before.items():self.assertTrue(torch.equal(v,book.state_dict()[k]))
 def test_ema_updates_no_trainable_code_parameters(self):
  book=EMAVectorQuantizer(4,3);book.initialize(torch.randn(64,3),2);before=book.codes.clone();book(torch.randn(11,3));self.assertFalse(torch.equal(before,book.codes));self.assertEqual(len(list(book.parameters())),0)
 def test_token_only_decode_replays_quantized_eval(self):
  self.model.coarse_book.initialize(torch.randn(32,16),2);self.model.fine_book.initialize(torch.randn(64,8),2)
  o=self.model(self.x,True);c,f=self.model.tokens(self.x);a=self.model.decode_tokens(c,f)
  torch.testing.assert_close(o['backbone'],a['backbone'],atol=2e-5,rtol=2e-5)
 def test_all_modules_receive_loss_and_short_chains_work(self):
  weights=dict(coarse=1.,fine=1.,full=.1,bond=.1,angle=.05,clash=.01,latent=1e-4)
  loss,_=tokenizer_loss(self.model(self.x),weights);loss.backward()
  for m in [self.model.coarse_encoder,self.model.coarse_decoder,self.model.fine_encoder,self.model.fine_decoder]:self.assertGreater(sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None),0)
  for n in [1,4,5,9]:self.assertTrue(torch.isfinite(self.model(self.x[:n])['backbone']).all())


if __name__=='__main__':unittest.main()
