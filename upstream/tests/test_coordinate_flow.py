import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from coordinate_flow import frames_from_backbone,backbone_from_frames,random_prior,interpolate,CoordinateFlow,training_loss

class CoordinateTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(91);torch.set_num_threads(1)
 def test_ideal_frame_roundtrip(self):
  r,t=random_prior(1,6,'cpu');bb=backbone_from_frames(r,t);rr,tt=frames_from_backbone(bb);torch.testing.assert_close(rr,r,atol=2e-6,rtol=2e-6);torch.testing.assert_close(tt,t)
 def test_prior_proper_rotations_and_center(self):
  r,t=random_prior(2,8,'cpu');torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=1e-6,rtol=1e-6);torch.testing.assert_close(torch.det(r),torch.ones(2,8),atol=1e-6,rtol=1e-6);self.assertLess(t.mean(1).abs().max().item(),1e-6)
 def test_path_endpoints(self):
  r,x=random_prior(1,5,'cpu');r0,x0=random_prior(1,5,'cpu')
  for t in [0.,1.]:
   rr,xx=interpolate(r,x,r0,x0,torch.tensor([t]));torch.testing.assert_close(rr,r if t else r0,atol=2e-5,rtol=2e-5);torch.testing.assert_close(xx,x if t else x0)
 def test_model_equivariance_and_condition(self):
  model=CoordinateFlow().eval();r,x=random_prior(1,5,'cpu');h=torch.randn(1,5,1536);t=torch.tensor([.4]);q,_=random_prior(1,1,'cpu');q=q[:,0]
  with torch.no_grad():
   pr,px=model(r,x,t,h);rr,xx=model(q[:,None]@r,torch.einsum('bij,bnj->bni',q,x),t,h)
  torch.testing.assert_close(rr,q[:,None]@pr,atol=3e-5,rtol=3e-5);torch.testing.assert_close(xx,torch.einsum('bij,bnj->bni',q,px),atol=3e-5,rtol=3e-5)
 def test_real_backward_finite(self):
  model=CoordinateFlow();r,x=random_prior(1,5,'cpu');r0,x0=random_prior(1,5,'cpu');loss,_=training_loss(model,r,x,torch.randn(1,5,1536),torch.tensor([.7]),r0,x0);self.assertTrue(torch.isfinite(loss));loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None];self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

if __name__=='__main__':unittest.main()
