import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from repa_alignment import alignment_loss
from spatial_packing import segments,pooled_broadcast

class ProbeTests(unittest.TestCase):
 def test_alignment_target_detached(self):
  torch.manual_seed(8);upstream=torch.nn.Linear(3,4);projector=torch.nn.Linear(4,2);target=torch.randn(5,2,requires_grad=True);loss=alignment_loss(projector(upstream(torch.randn(5,3))),target,torch.zeros(2),1.);loss.backward();self.assertGreater(upstream.weight.grad.norm().item(),0);self.assertIsNone(target.grad)
 def test_cosine_same_target_zero(self):
  x=torch.tensor([[1.,2.],[-1.,3.]]);self.assertAlmostEqual(alignment_loss(x,x,torch.zeros(2),1.).item(),0,places=6)
 def test_partition_boundaries_and_cap(self):
  self.assertEqual(segments('HHHCCEEEE',2),[(0,2),(2,3),(3,5),(5,7),(7,9)])
 def test_missing_dssp_singletons(self):
  self.assertEqual(segments('HXXCC'),[(0,1),(1,2),(2,3),(3,5)])
 def test_singleton_identity(self):
  x=np.arange(20,dtype=float).reshape(5,4);np.testing.assert_array_equal(pooled_broadcast(x,segments('HHHHH',1)),x)
 def test_pool_mean(self):
  x=np.array([[0.,2.],[4.,6.],[8.,10.]]);np.testing.assert_array_equal(pooled_broadcast(x,[(0,2),(2,3)]),[[2,4],[2,4],[8,10]])
 def test_reject_incomplete(self):
  with self.assertRaises(ValueError):pooled_broadcast(np.zeros((3,2)),[(0,2)])
if __name__=='__main__':unittest.main()
