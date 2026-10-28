import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from hierarchical_latent_contract import pool_ca,lift_anchors,remove_block_translation,compose_levels,split_reference

class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);torch.manual_seed(13)
 def test_decomposition_reconstructs_all_atoms_including_ragged_end(self):
  for n in [1,3,4,5,17]:
   x=torch.randn(n,3,3,dtype=torch.float64);c,r=split_reference(x)
   torch.testing.assert_close(compose_levels(c,r),x,atol=1e-12,rtol=1e-12)
 def test_fine_changes_cannot_move_coarse_centroids(self):
  c=torch.randn(5,3,dtype=torch.float64)
  for scale in [1.,100.]:
   output=compose_levels(c,torch.randn(17,3,3,dtype=torch.float64)*scale)
   torch.testing.assert_close(pool_ca(output),c,atol=1e-12,rtol=1e-12)
 def test_rigid_transform_equivariance(self):
  x=torch.randn(17,3,3,dtype=torch.float64);q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q);translation=torch.randn(3,dtype=torch.float64)
  c,r=split_reference(x);d,s=split_reference(x@q+translation)
  torch.testing.assert_close(d,c@q+translation);torch.testing.assert_close(s,r@q)
 def test_block_perturbation_stays_inside_block(self):
  c=torch.randn(5,3);r=torch.randn(17,3,3);changed=r.clone();changed[5,0]+=2
  a=compose_levels(c,r);b=compose_levels(c,changed);mask=torch.ones(17,dtype=torch.bool);mask[4:8]=False
  torch.testing.assert_close(a[mask],b[mask])
 def test_coarse_loss_has_no_fine_gradient(self):
  c=torch.randn(4,3,dtype=torch.float64,requires_grad=True);r=torch.randn(15,3,3,dtype=torch.float64,requires_grad=True)
  loss=pool_ca(compose_levels(c,r)).square().sum();loss.backward()
  self.assertGreater(float(c.grad.norm()),0);self.assertLess(float(r.grad.norm()),1e-12)
 def test_fine_path_has_gradient_and_singleton_atom_geometry(self):
  c=torch.randn(2,3,dtype=torch.float64,requires_grad=True);r=torch.randn(5,3,3,dtype=torch.float64,requires_grad=True)
  output=compose_levels(c,r);output[4,0].square().sum().backward();self.assertGreater(float(r.grad[4,0].norm()),0)
 def test_no_global_or_shape_bypass(self):
  with self.assertRaises(ValueError):pool_ca(torch.zeros(2,3))
  with self.assertRaises(ValueError):compose_levels(torch.zeros(9,3),torch.zeros(17,3,3))
  with self.assertRaises(ValueError):pool_ca(torch.zeros(0,3,3))

if __name__=='__main__':unittest.main()
