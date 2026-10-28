import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from coarse_solver_audit import integrate,weighted_rmsd
from latent_flow_head import euler_sample
from structural_metrics import aligned_rmsd


class Linear(torch.nn.Module):
 def __init__(self):super().__init__();self.calls=0
 def forward(self,x,t,c,valid):self.calls+=1;return x


class Tests(unittest.TestCase):
 def test_euler_identical_to_existing_sampler(self):
  x=torch.randn(1,5,3);c=torch.randn(1,5,2);valid=torch.ones(1,5,dtype=torch.bool)
  self.assertTrue(torch.equal(integrate(Linear(),x,c,valid,16),euler_sample(Linear(),x,c,valid,16)))
 def test_midpoint_converges_and_matches_nfe(self):
  x=torch.ones(1,5,3);c=torch.zeros(1,5,2);valid=torch.ones(1,5,dtype=torch.bool)
  a,b=Linear(),Linear();e=integrate(a,x,c,valid,64);m=integrate(b,x,c,valid,32,'midpoint')
  self.assertEqual(a.calls,64);self.assertEqual(b.calls,64)
  self.assertLess(float((m-np.e).abs().mean()),float((e-np.e).abs().mean()))
 def test_weighted_alignment_matches_residue_expansion(self):
  rng=np.random.default_rng(23);a=rng.normal(size=(5,3));b=rng.normal(size=(5,3));counts=np.array([4,4,4,4,1])
  self.assertAlmostEqual(weighted_rmsd(a,b,counts),aligned_rmsd(np.repeat(a,counts,axis=0),np.repeat(b,counts,axis=0)),places=12)
 def test_proper_rotation_does_not_reflect(self):
  a=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3]],float);b=a.copy();b[:,0]*=-1
  self.assertGreater(weighted_rmsd(a,b,np.ones(4)),.1)


if __name__=='__main__':unittest.main()
