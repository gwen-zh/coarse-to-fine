import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
from representation_motif_geometry import fragment_descriptors,dihedral,circular_error_degrees,fit_templates,predict_templates

class RepresentationMotifGeometryTests(unittest.TestCase):
 def setUp(self):
  with np.load('artifacts/pdb_training_v2/backbone/109L_1.npz') as z:self.bb=z['backbone'].astype(float)
 def test_frame_invariance_under_rigid_motion(self):
  rng=np.random.default_rng(71);r,_=np.linalg.qr(rng.normal(size=(3,3)))
  if np.linalg.det(r)<0:r[:,-1]*=-1
  a=fragment_descriptors(self.bb);b=fragment_descriptors(self.bb@r+rng.normal(size=3))
  np.testing.assert_allclose(a['local'],b['local'],atol=1e-10,rtol=1e-10)
  np.testing.assert_allclose(circular_error_degrees(a['angles'],b['angles']),0,atol=1e-10)
  np.testing.assert_allclose(a['local'][:,1,1],0,atol=1e-10)
 def test_dihedral_matches_mdtraj(self):
  import mdtraj as md
  top=md.Topology();chain=top.add_chain()
  residue=top.add_residue('ALA',chain)
  for _ in range(4):top.add_atom('C',md.element.carbon,residue)
  x=np.stack([self.bb[:-1,1],self.bb[:-1,2],self.bb[1:,0],self.bb[1:,1]],axis=1)
  traj=md.Trajectory((x/10).astype(np.float32),top)
  expected=md.compute_dihedrals(traj,np.array([[0,1,2,3]]),periodic=False)[:,0]
  actual=dihedral(x[:,0],x[:,1],x[:,2],x[:,3])
  np.testing.assert_allclose(circular_error_degrees(actual,expected),0,atol=1e-3)
 def test_circular_wrap(self):
  np.testing.assert_allclose(circular_error_degrees(np.radians([179,-179]),np.radians([-179,179])),[2,2],atol=1e-10)
 def test_templates_use_train_only_and_unseen_fallback(self):
  d=fragment_descriptors(self.bb);n=len(d['slot']);code=np.arange(n)%3
  fit=fit_templates(d['local'],d['angles'],d['slot'],code)
  before=fit['pair']['local'].copy()
  shape,angles=predict_templates(fit,d['slot'],np.full(n,999))
  np.testing.assert_equal(shape,fit['slot']['local'][d['slot']])
  np.testing.assert_equal(angles,fit['slot']['angles'][d['slot']])
  shape[:]=1e8;angles[:]=0
  np.testing.assert_equal(before,fit['pair']['local'])
 def test_descriptor_coverage(self):
  d=fragment_descriptors(self.bb)
  np.testing.assert_equal(d['residue'],np.arange(1,len(self.bb)-1))
  self.assertTrue(np.isfinite(d['local']).all())

if __name__=='__main__':unittest.main()
