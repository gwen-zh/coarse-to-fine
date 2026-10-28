import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from latent_sensitivity_math import covariance_root,unit_rms_noise,nearest_cell_faces,aligned_ca_distances
from structural_metrics import aligned_rmsd

class SensitivityMathTests(unittest.TestCase):
    def test_covariance_root(self):
        cov=np.array([[4.,1.],[1.,2.]])
        root=covariance_root(cov);np.testing.assert_allclose(root@root.T,cov,atol=1e-12)
        noise=unit_rms_noise(np.random.default_rng(3).normal(size=(100,2)),root)
        self.assertAlmostEqual(float(np.mean(noise*noise)),1.)
    def test_bisector_distance_and_crossing(self):
        codes=np.array([[0.,0.],[2.,0.],[0.,4.]])
        z=np.array([[.2,.1],[1.7,0.]])
        margin,neighbor,direction=nearest_cell_faces(z,codes,np.array([0,1]))
        np.testing.assert_allclose(margin,[.8,.7]);np.testing.assert_array_equal(neighbor,[1,0])
        for factor,expected in [(.99,[0,1]),(1.01,[1,0])]:
            x=z+factor*margin[:,None]*direction
            np.testing.assert_array_equal(((x[:,None]-codes)**2).sum(-1).argmin(-1),expected)
    def test_duplicate_code_does_not_create_fake_zero_margin(self):
        r,_,_=nearest_cell_faces(np.array([[.1,0.]]),np.array([[0.,0.],[0.,0.],[2.,0.]]),np.array([0]))
        np.testing.assert_allclose(r,[.9])
    def test_bad_assignment_rejected(self):
        with self.assertRaises(ValueError):nearest_cell_faces(np.array([[2.,0.]]),np.array([[0.,0.],[2.,0.]]),np.array([0]))
    def test_aligned_distance_matches_existing_rmsd(self):
        x=np.random.default_rng(2).normal(size=(20,3));y=x.copy();y[0]+=1
        d=aligned_ca_distances(y,x);self.assertAlmostEqual(float(np.sqrt(np.mean(d*d))),aligned_rmsd(y,x),places=12)

if __name__=="__main__":unittest.main()
