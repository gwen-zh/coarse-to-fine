import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from backbone_quality_audit import angle,omega,measurements

class QualityTests(unittest.TestCase):
    def test_angle(self):
        np.testing.assert_allclose(angle(np.array([[1.,0,0]]),np.zeros((1,3)),np.array([[0.,1,0]])),[90.])
    def test_identical_reference_and_rigid_motion(self):
        x=np.random.default_rng(4).normal(size=(8,3,3));q,_=np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)));q[:,0]*=np.linalg.det(q)
        a=measurements(x,x);b=measurements(x@q+5,x)
        for key in a:
            self.assertAlmostEqual(a[key],b[key],places=10)
            if key!='peptide_planarity_deviation_degrees':self.assertAlmostEqual(a[key],0.,places=10)
    def test_cis_and_trans_both_planar(self):
        for side in [-1.,1.]:
            x=np.array([[[0,2,1],[0,1,0],[0,0,0]],[[1,0,0],[1,side,0],[2,1,1]]],float)
            self.assertAlmostEqual(measurements(x,x)['peptide_planarity_deviation_degrees'],0.)
    def test_nonfinite_rejected(self):
        x=np.ones((3,3,3));x[0,0,0]=np.nan
        with self.assertRaises(ValueError):measurements(x,x)

if __name__=="__main__":unittest.main()
