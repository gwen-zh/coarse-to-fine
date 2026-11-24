from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from block_centroid_transport_frames import centroid_transport_frames
from block_local_frames import FrameGeometryError


class CentroidTransportTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(73)
    def test_proper_frames_and_rigid_equivariance(self):
        c=torch.randn(14,3,dtype=torch.float64);q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        a,_=centroid_transport_frames(c);b,_=centroid_transport_frames(c@q+5.)
        torch.testing.assert_close(a.transpose(-1,-2)@a,torch.eye(3,dtype=a.dtype).expand(14,3,3),atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(torch.linalg.det(a),torch.ones(14,dtype=a.dtype),atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(b,q.T@a,atol=1e-9,rtol=1e-9)
    def test_seed_stable_when_opposite_farthest_anchors_switch(self):
        c=torch.tensor([[0.,0.,0.],[4.,0.,0.],[8.,1.,0.],[8.,0.,10.],[8.,0.,-10.]],dtype=torch.float64)
        c[-1,-1]-=1e-5;a,_=centroid_transport_frames(c);c[-1,-1]+=2e-5;b,_=centroid_transport_frames(c)
        self.assertLess(float((a-b).abs().max()),1e-3)
    def test_antiparallel_remains_finite_and_proper(self):
        c=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,0.,0.],[-1.,1.,0.]],dtype=torch.float64);a,_=centroid_transport_frames(c)
        self.assertTrue(torch.isfinite(a).all());torch.testing.assert_close(torch.linalg.det(a),torch.ones(4,dtype=a.dtype),atol=1e-9,rtol=1e-9)
    def test_collinear_failure_is_explicit(self):
        with self.assertRaises(FrameGeometryError):centroid_transport_frames(torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]))


if __name__=='__main__':unittest.main()
