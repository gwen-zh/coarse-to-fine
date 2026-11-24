from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from block_transport_frames import transported_block_frames
from block_local_frames import FrameGeometryError,vectors_to_local,vectors_to_world


class TransportFrameTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(63)
    def test_proper_frames_equivariance_and_inverse(self):
        c=torch.randn(12,3,dtype=torch.float64);q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        a,_=transported_block_frames(c);b,_=transported_block_frames(c@q+3.)
        torch.testing.assert_close(a.transpose(-1,-2)@a,torch.eye(3,dtype=a.dtype).expand(12,3,3),atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(torch.linalg.det(a),torch.ones(12,dtype=a.dtype),atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(b,q.T@a,atol=1e-9,rtol=1e-9)
        x=torch.randn(46,3,3,dtype=a.dtype);torch.testing.assert_close(vectors_to_world(vectors_to_local(x,a),a),x,atol=1e-9,rtol=1e-9)
    def test_antiparallel_tangent_case_is_finite(self):
        c=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,0.,0.],[-1.,1.,0.]],dtype=torch.float64)
        frames,info=transported_block_frames(c);self.assertTrue(torch.isfinite(frames).all());self.assertGreaterEqual(info['near_antiparallel_transports'],1)
        torch.testing.assert_close(torch.linalg.det(frames),torch.ones(4,dtype=frames.dtype),atol=1e-9,rtol=1e-9)
    def test_locally_straight_segment_stability(self):
        c=torch.tensor([[0.,0.,0.],[4.,0.,0.],[8.,0.,0.],[12.,0.,0.],[16.,4.,3.]],dtype=torch.float64)
        a,_=transported_block_frames(c);c[2,1]=1e-5;b,_=transported_block_frames(c);self.assertLess(float((a-b).abs().max()),1e-3)
    def test_globally_undefined_trace_fails(self):
        with self.assertRaises(FrameGeometryError):transported_block_frames(torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]))


if __name__=='__main__':unittest.main()
