from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from block_local_frames import block_frames,FrameGeometryError,vectors_to_local,vectors_to_world
from block_frame_tokenizer import BlockFrameFineTokenizer
from hierarchical_latent_contract import pool_ca,remove_block_translation
from hierarchical_rigid_vq import BB_LOCAL


class BlockLocalFrameTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(31)
        self.centers=torch.tensor([[0.,0.,0.],[4.,1.,0.],[7.,3.,2.],[9.,6.,-1.]],dtype=torch.float64)
        q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        self.rotation=q
    def test_proper_orthonormal_frames_and_equivariance(self):
        frames,_=block_frames(self.centers);rotated,_=block_frames(self.centers@self.rotation+5.)
        torch.testing.assert_close(frames.transpose(-1,-2)@frames,torch.eye(3,dtype=frames.dtype).expand(4,3,3),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(torch.linalg.det(frames),torch.ones(4,dtype=frames.dtype),atol=1e-10,rtol=1e-10)
        torch.testing.assert_close(rotated,self.rotation.T@frames,atol=1e-10,rtol=1e-10)
    def test_local_coordinate_roundtrip_and_rotation_invariance(self):
        frames,_=block_frames(self.centers);x=torch.randn(14,3,3,dtype=torch.float64);local=vectors_to_local(x,frames)
        torch.testing.assert_close(vectors_to_world(local,frames),x,atol=1e-10,rtol=1e-10)
        rotated,_=block_frames(self.centers@self.rotation)
        torch.testing.assert_close(vectors_to_local(x@self.rotation,rotated),local,atol=1e-10,rtol=1e-10)
    def test_near_straight_local_segment_is_stable_with_global_bend(self):
        c=torch.tensor([[0.,0.,0.],[4.,0.,0.],[8.,0.,0.],[12.,0.,0.],[16.,4.,3.]],dtype=torch.float64)
        a,_=block_frames(c);c[2,1]=1e-5;b,_=block_frames(c)
        self.assertTrue(torch.isfinite(a).all());self.assertLess(float((a-b).abs().max()),1e-3)
    def test_undefined_geometries_fail_explicitly(self):
        for c in [torch.zeros(2,3),torch.zeros(4,3),torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])]:
            with self.assertRaises(FrameGeometryError):block_frames(c)
    def test_two_modes_have_identical_capacity_and_initial_parameters(self):
        torch.manual_seed(43);a=BlockFrameFineTokenizer('reference');torch.manual_seed(43);b=BlockFrameFineTokenizer('local')
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]))
    def test_model_local_encoder_invariance_and_decoder_equivariance(self):
        model=BlockFrameFineTokenizer('local',fine_size=16).double().eval();c=self.centers;x=torch.randn(14,3,3,dtype=torch.float64);context=model.context(c,14);f=model.encode(x,context)
        moved=model.context(c@self.rotation+5.,14);f2=model.encode(x@self.rotation+5.,moved)
        torch.testing.assert_close(f,f2,atol=1e-9,rtol=1e-9)
        torch.testing.assert_close(model.decode_vectors(f2,moved,14),model.decode_vectors(f,context,14)@self.rotation+5.,atol=1e-8,rtol=1e-8)
    def test_gradients_rigid_geometry_and_center_contract(self):
        model=BlockFrameFineTokenizer('local',fine_size=16);c=self.centers.float();x=torch.randn(14,3,3);context=model.context(c,14);out=model(x,context);loss=(out['backbone']-x).square().mean();loss.backward()
        self.assertGreater(sum(float(p.grad.norm()) for p in model.encoder.parameters()),0)
        self.assertGreater(sum(float(p.grad.norm()) for p in model.decoder.parameters()),0)
        torch.testing.assert_close(pool_ca(out['backbone']),c,atol=2e-5,rtol=1e-6)
        ideal=torch.tensor(BB_LOCAL)
        for a,b in [(0,1),(1,2),(0,2)]:torch.testing.assert_close((out['backbone'][:,a]-out['backbone'][:,b]).norm(dim=-1),(ideal[a]-ideal[b]).norm().expand(14),atol=2e-5,rtol=1e-5)
    def test_quantizer_training_and_eval_replay(self):
        model=BlockFrameFineTokenizer('local',fine_size=16);x=torch.randn(14,3,3);context=model.context(self.centers.float(),14)
        model.book.initialize(torch.randn(64,8),2);out=model(x,context,True);self.assertEqual(out['fine_ids'].shape,(14,));(out['backbone'].square().mean()+out['commitment']).backward()
        model.eval();before={k:v.clone() for k,v in model.book.state_dict().items()};a=model(x,context,True);b=model.decode_vectors(model.book.codes[a['fine_ids']],context,14)
        torch.testing.assert_close(a['backbone'],b,atol=1e-6,rtol=1e-6)
        for k,v in before.items():self.assertTrue(torch.equal(v,model.book.state_dict()[k]))


if __name__=='__main__':unittest.main()
