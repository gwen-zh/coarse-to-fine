import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from coarse_geometry_supervision import shape_losses,geometry_flow_loss,ARMS
from mixed_hierarchical_generator import MixedHierarchicalGenerator,HybridTokenizer
from hierarchical_rigid_vq import HierarchicalRigidVQ
from coarse_solver_audit import weighted_rmsd

class Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(8)
    def test_proper_rotation_translation_invariance(self):
        y=torch.randn(7,3,dtype=torch.float64);q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        if torch.det(q)<0:q[:,0]*=-1
        x=y@q+torch.tensor([4.,-2.,7.]);a,d=shape_losses(x,y,torch.tensor([4,4,4,4,4,4,1]))
        self.assertLess(float(a),1e-20);self.assertLess(float(d),1e-20)
    def test_reflection_is_not_accepted_as_proper_alignment(self):
        y=torch.tensor([[0.,0,0],[1,0,0],[0,2,0],[0,0,3]],dtype=torch.float64);x=y.clone();x[:,0]*=-1
        a,d=shape_losses(x,y,torch.ones(4));self.assertGreater(float(a),1e-4);self.assertLess(float(d),1e-20)
    def test_alignment_matches_independent_numpy_weighted_metric(self):
        x=torch.randn(7,3,dtype=torch.float64);y=torch.randn(7,3,dtype=torch.float64);w=torch.tensor([4,4,4,4,4,4,1])
        a,_=shape_losses(x,y,w);expected=weighted_rmsd(x.numpy(),y.numpy(),w.numpy())**2/300.
        self.assertAlmostEqual(float(a),expected,places=12)
    def test_geometry_gradient_matches_finite_differences(self):
        x=torch.randn(6,3,dtype=torch.float64,requires_grad=True);y=torch.randn(6,3,dtype=torch.float64);w=torch.tensor([4,4,4,4,4,1])
        self.assertTrue(torch.autograd.gradcheck(lambda a:sum(shape_losses(a,y,w)),(x,),eps=1e-6,atol=1e-5,rtol=1e-4))
    def test_degenerate_collapsed_prediction_has_finite_gradient(self):
        x=torch.zeros(6,3,requires_grad=True);y=torch.randn(6,3);loss=sum(shape_losses(x,y,torch.ones(6)));loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(float(x.grad.norm()),0.)
    def test_control_is_exact_existing_flow_loss_and_rng_is_paired(self):
        h=torch.randn(13,12);c=torch.tanh(torch.randn(4,16));decoder=HybridTokenizer(HierarchicalRigidVQ(width=16,layers=1,fine_size=8))
        m=MixedHierarchicalGenerator(torch.zeros(16),torch.ones(16),condition_dim=12,width=32,coarse_layers=1,fine_layers=1,fine_size=8)
        for seed in [7,11,29]:
            torch.manual_seed(seed);expected,_=m.coarse_loss(h,c,decoder,.1)
            torch.manual_seed(seed);actual,_=geometry_flow_loss(m,decoder,h,c,'control')
            self.assertTrue(torch.equal(actual,expected))
        hashes=[];rng=[]
        for arm in ARMS:
            m.zero_grad(set_to_none=True);torch.manual_seed(57);loss,t=geometry_flow_loss(m,decoder,h,c,arm);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None for p in decoder.parameters()))
            hashes.append(t['draw_sha256']);rng.append(torch.get_rng_state().clone())
        self.assertEqual(len(set(hashes)),1);self.assertTrue(all(torch.equal(s,rng[0]) for s in rng))
    def test_invalid_centers_and_counts_rejected(self):
        with self.assertRaises(ValueError):shape_losses(torch.ones(3,3),torch.ones(3,3),torch.tensor([4,0,4]))
        with self.assertRaises(ValueError):shape_losses(torch.full((3,3),float('nan')),torch.ones(3,3),torch.ones(3))

if __name__=='__main__':unittest.main()
