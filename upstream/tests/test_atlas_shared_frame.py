from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from atlas_shared_frame import align_backbone_to_training_reference, encode_in_supplied_frame
from hierarchical_rigid_vq import HierarchicalRigidVQ


class SharedFrameTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(19)
    def test_alignment_removes_proper_rigid_pose_and_preserves_atoms(self):
        x=torch.randn(12,3,3);q,_=torch.linalg.qr(torch.randn(3,3))
        if torch.linalg.det(q)<0:q[:,0]*=-1
        moved=x@q+torch.tensor([3.,-4.,5.]);aligned=align_backbone_to_training_reference(moved,x)
        torch.testing.assert_close(aligned,x,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close((aligned[:,0]-aligned[:,2]).norm(dim=-1),(moved[:,0]-moved[:,2]).norm(dim=-1),atol=3e-6,rtol=3e-6)
    def test_supplied_original_frame_replays_encoder_exactly(self):
        model=HierarchicalRigidVQ(width=32,layers=1,coarse_size=8,fine_size=16).eval();x=torch.randn(12,3,3)
        c,f,target=model.encode_latents(x);c2,f2=encode_in_supplied_frame(model,target)
        torch.testing.assert_close(c,c2,atol=0,rtol=0);torch.testing.assert_close(f,f2,atol=0,rtol=0)
    def test_missing_or_mismatched_structure_rejected(self):
        with self.assertRaises(ValueError):align_backbone_to_training_reference(torch.zeros(8,3,3),torch.zeros(9,3,3))


if __name__=='__main__':unittest.main()
