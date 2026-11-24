from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from aa_structural_coarse_condition import GAUGES, coarse_prompt, counts, proper_rotation, center_distances
from block_local_frames import FrameGeometryError
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from hierarchical_latent_contract import pool_ca
from sequence_multiscale_ensemble import MultiscaleStructureCodec


class StructuralCoarseConditionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(65)
        self.x=torch.randn(21,3,3)*4+torch.tensor([11.,-23.,17.])
        self.own=pool_ca(self.x)-self.x[:,1].mean(0)

    def test_all_gauges_preserve_shape_distances_handedness_and_weighted_center(self):
        source=pool_ca(self.x)
        for gauge in GAUGES:
            c,r=coarse_prompt(self.x,self.own,gauge)
            torch.testing.assert_close(torch.cdist(c,c),torch.cdist(source,source),atol=6e-6,rtol=2e-5)
            torch.testing.assert_close(r.T@r,torch.eye(3),atol=1e-6,rtol=1e-6)
            self.assertGreater(float(torch.linalg.det(r)),.99999)
            torch.testing.assert_close((c*counts(21,c)[:,None]).sum(0),torch.zeros(3),atol=6e-6,rtol=0)

    def test_weighted_alignment_recovers_rigid_target_with_partial_final_block(self):
        x=self.own.double();q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
        q[:,-1]*=torch.linalg.det(q);target=x@q+7
        found=proper_rotation(x,target,21)
        torch.testing.assert_close(found,q,atol=1e-12,rtol=1e-12)

    def test_distance_audit_is_stable_for_more_than_25_centers(self):
        x=torch.randn(109,3,3)*20+torch.tensor([110.,-230.,170.])
        source=pool_ca(x);own=source-x[:,1].mean(0)
        for gauge in GAUGES:
            c,_=coarse_prompt(x,own,gauge)
            distance=center_distances(c)
            torch.testing.assert_close(distance.diagonal(),torch.zeros(28,dtype=torch.float64),atol=0,rtol=0)
            torch.testing.assert_close(distance,center_distances(source),atol=2e-5,rtol=2e-5)

    def test_translation_does_not_change_center_prompt(self):
        for gauge in GAUGES:
            a,_=coarse_prompt(self.x,self.own,gauge)
            b,_=coarse_prompt(self.x+50,self.own,gauge)
            torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)

    def test_undefined_first_residue_frame_is_explicit_failure(self):
        x=self.x.clone();x[0]=0
        with self.assertRaises(FrameGeometryError):coarse_prompt(x,self.own,"first_residue")

    def test_encoder_fine_can_be_decoded_after_a_proper_prompt_rotation(self):
        codec=MultiscaleStructureCodec(torch.nn.Linear(3,3),StableBlockFrameFineTokenizer("local"),"centers3")
        c,f=codec.encode(self.x);base=codec.decode(c,f)
        for gauge in GAUGES:
            cc,r=coarse_prompt(self.x,self.own,gauge)
            value=codec.decode(cc,f)
            torch.testing.assert_close(value,base@r,atol=2e-4,rtol=2e-5)
            torch.testing.assert_close(pool_ca(value),cc,atol=2e-5,rtol=2e-5)


if __name__=="__main__":unittest.main()
