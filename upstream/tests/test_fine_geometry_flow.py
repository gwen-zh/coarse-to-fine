from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from fine_geometry_flow import FixedCoarseFineFlow,sample_training_batch,draw_seed,fine_flow_loss,fine_rollout_geometry_loss
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow,component_seed
from joint_sequence_predictor import encode_aa
from block_stable_frame_tokenizer import StableBlockFrameFineTokenizer
from sequence_multiscale_ensemble import MultiscaleStructureCodec


def case(name,split="train"):
    return dict(id=name,split=split,sequence="A"*13,target=torch.zeros(13,3,3),coarse=torch.zeros(4,3))


def make_flow():
    torch.manual_seed(24)
    return HierarchicalEnsembleFlow(torch.zeros(3),torch.ones(3),torch.zeros(8),torch.ones(8),
        width=32,layers=1,coarse_dim=3)


class FineGeometryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_branch_separation_preserves_coarse_after_fine_update(self):
        coarse=make_flow();fine=make_flow();model=FixedCoarseFineFlow(coarse,fine)
        aa=encode_aa("A"*13);before=model.sample_coarse(aa,77)
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=.01)
        model.train();self.assertFalse(model.coarse_model.training)
        opt.zero_grad();model.fine_model.fine_velocity(aa,torch.ones(13,8),.7,torch.ones(4,3)).square().mean().backward();opt.step()
        torch.testing.assert_close(before,model.sample_coarse(aa,77),rtol=0,atol=0)
        self.assertTrue(all(v.grad is None for v in model.coarse_model.parameters()))

    def test_initial_wrapper_replays_both_parent_samplers(self):
        a=make_flow();b=make_flow();model=FixedCoarseFineFlow(a,b).eval();aa=encode_aa("A"*13)
        c=a.sample_coarse(aa,35)
        torch.testing.assert_close(c,model.sample_coarse(aa,35),rtol=0,atol=0)
        torch.testing.assert_close(a.sample_fine(aa,c,65),model.sample_fine(aa,c,65),rtol=0,atol=0)

    def test_paired_batches_are_reproducible_train_only_and_three_to_one(self):
        groups=dict(PDB_train=[case("PDB:a"),case("PDB:b")],train={"m":[case("MD:a"),case("MD:b")]},validation=None)
        a=sample_training_batch(groups,np.random.default_rng(72));b=sample_training_batch(groups,np.random.default_rng(72))
        self.assertEqual([x["id"] for x in a],[x["id"] for x in b])
        self.assertTrue(all(x["id"].startswith("PDB") for x in a[:3]))
        self.assertTrue(a[3]["id"].startswith("MD"))
        groups["train"]["m"]=[case("bad",split="validation")]
        with self.assertRaises(ValueError):sample_training_batch(groups,np.random.default_rng(72))

    def test_loss_draws_match_and_geometry_reaches_only_fine_parameters(self):
        codec=MultiscaleStructureCodec(torch.nn.Identity(),StableBlockFrameFineTokenizer("local"),"centers3")
        model=FixedCoarseFineFlow(make_flow(),make_flow())
        aa=encode_aa("A"*13)
        coarse=torch.tensor([[0.,0.,0.],[4.,1.,0.],[8.,3.,1.],[11.,3.,2.]])
        fine=torch.randn(13,8).tanh()
        state=torch.get_rng_state().clone()
        loss0,terms0=fine_flow_loss(model,aa,coarse,fine,torch.Generator().manual_seed(90))
        loss1,terms1=fine_flow_loss(model,aa,coarse,fine,torch.Generator().manual_seed(90))
        self.assertEqual(terms0["draw_sha256"],terms1["draw_sha256"])
        self.assertEqual(terms0["flow"],terms1["flow"])
        geometry,_,_,_=fine_rollout_geometry_loss(model,codec,aa,coarse,19,nfe=4)
        grad=torch.autograd.grad(geometry,model.fine_model.fine_head[-1].weight)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_zero_geometry_weight_is_the_plain_flow_objective(self):
        codec=MultiscaleStructureCodec(torch.nn.Identity(),StableBlockFrameFineTokenizer("local"),"centers3")
        model=FixedCoarseFineFlow(make_flow(),make_flow())
        aa=encode_aa("A"*13);coarse=torch.randn(4,3);fine=torch.randn(13,8).tanh()
        a,_=fine_flow_loss(model,aa,coarse,fine,torch.Generator().manual_seed(15))
        geometry,_,_,_=fine_rollout_geometry_loss(model,codec,aa,coarse,29,nfe=4)
        a=a+0.*geometry
        rng=torch.Generator().manual_seed(15);f=model.fine_model.normalize_fine(fine)
        noise=torch.randn(f.shape,generator=rng);t=torch.rand((),generator=rng)
        v=model.fine_model.fine_velocity(aa,(1-t)*noise+t*f,t,model.fine_model.normalize_coarse(coarse))
        b=(v-(f-noise)).square().mean()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,model.fine_model.fine_head[-1].weight)[0]
        gb=torch.autograd.grad(b,model.fine_model.fine_head[-1].weight)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-6,atol=1e-6)

    def test_checkpointed_full_rollout_matches_direct_sampler_and_gradients(self):
        codec=MultiscaleStructureCodec(torch.nn.Identity(),StableBlockFrameFineTokenizer("local"),"centers3")
        model=FixedCoarseFineFlow(make_flow(),make_flow()).eval()
        aa=encode_aa("A"*13);coarse=torch.randn(4,3)
        a,da,fa,ba=fine_rollout_geometry_loss(model,codec,aa,coarse,37,nfe=4,checkpoint_velocity=True)
        b,db,fb,bb=fine_rollout_geometry_loss(model,codec,aa,coarse,37,nfe=4,checkpoint_velocity=False)
        expected=model.sample_fine(aa,coarse,component_seed(37,"fine"),4)
        torch.testing.assert_close(fa,expected,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(fa,fb,rtol=0,atol=0)
        self.assertEqual(da,db)
        params=model.trainable_parameters()
        ga=torch.autograd.grad(a,params,allow_unused=True)
        gb=torch.autograd.grad(b,params,allow_unused=True)
        for x,y in zip(ga,gb):
            self.assertEqual(x is None,y is None)
            if x is not None:torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)

    def test_draw_seed_is_slot_and_step_specific(self):
        values={draw_seed(2026,step,slot) for step in range(3) for slot in range(4)}
        self.assertEqual(len(values),12);self.assertEqual(draw_seed(2026,2,3),draw_seed(2026,2,3))


if __name__=="__main__":unittest.main()
