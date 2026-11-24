import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from coarse_relative_geometry import CoarseRelativeGeometry, GeometryCorrectedEnsemble, update
from coarse_full_cohort_flow import batch_rng, identity_record, sample_training_batch
from compact_training_trace import CompactTrainingTrace, restore_trace_prefix
from hierarchical_ensemble_flow import HierarchicalEnsembleFlow
from hierarchical_pipeline_io import restore_checkpoint, save_checkpoint
from joint_sequence_predictor import encode_aa


def branch():
    torch.manual_seed(73)
    return HierarchicalEnsembleFlow(torch.tensor([.1,-.2,.3]),torch.tensor([1.,2.,3.]),
        torch.zeros(8),torch.ones(8),width=32,layers=1,coarse_dim=3)


def model(kind):
    base=branch();fine=copy.deepcopy(base)
    torch.manual_seed(81)
    return GeometryCorrectedEnsemble(CoarseRelativeGeometry(base,kind,hidden=16),fine)


def case(name,offset,split="train"):
    return dict(id=name,protein=name,split=split,sequence="ACDEFGHIKLMNPQRSTVWYA",target=torch.zeros(21,3,3),
        coarse=torch.randn((6,3),generator=torch.Generator().manual_seed(83))+offset)


def groups():return dict(PDB_train=[case("p1",.1),case("p2",.2)],train={"m1":[case("m1",.3)]})


class RelativeGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.aa=encode_aa("ACDEFGHIKLMNPQRSTVWYA")
        self.p=dict(updates=4,lr=3e-4,min_lr=3e-5,gradient_clip=1.,correction_penalty=.01)

    def test_zero_initialization_replays_both_parent_samplers(self):
        for kind in ["cartesian","radial"]:
            m=model(kind).eval();c=m.coarse_model.base.sample_coarse(self.aa,6,16)
            torch.testing.assert_close(m.sample_coarse(self.aa,6,16),c,atol=0,rtol=0)
            torch.testing.assert_close(m.sample_fine(self.aa,c,7,16),m.fine_model.sample_fine(self.aa,c,7,16),atol=0,rtol=0)

    def test_equal_parameter_count_and_exact_initial_adapter_weights(self):
        a,b=model("cartesian"),model("radial")
        self.assertEqual(sum(v.numel() for v in a.trainable_parameters()),sum(v.numel() for v in b.trainable_parameters()))
        for k,v in a.coarse_model.adapter.state_dict().items():torch.testing.assert_close(v,b.coarse_model.adapter.state_dict()[k],atol=0,rtol=0)

    def test_radial_correction_rotation_translation_equivariance_and_centering(self):
        m=model("radial").eval();b=m.coarse_model
        with torch.no_grad():b.adapter[-1].weight.fill_(.03);b.adapter[-1].bias.fill_(.01)
        x=torch.randn((6,3),generator=torch.Generator().manual_seed(21))
        rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        v=b.correction(self.aa,b.normalize_coarse(x),.7)*b.base.coarse_std
        changed=b.correction(self.aa,b.normalize_coarse(x@rotation+4.),.7)*b.base.coarse_std
        torch.testing.assert_close(changed,v@rotation,atol=2e-6,rtol=2e-5)
        counts=torch.tensor([4.,4.,4.,4.,4.,1.])
        torch.testing.assert_close((v*counts[:,None]).sum(0),torch.zeros(3),atol=1e-6,rtol=0)
        self.assertEqual(float(b.correction(self.aa,b.normalize_coarse(x),0.).abs().max()),0.)

    def test_only_adapter_changes_after_actual_update_and_fine_remains_exact(self):
        m=model("radial");base=copy.deepcopy(m.coarse_model.base.state_dict());fine=copy.deepcopy(m.fine_model.state_dict())
        target=case("x",.3)["coarse"];before=m.sample_fine(self.aa,target,6,16)
        opt=torch.optim.AdamW(m.trainable_parameters(),lr=self.p["lr"])
        row=update(m,opt,sample_training_batch(groups(),batch_rng(2026)),2026,0,self.p)
        self.assertGreater(row["gradient_norm"],0.)
        self.assertGreater(float(m.coarse_model.adapter[-1].weight.abs().sum()),0.)
        for k,v in base.items():torch.testing.assert_close(v,m.coarse_model.base.state_dict()[k],atol=0,rtol=0)
        for k,v in fine.items():torch.testing.assert_close(v,m.fine_model.state_dict()[k],atol=0,rtol=0)
        torch.testing.assert_close(before,m.sample_fine(self.aa,target,6,16),atol=0,rtol=0)
        self.assertTrue(all(v.grad is None for v in m.coarse_model.base.parameters()))
        self.assertFalse(m.coarse_model.base.training);self.assertFalse(m.fine_model.training)

    def test_arms_have_identical_training_draws_and_reject_development(self):
        identities=[]
        for kind in ["cartesian","radial"]:
            m=model(kind);opt=torch.optim.AdamW(m.trainable_parameters(),lr=self.p["lr"])
            batch=sample_training_batch(groups(),batch_rng(2026))
            identities.append(identity_record(update(m,opt,batch,2026,0,self.p)))
            batch[0]=case("dev",.3,"validation")
            with self.assertRaises(ValueError):update(m,opt,batch,2026,1,self.p)
        self.assertEqual(identities[0],identities[1])

    def test_checkpoint_resume_preserves_adapter_optimizer_draws_and_trace(self):
        a=model("radial");oa=torch.optim.AdamW(a.trainable_parameters(),lr=self.p["lr"]);rng=batch_rng(2026);rows=[]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);writer=CompactTrainingTrace(4)
            for step in range(4):
                rows.append(update(a,oa,sample_training_batch(groups(),rng),2026,step,self.p))
                if step==1:save_checkpoint(folder/"last.pt",a,oa,rng,updates=2)
                if step==2:writer(folder/"trace.json",rows)
            b=model("radial");ob=torch.optim.AdamW(b.trainable_parameters(),lr=self.p["lr"]);rr=batch_rng(0)
            restore_checkpoint(folder/"last.pt",b,ob,rr,torch.device("cpu"));restore_trace_prefix(folder,2)
            replay=[json.loads(s) for s in (folder/"trace.jsonl").read_text().splitlines()]
            for step in range(2,4):replay.append(update(b,ob,sample_training_batch(groups(),rr),2026,step,self.p))
            self.assertEqual(rows,replay)
            for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)

    def test_training_feature_bank_assignment_reaches_frozen_base(self):
        m=model("radial");bank={"sequence":torch.ones(3,2)}
        m.coarse_model.feature_bank=bank
        self.assertIs(m.coarse_model.base.feature_bank,bank)

    def test_freezing_parent_keeps_its_input_state_derivative(self):
        m=model("radial").eval();state=torch.randn((6,3),generator=torch.Generator().manual_seed(61),requires_grad=True)
        reference=m.coarse_model.base.coarse_velocity(self.aa,state,.7).square().sum()
        expected=torch.autograd.grad(reference,state)[0]
        actual=m.coarse_model.coarse_velocity(self.aa,state,.7).square().sum()
        gradient=torch.autograd.grad(actual,state)[0]
        self.assertGreater(float(expected.abs().sum()),0.)
        torch.testing.assert_close(gradient,expected,atol=1e-7,rtol=1e-5)


if __name__=="__main__":unittest.main()
