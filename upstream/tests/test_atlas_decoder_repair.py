import copy
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from atlas_decoder_repair import enable_fine_decoder_training, frozen_structure_state, assert_frozen_structure, repair_loss
from hierarchical_rigid_vq import HierarchicalRigidVQ, BB_LOCAL
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_latent_contract import pool_ca


class DecoderRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(93)
        model=HierarchicalRigidVQ(width=32,layers=1,coarse_size=8,fine_size=16)
        model.fine_book.initialize(torch.randn(64,8),2)
        self.tokenizer=HybridTokenizer(model)
        self.c=torch.randn(3,16);self.ids=torch.randint(16,(10,))
    def test_enabling_repair_is_exact_epoch_zero_and_limits_trainable_parameters(self):
        before=self.tokenizer.decode(self.c,self.ids).detach().clone()
        enable_fine_decoder_training(self.tokenizer)
        torch.testing.assert_close(before,self.tokenizer.decode(self.c,self.ids),atol=0,rtol=0)
        self.assertTrue(all(name.startswith('model.fine_decoder.') for name,p in self.tokenizer.named_parameters() if p.requires_grad))
        self.assertFalse(self.tokenizer.training);self.assertFalse(self.tokenizer.model.fine_book.training)
    def test_step_changes_only_fine_decoder_and_keeps_rigid_atoms_and_centers(self):
        before=frozen_structure_state(self.tokenizer)
        params=enable_fine_decoder_training(self.tokenizer);old=[p.detach().clone() for p in params]
        opt=torch.optim.AdamW(params,lr=1e-3)
        target=self.tokenizer.decode(self.c,self.ids).detach()+torch.randn(10,3,3)*.1
        pred=self.tokenizer.decode(self.c,self.ids);loss,_=repair_loss(pred,target,True);loss.backward();opt.step()
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(old,params)))
        assert_frozen_structure(self.tokenizer,before)
        result=self.tokenizer.decode(self.c,self.ids)
        torch.testing.assert_close(pool_ca(result),self.tokenizer.coarse_centers(self.c,10),atol=3e-5,rtol=1e-6)
        ideal=torch.tensor(BB_LOCAL)
        for a,b in [(0,1),(1,2),(0,2)]:
            torch.testing.assert_close((result[:,a]-result[:,b]).norm(dim=-1),(ideal[a]-ideal[b]).norm().expand(10),atol=1e-5,rtol=1e-5)
    def test_geometry_loss_has_nonzero_gradient_and_target_is_detached(self):
        pred=self.tokenizer.decode(self.c,self.ids).detach().requires_grad_();target=(pred.detach()+torch.randn_like(pred)*.2).requires_grad_()
        a,terms=repair_loss(pred,target,False);b,_=repair_loss(pred,target,True)
        torch.testing.assert_close(a,terms['coordinate']);self.assertGreater(float(b),float(a))
        (b-a).backward();self.assertGreater(float(pred.grad.norm()),0);self.assertIsNone(target.grad)
    def test_boundary_term_uses_only_cross_block_links(self):
        target=torch.randn(10,3,3);pred=target.clone();pred[4,0]+=torch.tensor([1.,2.,3.])
        _,terms=repair_loss(pred,target,True)
        error=((pred[:-1,2]-pred[1:,0]).norm(dim=-1)-(target[:-1,2]-target[1:,0]).norm(dim=-1)).square()
        torch.testing.assert_close(terms['boundary_cn'],error[torch.tensor([3,7])].mean())
    def test_frozen_state_check_detects_accidental_encoder_change(self):
        before=frozen_structure_state(self.tokenizer)
        with torch.no_grad():next(self.tokenizer.model.coarse_encoder.parameters()).add_(.01)
        with self.assertRaises(RuntimeError):assert_frozen_structure(self.tokenizer,before)
    def test_invalid_or_nonfinite_chain_rejected(self):
        with self.assertRaises(ValueError):repair_loss(torch.zeros(4,3,3),torch.zeros(4,3,3),True)
        with self.assertRaises(ValueError):repair_loss(torch.full((10,3,3),float('nan')),torch.zeros(10,3,3),True)


if __name__=='__main__':unittest.main()
