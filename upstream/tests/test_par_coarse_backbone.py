from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from par_coarse_backbone import SequenceConditionedFactory,coarse_batch,guard_unused_openfold,flow_batch


class Base(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(3,32)
    def forward(self,batch):return self.linear(batch["x_input"])


class PARCoarseBackboneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(79)

    def factory(self):return SequenceConditionedFactory(Base(),feature_dim=8,width=4,output_dim=32)

    def test_zero_sequence_branch_preserves_nonzero_base_exactly(self):
        factory=self.factory();batch=coarse_batch(13,4,'cpu');factory.features=torch.randn(13,8)
        torch.testing.assert_close(factory(batch),factory.base(batch),atol=0,rtol=0)
        factory.features=None
        torch.testing.assert_close(factory(batch),factory.base(batch),atol=0,rtol=0)

    def test_update_reaches_new_head_but_not_base_or_frozen_features(self):
        factory=self.factory();batch=coarse_batch(13,4,'cpu');features=torch.randn(13,8,requires_grad=True)
        factory.features=features;factory(batch).square().mean().backward()
        self.assertGreater(float(factory.block_projection.weight.grad.norm()),0)
        self.assertTrue(all(v.grad is None and not v.requires_grad for v in factory.base.parameters()))
        self.assertIsNone(features.grad)

    def test_order_is_preserved_and_partial_block_has_correct_shape(self):
        factory=self.factory();features=torch.randn(13,8)
        with torch.no_grad():factory.block_projection.weight.normal_(0,.1)
        a=factory.aa_blocks(features);reordered=features.clone();reordered[:4]=features[:4].flip(0)
        b=factory.aa_blocks(reordered)
        self.assertEqual(a.shape,(1,4,32));self.assertFalse(torch.equal(a[:,0],b[:,0]))
        torch.testing.assert_close(a[:,1:],b[:,1:],atol=0,rtol=0)

    def test_incompatible_length_and_active_openfold_branch_fail_explicitly(self):
        factory=self.factory();factory.features=torch.randn(13,8)
        with self.assertRaises(ValueError):factory(coarse_batch(13,3,'cpu'))
        cfg=SimpleNamespace(model=SimpleNamespace(nn={'use_attn_pair_bias':True},atom_diff=SimpleNamespace(nn={})))
        with self.assertRaisesRegex(RuntimeError,'no-pair'):guard_unused_openfold(cfg)

    def test_batched_same_sequence_condition_preserves_masks_and_time_shape(self):
        factory=self.factory();factory.features=torch.randn(13,8)
        batch=coarse_batch(13,4,'cpu',batch_size=3)
        with torch.no_grad():factory.block_projection.weight.normal_(0,.1)
        result=factory(batch)
        self.assertEqual(result.shape,(3,4,32))
        torch.testing.assert_close(result[0],result[1],atol=0,rtol=0)
        b=flow_batch(batch,result,torch.randn(3,4,3),.4)
        self.assertEqual(b['t'].shape,(3,))
        self.assertEqual(b['int_pos_idx'].shape,(3,4))


if __name__=='__main__':unittest.main()
