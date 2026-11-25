from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from par_chiral_velocity import (VelocityResidual, ResidualFlow, evaluation_keys, update,
                                paired_identity, DRAW_OFFSET)
from par_signed_volume_replication import evaluation_noise_keys
from par_coarse_capacity import evaluation_keys as old_keys


class ChiralVelocityTests(unittest.TestCase):
    def make(self, kind):
        torch.manual_seed(11)
        return VelocityResidual(kind, condition_dim=7, width=12).double()

    def inputs(self):
        g = torch.Generator().manual_seed(12)
        return (torch.randn(2, 9, 3, generator=g, dtype=torch.float64),
                torch.randn(2, 9, 7, generator=g, dtype=torch.float64),
                torch.tensor([.1, .7], dtype=torch.float64),
                torch.randn(2, 9, 3, generator=g, dtype=torch.float64))

    def test_equal_parameters_initial_weights_and_exact_zero_function(self):
        a, b = self.make('polar'), self.make('axial')
        self.assertEqual(sum(v.numel() for v in a.parameters()), sum(v.numel() for v in b.parameters()))
        for key in a.state_dict():
            torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        x, z, t, sc = self.inputs()
        for m in [a, b]:
            torch.testing.assert_close(m(x, z, t, sc), torch.zeros_like(x), rtol=0, atol=0)
            (m(x, z, t, sc) - x).square().mean().backward()
            self.assertGreater(float(m.output.weight.grad.norm()), 0.)

    def test_nonzero_residual_rotation_translation_and_parity_laws(self):
        x, z, t, sc = self.inputs()
        rotation = x.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        reflection = x.new_tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        translation = x.new_tensor([3., -8., 2.])
        for kind in ['polar', 'axial']:
            m = self.make(kind)
            with torch.no_grad():
                m.output.weight.normal_(0, .05)
            for context in [None, sc]:
                out = m(x, z, t, context)
                self.assertGreater(float(out.norm()), 1e-5)
                transformed = None if context is None else context@rotation + translation
                torch.testing.assert_close(m(x@rotation+translation, z, t, transformed), out@rotation,
                                           atol=1e-12, rtol=1e-10)
                mirrored = None if context is None else context@reflection + translation
                sign = -1 if kind == 'axial' else 1
                torch.testing.assert_close(m(x@reflection+translation, z, t, mirrored), sign*out@reflection,
                                           atol=1e-12, rtol=1e-10)

    def test_degenerate_chains_have_finite_input_and_parameter_gradients(self):
        for kind in ['polar', 'axial']:
            m = self.make(kind)
            with torch.no_grad():m.output.weight.fill_(.03)
            for straight in [False, True]:
                x = torch.zeros(1, 5, 3, dtype=torch.float64)
                if straight:x[0, :, 0] = torch.arange(5, dtype=torch.float64)
                x.requires_grad_(True)
                z = torch.ones(1, 5, 7, dtype=torch.float64, requires_grad=True)
                out = m(x, z, .5, x.clone())
                out.square().sum().backward()
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(z.grad).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_wrapper_preserves_native_outputs_and_freezes_only_the_given_base(self):
        class Base(nn.Module):
            def __init__(self):
                super().__init__();self.linear=nn.Linear(3,3).double().requires_grad_(False)
            def forward(self, batch):return dict(coors_pred=self.linear(batch['x_t']), auxiliary='retained')
        x, z, t, sc = self.inputs()
        base = Base();m = ResidualFlow(base, self.make('axial'))
        batch=dict(x_t=x,z_cond=z,t=t,x_sc=sc,mask=torch.ones(x.shape[:2],dtype=torch.bool))
        expected=base(batch);observed=m(batch)
        self.assertEqual(observed['auxiliary'], expected['auxiliary'])
        torch.testing.assert_close(observed['coors_pred'],expected['coors_pred'],atol=0,rtol=0)
        observed['coors_pred'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(float(m.residual.output.weight.grad.norm()),0.)
        with self.assertRaises(ValueError):m.residual(x,z,t,sc,torch.zeros(x.shape[:2],dtype=torch.bool))

    def test_new_keys_and_draw_offset_do_not_reuse_completed_studies(self):
        keys=evaluation_keys()
        self.assertEqual(len(keys),64);self.assertEqual(len(set(keys)),64)
        self.assertFalse(set(keys)&set(old_keys(2026)))
        self.assertFalse(set(keys)&set(evaluation_noise_keys()))
        row=dict(update=6145,loss=.2,lr=0.,gradient_norm=.5,batch=[dict(time=.4,case_id='train')])
        fake=SimpleNamespace(atom_diffusion=SimpleNamespace(nn=nn.Identity()))
        with patch('par_chiral_velocity.native_update',return_value=row) as call:
            result=update(fake,None,None,None,2029,0)
        self.assertEqual(call.call_args.args[6],DRAW_OFFSET)
        self.assertEqual(call.call_args.args[7]['lr'],call.call_args.args[7]['min_lr'])
        self.assertEqual(result['update'],1);self.assertEqual(result['draw_step'],6144)
        self.assertEqual(result['total_adapter_updates'],4097)
        self.assertEqual(paired_identity(result)['batch'][0]['time'],.4)


if __name__ == '__main__':unittest.main()
