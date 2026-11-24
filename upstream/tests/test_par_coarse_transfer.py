from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from par_coarse_backbone import attach_sequence_conditioner,ar_condition
from par_coarse_transfer import conditioning_features,adapter_view,draw,update,identity,frozen_versions,assert_frozen
from hierarchical_pipeline_io import save_checkpoint,restore_checkpoint


class Base(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(3,32)
    def forward(self,batch):return self.linear(batch['x_input'])


class AR(nn.Module):
    def __init__(self):super().__init__();self.init_repr_factory=Base();self.transformer_layers=nn.ModuleList()
    def forward(self,batch):return {'cond':self.init_repr_factory(batch)}


class Flow(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(39,3)
    def forward(self,batch):
        x=batch['x_t'];t=batch['t'][:,None,None].expand(len(x),x.shape[1],1)
        sc=batch.get('x_sc',torch.zeros_like(x))
        return {'coors_pred':self.linear(torch.cat([batch['z_cond'],x,t,sc],-1))}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.nn=AR();self.atom_diffusion=nn.Module();self.atom_diffusion.nn=Flow()
        self.cfg_exp=SimpleNamespace(model=SimpleNamespace(nn=SimpleNamespace(token_dim=32)))


def model():
    torch.manual_seed(81);m=Tiny().eval().requires_grad_(False)
    factory=attach_sequence_conditioner(m,width=4)
    return m,factory,adapter_view(factory)


class PARCoarseTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g=torch.Generator().manual_seed(73)
        self.h=torch.randn(13,1536,generator=g)
        self.case=dict(id='p',protein='p',split='train',sequence='ACDEFGHIKLMNP',coarse=torch.randn(4,3,generator=g))
        self.p=dict(updates=4,lr=1e-4,min_lr=1e-5,gradient_clip=1.)

    def test_draws_are_paired_finite_centered_and_target_sensitive(self):
        a=draw(self.case['coarse'],2026,0,0);b=draw(self.case['coarse'],2026,0,0)
        self.assertEqual(a[-1],b[-1]);torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
        self.assertTrue(torch.isfinite(a[0]).all());self.assertGreaterEqual(float(a[2]),0);self.assertLess(float(a[2]),1)
        torch.testing.assert_close(a[0].mean(0),torch.zeros(3),atol=1e-6,rtol=0)
        self.assertNotEqual(a[-1],draw(self.case['coarse']*2,2026,0,0)[-1])

    def test_information_control_is_order_insensitive_and_sequence_arm_is_not(self):
        m,factory,view=model()
        with torch.no_grad():factory.block_projection.weight.normal_(0,.1)
        _,a=ar_condition(m,conditioning_features(self.h,'position_only'),13)
        _,b=ar_condition(m,conditioning_features(self.h.flip(0),'position_only'),13)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        _,c=ar_condition(m,conditioning_features(self.h,'sequence'),13)
        _,d=ar_condition(m,conditioning_features(self.h.flip(0),'sequence'),13)
        self.assertFalse(torch.equal(c,d))

    def test_update_is_decoder_free_and_changes_only_sequence_adapter(self):
        m,factory,view=model();expected=frozen_versions(m);before={k:v.clone() for k,v in view.state_dict().items()}
        opt=torch.optim.AdamW(view.parameters(),lr=1e-4)
        row=update(m,opt,[self.case]*4,lambda seq:self.h,'sequence',2026,0,self.p)
        assert_frozen(m,expected);self.assertGreater(row['gradient_norm'],0)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in view.state_dict().items()))
        self.assertTrue(all(not k.startswith('base.') for k in view.state_dict()))

    def test_development_cannot_enter_optimizer(self):
        m,factory,view=model();opt=torch.optim.AdamW(view.parameters(),lr=1e-4)
        case=dict(self.case,split='validation')
        with self.assertRaisesRegex(ValueError,'Development'):
            update(m,opt,[case]*4,lambda seq:self.h,'sequence',2026,0,self.p)

    def test_adapter_only_checkpoint_restores_live_model_and_optimizer(self):
        full,_,view=model();opt=torch.optim.AdamW(view.parameters(),lr=1e-4)
        for step in range(4):update(full,opt,[self.case]*4,lambda seq:self.h,'sequence',2026,step,self.p)
        part,_,part_view=model();opt2=torch.optim.AdamW(part_view.parameters(),lr=1e-4)
        for step in range(2):update(part,opt2,[self.case]*4,lambda seq:self.h,'sequence',2026,step,self.p)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt';rng=np.random.default_rng(83)
            save_checkpoint(path,part_view,opt2,rng,updates=2)
            resumed,_,resumed_view=model();opt3=torch.optim.AdamW(resumed_view.parameters(),lr=1e-4)
            restore_checkpoint(path,resumed_view,opt3,rng,torch.device('cpu'))
            for step in range(2,4):update(resumed,opt3,[self.case]*4,lambda seq:self.h,'sequence',2026,step,self.p)
            for k,v in full.state_dict().items():torch.testing.assert_close(v,resumed.state_dict()[k],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
