import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from position_aligned_fine import ARMS, PositionedFineModel, denoise_loss, supported_cross_entropy
from hierarchical_pipeline_io import save_checkpoint, restore_checkpoint


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(23)
        self.counts = torch.zeros(8,4,dtype=torch.long)
        for r in range(4): self.counts[r,r]=1;self.counts[r+4,r]=1
        self.h=torch.randn(13,12);self.c=torch.tanh(torch.randn(4,16));self.f=torch.arange(13)%4

    def model(self,arm):
        torch.manual_seed(25)
        return PositionedFineModel(torch.zeros(16),torch.ones(16),self.counts,
                                    condition_dim=12,coarse_dim=16,width=32,layers=1,
                                    fine_size=8,dropout=.1,arm=arm)

    def test_zero_initialized_adapters_preserve_initial_function(self):
        outputs=[self.model(arm).eval().fine(self.h,self.c,self.f,.5) for arm in ARMS]
        for output in outputs[1:]:self.assertTrue(torch.equal(outputs[0],output))

    def test_broadcast_maps_only_corresponding_block_without_attention(self):
        model=self.model('position_coarse').eval()
        model.fine.layers=torch.nn.Identity();model.fine.head=torch.nn.Identity()
        model.fine.local_coarse_gate.data.fill_(1.)
        a=model.fine(self.h,self.c,self.f,.5)
        changed=self.c.clone();changed[1]+=.3
        b=model.fine(self.h,changed,self.f,.5)
        self.assertTrue(torch.equal(a[:4],b[:4]));self.assertTrue(torch.equal(a[8:],b[8:]))
        self.assertFalse(torch.equal(a[4:8],b[4:8]))

    def test_smoothing_ignores_illegal_classes_and_has_finite_gradients(self):
        logits=torch.randn(3,4,requires_grad=True)
        target=torch.tensor([0,1,0]);support=torch.tensor([[1,1,0,0]]*3,dtype=torch.bool)
        loss,_=supported_cross_entropy(logits,target,support,.05)
        expected=torch.nn.functional.cross_entropy(logits[:,:2],target,label_smoothing=.05)
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[:,2:].abs().sum()),0.)

    def test_illegal_target_is_rejected_instead_of_dropped(self):
        with self.assertRaises(ValueError):supported_cross_entropy(torch.randn(1,4),torch.tensor([3]),torch.tensor([[1,1,0,0]],dtype=torch.bool),.05)

    def test_role_and_local_coarse_gate_receive_gradients(self):
        m=self.model('position_coarse').train()
        loss,_=denoise_loss(m,self.h,self.c,self.f,rate=1.);loss.backward()
        self.assertGreater(float(m.fine.block_position.weight.grad.norm()),0.)
        self.assertGreater(float(m.fine.local_coarse_gate.grad.abs()),0.)
        self.assertFalse(m.position_counts.requires_grad)

    def test_all_arms_pair_mask_condition_and_dropout_randomness(self):
        inputs=[];states=[];losses=[]
        for arm in ARMS:
            m=self.model(arm).train();captured=[]
            def hook(module,args):captured.append((args[1].clone(),args[2].clone(),args[3]))
            handle=m.fine.register_forward_pre_hook(hook)
            torch.manual_seed(55);loss,_=denoise_loss(m,self.h,self.c,self.f);loss.backward()
            inputs.append(captured[0]);states.append(torch.get_rng_state().clone());losses.append(float(loss.detach()));handle.remove()
        for i in [1,2]:
            self.assertTrue(torch.equal(inputs[0][0],inputs[i][0]));self.assertTrue(torch.equal(inputs[0][1],inputs[i][1]))
            self.assertEqual(inputs[0][2],inputs[i][2]);self.assertTrue(torch.equal(states[0],states[i]));self.assertEqual(losses[0],losses[i])

    def test_supported_sampling_is_seeded_for_all_arms(self):
        for arm in ARMS:
            m=self.model(arm).eval();a=m.sample(self.h,self.c,19,4);b=m.sample(self.h,self.c,19,4)
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a%4,torch.arange(13)%4))

    def test_checkpoint_replays_dropout_and_adam_next_update(self):
        for arm in ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as directory:
                m=self.model(arm).train();opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
                def step(model,optimizer):
                    optimizer.zero_grad(set_to_none=True);loss,_=denoise_loss(model,self.h,self.c,self.f);loss.backward();optimizer.step();return float(loss.detach())
                step(m,opt);path=Path(directory)/'state.pt';save_checkpoint(path,m,opt,np.random.default_rng(2))
                expected=step(m,opt);weights={k:v.clone() for k,v in m.state_dict().items()};adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
                restored=self.model(arm).train();opt2=torch.optim.AdamW(restored.parameters(),lr=3e-4)
                restore_checkpoint(path,restored,opt2,np.random.default_rng(5),torch.device('cpu'))
                self.assertEqual(expected,step(restored,opt2));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for k,v in weights.items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))
                for k,values in adam['state'].items():
                    for key,value in values.items():self.assertTrue(torch.equal(value,opt2.state_dict()['state'][k][key]))


if __name__=='__main__':unittest.main()
