import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from coarse_smallset import probe_loss,predict,state_hash
from mixed_hierarchical_generator import MixedHierarchicalGenerator,HybridTokenizer
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_pipeline_io import save_checkpoint,restore_checkpoint

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.h=torch.randn(13,12);self.c=torch.tanh(torch.randn(4,16))
        self.decoder=HybridTokenizer(HierarchicalRigidVQ(width=16,layers=1,fine_size=8))
    def model(self):
        m=MixedHierarchicalGenerator(torch.zeros(16),torch.ones(16),condition_dim=12,width=32,coarse_layers=1,fine_layers=1,fine_size=8)
        m.fine.requires_grad_(False);return m
    def test_flow_loss_is_exact_existing_objective(self):
        m=self.model();torch.manual_seed(37);expected,_=m.coarse_loss(self.h,self.c,self.decoder,.1)
        torch.manual_seed(37);actual,_=probe_loss(m,self.decoder,self.h,self.c,'flow',.1)
        self.assertTrue(torch.equal(actual,expected))
    def test_flow_sampler_is_exact_existing_path(self):
        m=self.model().eval()
        self.assertTrue(torch.equal(predict(m,self.h,'flow',13,16),m.sample_coarse(self.h,13,16)))
    def test_regression_ignores_sampling_seed(self):
        m=self.model().eval()
        self.assertTrue(torch.equal(predict(m,self.h,'regression',13),predict(m,self.h,'regression',999)))
    def test_losses_pair_noise_time_and_do_not_update_frozen_structure(self):
        m=self.model();fixed=state_hash(self.decoder);hashes=[];states=[]
        for arm in ['flow','regression']:
            m.zero_grad(set_to_none=True);torch.manual_seed(57)
            loss,r=probe_loss(m,self.decoder,self.h,self.c,arm);hashes.append(r['draw_sha256']);states.append(torch.get_rng_state().clone())
            loss.backward();self.assertGreater(float(m.coarse_flow.output.weight.grad.norm()),0.)
            self.assertTrue(all(v.grad is None for v in self.decoder.parameters()))
            self.assertTrue(all(v.grad is None for v in m.fine.parameters()))
        self.assertEqual(hashes[0],hashes[1]);self.assertTrue(torch.equal(states[0],states[1]));self.assertEqual(fixed,state_hash(self.decoder))
    def test_exact_next_update_resume_for_both_objectives(self):
        for arm in ['flow','regression']:
            torch.manual_seed(24);m=self.model();opt=torch.optim.AdamW(m.coarse_flow.parameters(),lr=1e-3)
            def step(a,o):
                a.train();o.zero_grad(set_to_none=True);loss,r=probe_loss(a,self.decoder,self.h,self.c,arm);loss.backward();o.step();return float(loss),r['draw_sha256']
            step(m,opt)
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/'last.pt';save_checkpoint(path,m,opt,np.random.default_rng(12))
                expected=step(m,opt);weights=copy.deepcopy(m.state_dict());adam=copy.deepcopy(opt.state_dict());trng=torch.get_rng_state().clone()
                other=self.model();opt2=torch.optim.AdamW(other.coarse_flow.parameters(),lr=1e-3)
                restore_checkpoint(path,other,opt2,np.random.default_rng(9),torch.device('cpu'));actual=step(other,opt2)
                self.assertEqual(expected,actual);self.assertTrue(torch.equal(trng,torch.get_rng_state()))
                for name,v in weights.items():self.assertTrue(torch.equal(v,other.state_dict()[name]))
                for key,vs in adam['state'].items():
                    for name,v in vs.items():self.assertTrue(torch.equal(v,opt2.state_dict()['state'][key][name]))

if __name__=='__main__':unittest.main()
