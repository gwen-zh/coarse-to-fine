import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from codebook_capacity import nearest,refit_codes,counts_by_slot,paired_model,training_loss,decode_codes,state_hash
from hierarchical_rigid_vq import HierarchicalRigidVQ
from hierarchical_pipeline_io import save_checkpoint,restore_checkpoint

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.h=torch.randn(13,12);self.c=torch.tanh(torch.randn(4,16));self.y=torch.arange(13)%8
    def model(self,k):
        return paired_model(torch.zeros(16),torch.ones(16),k,31,maximum_k=32,condition_dim=12,width=32,layers=1,dropout=.1)
    def step(self,m,opt):
        m.train();opt.zero_grad(set_to_none=True);loss,terms=training_loss(m,self.h,self.c,self.y);loss.backward();opt.step();return float(loss),terms
    def test_nearest_matches_bruteforce_across_chunks(self):
        z=torch.tensor([[0.,0.],[1.,2.],[4.,3.],[-2.,2.]])
        codes=torch.tensor([[0.,0.],[4.,4.],[-3.,1.]])
        expected=((z[:,None]-codes).square().sum(-1)).argmin(-1)
        self.assertTrue(torch.equal(nearest(z,codes,1),expected))
        self.assertTrue(torch.equal(nearest(z,codes,3),expected))
        with self.assertRaises(ValueError):nearest(z,codes.new_empty(0,2))
    def test_fit_is_deterministic_and_does_not_modify_pool(self):
        pool=torch.randn(150,8);original=pool.clone()
        a,trace=refit_codes(pool,16,20,iterations=4,chunk=23)
        b,other=refit_codes(pool,16,20,iterations=4,chunk=43)
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
        self.assertEqual(trace,other);self.assertTrue(torch.equal(pool,original))
        self.assertTrue(torch.isfinite(a).all())
    def test_empty_cluster_restart_remains_finite(self):
        pool=torch.zeros(50,8);codes,trace=refit_codes(pool,16,21,3)
        self.assertTrue(torch.isfinite(codes).all());self.assertGreater(trace[-1]['restarted'],0)
        with self.assertRaises(ValueError):refit_codes(pool,51,21,3)
    def test_shared_initialization_and_mask_embedding_match(self):
        small=self.model(8);large=self.model(16)
        for name,v in small.state_dict().items():
            other=large.state_dict()[name]
            if name=='fine.fine_embedding.weight':
                self.assertTrue(torch.equal(v[:8],other[:8]));self.assertTrue(torch.equal(v[8],other[16]))
            elif name in ['fine.head.1.weight','fine.head.1.bias']:self.assertTrue(torch.equal(v,other[:8]))
            else:self.assertTrue(torch.equal(v,other),name)
    def test_paired_masks_jitter_dropout_rng_across_vocabulary(self):
        records=[]
        for k in [8,16,32]:
            m=self.model(k).train();captured=[]
            def hook(module,args):captured.append((args[1].clone(),args[2]!=k,float(args[3])))
            handle=m.fine.register_forward_pre_hook(hook)
            torch.manual_seed(42);loss,terms=training_loss(m,self.h,self.c,self.y);loss.backward()
            records.append((captured[0],torch.get_rng_state().clone(),terms['mask_sha256']));handle.remove()
        for other in records[1:]:
            self.assertTrue(torch.equal(other[0][0],records[0][0][0]))
            self.assertTrue(torch.equal(other[0][1],records[0][0][1]))
            self.assertEqual(other[0][2],records[0][0][2]);self.assertTrue(torch.equal(other[1],records[0][1]))
            self.assertEqual(other[2],records[0][2])
    def test_sampling_needs_only_features_coarse_seed_and_is_repeatable(self):
        m=self.model(16).eval()
        a=m.sample(self.h,self.c,30,4);b=m.sample(self.h,self.c,30,4)
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(13,));self.assertTrue(((a>=0)&(a<16)).all())
    def test_decode_4096_ids_uses_supplied_vectors_without_mutating_native_book(self):
        decoder=SimpleNamespace(model=HierarchicalRigidVQ(width=16,layers=1,fine_size=8).eval().requires_grad_(False))
        original=state_hash(decoder.model);codes=torch.randn(4096,8);ids=torch.tensor([4000,4095,2,9,1024])
        c=torch.tanh(torch.randn(2,16))
        with torch.no_grad():
            actual=decode_codes(decoder,c,ids,codes);expected=decoder.model.decode(c,codes[ids],5)['backbone']
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(state_hash(decoder.model),original)
        with self.assertRaises(ValueError):decode_codes(decoder,c,torch.tensor([4096]),codes)
    def test_slot_counts_and_illegal_ids(self):
        counts=counts_by_slot([torch.tensor([0,1,2,3,4]),torch.tensor([0,3])],8)
        self.assertEqual(int(counts.sum()),7);self.assertEqual(int(counts[0,0]),2);self.assertEqual(int(counts[4,0]),1)
        with self.assertRaises(ValueError):counts_by_slot([torch.tensor([8])],8)
    def test_checkpoint_restores_exact_next_update_adam_and_rng(self):
        m=self.model(16);opt=torch.optim.AdamW(m.parameters(),lr=1e-3);rng=np.random.default_rng(30)
        torch.manual_seed(32);self.step(m,opt)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'last.pt';save_checkpoint(path,m,opt,rng)
            expected=self.step(m,opt);weights=copy.deepcopy(m.state_dict());state=copy.deepcopy(opt.state_dict());torch_rng=torch.get_rng_state().clone()
            other=self.model(16);opt2=torch.optim.AdamW(other.parameters(),lr=1e-3)
            restore_checkpoint(path,other,opt2,np.random.default_rng(99),torch.device('cpu'));actual=self.step(other,opt2)
            self.assertEqual(actual,expected);self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
            for name,v in weights.items():self.assertTrue(torch.equal(v,other.state_dict()[name]))
            for key,vs in state['state'].items():
                for name,v in vs.items():self.assertTrue(torch.equal(v,opt2.state_dict()['state'][key][name]))

if __name__=='__main__':unittest.main()
