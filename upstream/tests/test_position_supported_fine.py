import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from position_supported_fine import sample_supported, sample_position_prior


class Dummy(torch.nn.Module):
    fine_size=8
    def __init__(self):
        super().__init__();self.calls=0
    def forward(self,h,c,f,rate):
        self.calls+=1
        logits=torch.zeros(len(h),8)
        logits[:,0]=50.
        return logits


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.counts=torch.zeros(8,4,dtype=torch.long)
        for i in range(4):self.counts[i,i]=1;self.counts[i+4,i]=2
        self.h=torch.randn(13,12);self.c=torch.randn(4,16)
    def test_excludes_wrong_position_even_if_raw_logit_dominates(self):
        m=Dummy();f=sample_supported(m,self.h,self.c,self.counts,17,steps=4)
        self.assertTrue(torch.equal(f%4,torch.arange(13)%4));self.assertEqual(m.calls,4)
    def test_seed_replay_and_global_rng_unchanged(self):
        before=torch.get_rng_state().clone()
        a=sample_supported(Dummy(),self.h,self.c,self.counts,18,4)
        b=sample_supported(Dummy(),self.h,self.c,self.counts,18,4)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(before,torch.get_rng_state()))
    def test_position_prior_needs_no_generator_or_structure(self):
        f=sample_position_prior(self.counts,13,17)
        self.assertTrue(torch.equal(f%4,torch.arange(13)%4))
    def test_short_ragged_chains_fill_and_match_budget(self):
        for n in [1,4,5,13]:
            m=Dummy();f=sample_supported(m,self.h[:n],self.c[:(n+3)//4],self.counts,19,16)
            self.assertEqual(f.shape,(n,));self.assertEqual(m.calls,min(n,16))
            self.assertTrue((f<8).all())
    def test_empty_position_support_is_an_error(self):
        counts=self.counts.clone();counts[:,2]=0
        with self.assertRaises(ValueError):sample_supported(Dummy(),self.h,self.c,counts,19)


if __name__=='__main__':unittest.main()
