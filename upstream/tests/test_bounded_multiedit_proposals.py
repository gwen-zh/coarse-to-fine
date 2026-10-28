import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from bounded_multiedit_proposals import maximum_edits,bounded_proposals,apply_multiedit


class BoundedProposalTests(unittest.TestCase):
    def setUp(self):
        self.original=np.arange(100,dtype=np.int64)
        self.pool=np.stack([self.original+i*100 for i in range(5)],axis=1)

    def test_cap_boundary(self):
        self.assertEqual(maximum_edits(19,.1),1)
        self.assertEqual(maximum_edits(10,.3),3)
        self.assertEqual(maximum_edits(9,.1),0)
        for n in range(1,201):
            m=maximum_edits(n,.1)
            self.assertLessEqual(m/n,.1)
            self.assertGreater((m+1)/n,.1)

    def test_actual_unique_edits_and_pool_membership(self):
        for distribution in ['uniform','geometric']:
            stream=bounded_proposals(self.pool,self.original,500,7,.1,distribution)
            for proposal in stream:
                sites,tokens=proposal;candidate=apply_multiedit(self.original,proposal)
                self.assertEqual(len(set(sites)),len(sites))
                self.assertEqual(np.count_nonzero(candidate!=self.original),len(sites))
                self.assertLessEqual(len(sites)/len(candidate),.1)
                self.assertGreater(len(sites),0)
                self.assertTrue(all(t in self.pool[i] and t!=self.original[i] for i,t in zip(sites,tokens)))
            np.testing.assert_array_equal(self.original,np.arange(100))

    def test_deterministic_prefix(self):
        first=bounded_proposals(self.pool,self.original,10,71,.1,'geometric')
        second=bounded_proposals(self.pool,self.original,30,71,.1,'geometric')
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_independence_not_accumulation(self):
        first=apply_multiedit(self.original,(np.array([0,1]),np.array([100,101])))
        second=apply_multiedit(self.original,(np.array([2,3]),np.array([102,103])))
        self.assertEqual(first[0],100);self.assertEqual(second[0],0)
        self.assertEqual(np.count_nonzero(second!=self.original),2)

    def test_distributions_and_empty_budget(self):
        uniform=bounded_proposals(self.pool,self.original,2000,4,.1,'uniform')
        geometric=bounded_proposals(self.pool,self.original,2000,4,.1,'geometric')
        self.assertEqual({len(x[0]) for x in uniform},set(range(1,11)))
        self.assertLess(np.mean([len(x[0]) for x in geometric]),3)
        self.assertEqual(bounded_proposals(self.pool,self.original,0,4,.1,'uniform'),[])
        padded=bounded_proposals(self.pool[:9],self.original[:9],3,4,.1,'uniform')
        self.assertTrue(all(len(x[0])==0 for x in padded))

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError):maximum_edits(0,.1)
        with self.assertRaises(ValueError):bounded_proposals(self.pool,self.original,1,1,.1,'tuned')
        with self.assertRaises(ValueError):bounded_proposals(self.pool[:,1:],self.original,1,1,.1,'uniform')

if __name__=='__main__':unittest.main()
