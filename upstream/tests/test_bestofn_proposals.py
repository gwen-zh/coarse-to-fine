import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from bestofn_proposals import local_proposals,apply_proposal

class ProposalTests(unittest.TestCase):
    def test_shared_stream_and_no_origin(self):
        initial=np.array([0,2,1]);table=np.array([[0,1,2],[2,0,1],[0,1,2]])
        a=local_proposals(table,initial,50,7);self.assertEqual(a,local_proposals(table,initial,50,7))
        for i,t in a:self.assertNotEqual(initial[i],t)
    def test_independent_does_not_accumulate(self):
        initial=np.array([0,0,0]);first=apply_proposal(initial,(0,1));second=apply_proposal(initial,(1,2));greedy=apply_proposal(first,(1,2))
        self.assertEqual(np.count_nonzero(second!=initial),1);self.assertEqual(np.count_nonzero(greedy!=initial),2)
        np.testing.assert_array_equal(initial,[0,0,0])
    def test_zero_budget(self):
        self.assertEqual(local_proposals(np.array([[0,1]]),np.array([0]),0,1),[])

if __name__=="__main__":unittest.main()
