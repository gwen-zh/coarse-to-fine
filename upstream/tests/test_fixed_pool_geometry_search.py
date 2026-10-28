import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from fixed_pool_geometry_search import search_fixed_pool

class FixedPoolTests(unittest.TestCase):
    def setUp(self):
        self.ids=np.zeros(4,dtype=np.int64);self.pool=np.array([[0,1]]*4);self.bb=np.array([[[3.8*i,0,0],[3.8*i+1.45,0,0],[3.8*i+2.45,0,0]] for i in range(4)])
    def test_zero_budget_no_hidden_decode(self):
        def fail(ids):raise AssertionError("Unexpected decode")
        ids,bb,report=search_fixed_pool(self.ids,self.pool,self.bb,0,fail)
        np.testing.assert_array_equal(ids,self.ids);self.assertEqual(report['decoder_calls'],0)
    def test_exact_budget_padding_when_no_violations(self):
        seen=[]
        def decode(ids):seen.append(ids.copy());return self.bb.copy()
        ids,bb,r=search_fixed_pool(self.ids,self.pool,self.bb,3,decode)
        self.assertEqual(len(seen),3);self.assertEqual(r['padding_calls'],3);self.assertTrue(r['returned_original'])
    def test_invalid_candidates_fall_back_to_initial(self):
        bb=self.bb.copy();bb[1:,0,0]+=1
        def decode(ids):return np.full_like(bb,np.nan)
        ids,_,r=search_fixed_pool(self.ids,self.pool,bb,2,decode)
        np.testing.assert_array_equal(ids,self.ids);self.assertEqual(r['decoder_calls'],2);self.assertTrue(r['returned_original'])

if __name__=='__main__':unittest.main()
