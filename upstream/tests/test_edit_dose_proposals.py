import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from edit_dose_proposals import coupled_fixed_edits,apply_multiedit

class EditDoseTests(unittest.TestCase):
    def setUp(self):
        self.original=np.arange(102);self.pool=np.stack([self.original+102*i for i in range(5)],axis=1)
    def test_exact_edits_nested_and_independent(self):
        streams=coupled_fixed_edits(self.pool,self.original,50,19)
        for j in range(50):
            for k,stream in streams.items():
                sites,tokens=stream[j];candidate=apply_multiedit(self.original,stream[j])
                self.assertEqual(np.count_nonzero(candidate!=self.original),k)
                self.assertEqual(len(set(sites)),k)
                np.testing.assert_array_equal(sites,streams[8][j][0][:k]);np.testing.assert_array_equal(tokens,streams[8][j][1][:k])
                self.assertTrue(all(t in self.pool[i] for i,t in zip(sites,tokens)))
        np.testing.assert_array_equal(self.original,np.arange(102))
    def test_repeatable_and_budget_prefix(self):
        a=coupled_fixed_edits(self.pool,self.original,2,19);b=coupled_fixed_edits(self.pool,self.original,5,19)
        for k in a:
            for x,y in zip(a[k],b[k]):np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
    def test_zero_budget_retained(self):
        self.assertEqual(coupled_fixed_edits(self.pool,self.original,0,19),{1:[],2:[],4:[],8:[]})
    def test_no_clipping_to_cap(self):
        with self.assertRaises(ValueError):coupled_fixed_edits(self.pool[:70],self.original[:70],1,19)
    def test_invalid_dose_or_pool(self):
        for doses in [(0,),(1,1),(1.5,)]:
            with self.assertRaises(ValueError):coupled_fixed_edits(self.pool,self.original,1,19,doses=doses)
        with self.assertRaises(ValueError):coupled_fixed_edits(self.pool[:,1:],self.original,1,19)

if __name__=='__main__':unittest.main()
