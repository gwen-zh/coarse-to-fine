import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from bridge_search_candidates import ranked_candidates

class BridgeCandidatesTests(unittest.TestCase):
    def test_ranked_actual_distances(self):
        d=np.array([[4.,1.,3.,2.],[0.,3.,2.,1.]])
        np.testing.assert_array_equal(ranked_candidates(d,np.array([1,0]),3),[[1,3,2],[0,3,2]])
    def test_tied_original_preserved_without_duplicate(self):
        np.testing.assert_array_equal(ranked_candidates(np.zeros((1,4)),np.array([2]),3),[[2,0,1]])
    def test_wrong_original_rejected(self):
        with self.assertRaises(ValueError):ranked_candidates(np.array([[0.,2.]]),np.array([1]),2)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):ranked_candidates(np.array([[0.,np.nan]]),np.array([0]),2)

if __name__=="__main__":unittest.main()
