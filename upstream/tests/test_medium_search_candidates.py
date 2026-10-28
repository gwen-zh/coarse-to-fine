import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from medium_search_candidates import candidate_tables

class MediumCandidatesTests(unittest.TestCase):
    def test_same_group_candidates_and_origin(self):
        fine=np.array([[0.],[.2],[.4],[10.],[10.2],[10.4]])
        d=candidate_tables(fine,np.array([[0.],[10.]]),2)
        for token,row in enumerate(d['cluster']):
            self.assertEqual(row[0],token);self.assertEqual(len(set(row)),3)
            self.assertTrue(np.all(d['group'][row]==d['group'][token]))
        self.assertFalse(d['fallback'].any())
    def test_singleton_fallback_explicit(self):
        d=candidate_tables(np.array([[0.],[10.],[10.1],[10.2]]),np.array([[0.],[10.]]),2)
        np.testing.assert_array_equal(d['cluster'][0],[0,1,2]);np.testing.assert_array_equal(d['fallback'][0],[False,True,True])
    def test_global_nearest_can_cross_medium_boundary(self):
        d=candidate_tables(np.array([[.49],[.51],[.1],[.9]]),np.array([[0.],[1.]]),1)
        self.assertEqual(d['nearest'][0,1],1);self.assertEqual(d['cluster'][0,1],2)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):candidate_tables(np.array([[0.],[1.],[np.nan]]),np.array([[0.]]),1)

if __name__=="__main__":unittest.main()
