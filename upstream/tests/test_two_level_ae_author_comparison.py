import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from compare_two_level_ae_author import metrics,paired_rows,validate_pair


class Tests(unittest.TestCase):
    def setUp(self): self.x=np.random.default_rng(19).normal(size=(9,3,3))
    def test_rigid_transform_same_metrics_and_proper_alignment(self):
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        moved=self.x@rotation+np.array([4.,7.,-2.]);a=metrics(self.x,self.x);b=metrics(moved,self.x)
        for key in a:self.assertAlmostEqual(a[key],b[key],places=10)
        self.assertLess(b['ca_rmsd'],1e-10);self.assertLess(b['backbone_rmsd'],1e-10)
    def test_no_silent_residue_mask_or_nonfinite_exclusion(self):
        with self.assertRaises(ValueError):validate_pair(self.x[:-1],self.x,9)
        with self.assertRaises(ValueError):validate_pair(self.x,self.x,11)
        bad=self.x.copy();bad[3,1,0]=np.nan
        with self.assertRaises(ValueError):validate_pair(bad,self.x,9)
    def test_incomplete_or_duplicated_protein_pairing_rejected(self):
        targets=[dict(id='a',subset='train'),dict(id='b',subset='validation')]
        rows=[dict(target='a',subset='train'),dict(target='b',subset='validation')]
        self.assertEqual(paired_rows(rows,targets),rows)
        with self.assertRaises(ValueError):paired_rows(rows[:1],targets)
        with self.assertRaises(ValueError):paired_rows(rows+[rows[0]],targets)
        with self.assertRaises(ValueError):paired_rows([rows[0],rows[0]],targets)


if __name__=='__main__': unittest.main()
