import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from report_bounded_fine_search import interval

class PairedReportTests(unittest.TestCase):
    def test_constant_effect_has_degenerate_interval(self):
        result=interval([.5]*24)
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result['bootstrap95'],[.5,.5])
        self.assertEqual(result['proteins'],24)
    def test_nonfinite_not_silently_excluded(self):
        with self.assertRaises(ValueError):interval([1,np.nan])
    def test_reproducible(self):
        self.assertEqual(interval([-2,-1,0,1]),interval([-2,-1,0,1]))

if __name__=='__main__':unittest.main()
