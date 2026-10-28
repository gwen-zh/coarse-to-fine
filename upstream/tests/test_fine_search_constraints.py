import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from fine_search_constraints import evaluate_candidate

class GateTests(unittest.TestCase):
    def setUp(self):
        self.bb=np.arange(36,dtype=float).reshape(4,3,3)
        self.ids=np.zeros(4,dtype=int)
        self.base=dict(all_ca_finite=True,backbone_cn_finite=True,backbone_cn_outliers=3,upstream_ca_clashes=2)
    def gate(self,ids=None,bb=None,metrics=None):
        return evaluate_candidate(self.ids if ids is None else ids,self.bb if bb is None else bb,self.ids,self.bb,self.base if metrics is None else metrics,self.base,1.,.25)["eligible"]
    def test_strict_improvement_required(self):
        self.assertFalse(self.gate())
        self.assertTrue(self.gate(metrics={**self.base,"backbone_cn_outliers":2}))
    def test_tradeoff_rejected(self):
        self.assertFalse(self.gate(metrics={**self.base,"backbone_cn_outliers":2,"upstream_ca_clashes":3}))
    def test_edit_cap(self):
        m={**self.base,"backbone_cn_outliers":2}
        self.assertTrue(self.gate(ids=np.array([1,0,0,0]),metrics=m))
        self.assertFalse(self.gate(ids=np.array([1,1,0,0]),metrics=m))
    def test_displacement_and_finite(self):
        m={**self.base,"backbone_cn_outliers":2};bb=self.bb.copy();bb[0,1]+=20
        self.assertFalse(self.gate(bb=bb,metrics=m));bb[0,1]=np.nan
        self.assertFalse(self.gate(bb=bb,metrics=m))
    def test_reference_never_part_of_gate(self):
        self.assertTrue(self.gate(metrics={**self.base,"backbone_cn_outliers":2,"closest_available_reference_ca_rmsd":9999}))

if __name__=="__main__":unittest.main()
