import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from gpu_latency_io import pack,unpack,paired_summary

class GPULatencyTests(unittest.TestCase):
    def test_transport_preserves_fine_logit_argmax(self):
        rng=np.random.default_rng(7);a=rng.normal(size=(23,4096)).astype(np.float32)
        b=unpack(pack(a));np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a.argmax(-1),b.argmax(-1))
    def test_transport_noncontiguous_coordinates(self):
        a=np.arange(180,dtype=np.float32).reshape(10,6,3)[:,::2]
        np.testing.assert_array_equal(unpack(pack(a)),a)
    def test_ratio_of_means_not_mean_of_ratios(self):
        x=paired_summary([{'original_wall_seconds':1.,'refined_wall_seconds':3.},{'original_wall_seconds':3.,'refined_wall_seconds':3.}])
        self.assertEqual(x['ratio_of_means'],1.5);self.assertEqual(x['added_percent'],50.);self.assertEqual(x['mean_added_seconds'],1.)
    def test_invalid_measurements_fail(self):
        for rows in [[],[{'original_wall_seconds':0.,'refined_wall_seconds':1.}],[{'original_wall_seconds':1.,'refined_wall_seconds':float('nan')}]]:
            with self.assertRaises(ValueError):paired_summary(rows)

if __name__=='__main__':unittest.main()
