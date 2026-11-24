from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from native_ensemble_metrics import train_histogram_edges,histogram_probabilities,js_divergence,compare_ensembles,distance_features,rmsf


class NativeEnsembleMetricTests(unittest.TestCase):
    def test_overflow_bins_retain_samples_outside_train_range(self):
        edges=train_histogram_edges(np.array([[0.],[1.]]))
        p=histogram_probabilities(np.array([[-100.],[100.]]),edges)
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertEqual((p[0,0],p[0,-1]),(.5,.5))
    def test_js_is_divergence_in_nats_and_symmetric(self):
        p=np.array([[1.,0.]]);q=np.array([[0.,1.]])
        self.assertAlmostEqual(float(js_divergence(p,q)[0]),np.log(2.))
        np.testing.assert_array_equal(js_divergence(p,q),js_divergence(q,p))
        np.testing.assert_array_equal(js_divergence(p,p),[0.])
    def test_identical_ensemble_has_zero_distribution_and_coverage_error(self):
        ca=np.random.default_rng(3).normal(size=(4,8,3));d,r=distance_features(ca)
        result=compare_ensembles(ca,ca,train_histogram_edges(d),train_histogram_edges(r),ca[0])
        self.assertAlmostEqual(result['js_pairwise_distance_divergence_nats'],0.)
        self.assertAlmostEqual(result['rmsf_mae_A'],0.)
        self.assertLess(result['mean_sample_nearest_reference_ca_rmsd_A'],1e-7)
        self.assertEqual(result['reference_coverage_at_A']['0.5'],1.)
    def test_rmsf_removes_rigid_translation_and_rotation(self):
        rng=np.random.default_rng(9);ca=rng.normal(size=(8,3));q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        values=np.stack([ca,ca@q+3.])
        np.testing.assert_allclose(rmsf(values,ca),0.,atol=1e-7)


if __name__=='__main__':unittest.main()
