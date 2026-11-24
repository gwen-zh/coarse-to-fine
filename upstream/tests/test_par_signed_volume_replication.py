from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from par_coarse_capacity import evaluation_keys
from par_signed_volume_replication import evaluation_noise_keys


class ReplicationNoiseTests(unittest.TestCase):
    def test_new_cohort_is_disjoint_from_exploratory_noise(self):
        keys = evaluation_noise_keys()
        self.assertEqual(len(keys), 64)
        self.assertEqual(len(set(keys)), 64)
        self.assertFalse(set(keys) & set(evaluation_keys(2026)))

if __name__ == '__main__':
    unittest.main()
