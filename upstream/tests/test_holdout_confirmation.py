import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from holdout_confirmation import holdout_targets,paired_summary,fingerprint,load_locked


class HoldoutConfirmationTests(unittest.TestCase):
    def test_protein_unit_and_family_interval(self):
        result=paired_summary(np.full(58,-2.))
        self.assertEqual(result['bootstrap97_5'],[-2.,-2.])
        self.assertTrue(result['negative_family_adjusted_interval'])
        self.assertFalse(paired_summary(np.zeros(58))['negative_family_adjusted_interval'])
        with self.assertRaises(ValueError):paired_summary(np.ones(116))
        with self.assertRaises(ValueError):paired_summary(np.full(58,float('nan')))

    def test_fixed_unique_holdout_and_overlap_rejection(self):
        rows=[]
        for i in range(58):
            sequence='A'*(40+i);rows.append(dict(sequence=sequence,length=len(sequence),sequence_hash=hashlib.sha256(sequence.encode()).hexdigest(),paths=['fixed.pdb'],split='holdout'))
        self.assertEqual(len(holdout_targets(dict(all_groups=rows))),58)
        with self.assertRaises(ValueError):holdout_targets(dict(all_groups=rows+[dict(rows[0],split='development')]))

    def test_changed_locked_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'candidate';data.write_text('fixed')
            (root/'protocol.json').write_text(json.dumps(dict(inputs=dict(candidate=fingerprint(data)))))
            load_locked(root);data.write_text('changed checkpoint')
            with self.assertRaises(ValueError):load_locked(root)


if __name__=='__main__':unittest.main()
