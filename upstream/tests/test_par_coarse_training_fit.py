from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"experiments"))
from par_coarse_training_fit import presented, reference_indices, fixed_probe, select_targets


class TrainingFitTests(unittest.TestCase):
    def test_reference_budget_keeps_unique_endpoints(self):
        indices = reference_indices(502)
        self.assertEqual(len(indices), 128)
        self.assertEqual(len(set(indices)), 128)
        self.assertEqual((indices[0], indices[-1]), (0, 501))
        self.assertEqual(reference_indices(1), [0])
        with self.assertRaises(ValueError): reference_indices(0)

    def test_fixed_time_probe_retains_same_ot_pair_and_does_not_modify_truth(self):
        case = {"coarse": torch.randn(26, 3)}
        saved = case["coarse"].clone()
        a = fixed_probe(case, 2026, 0, 0, .05)
        b = fixed_probe(case, 2026, 0, 0, .9)
        torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
        torch.testing.assert_close(b[0]-a[0], .85*a[1], atol=2e-7, rtol=2e-6)
        self.assertEqual(a[3:], b[3:])
        torch.testing.assert_close(case["coarse"], saved, atol=0, rtol=0)
        torch.testing.assert_close(a[0].mean(0), torch.zeros(3), atol=1e-7, rtol=0)

    def test_cohort_uses_cases_seen_in_both_seeds_and_keeps_domains_separate(self):
        def case(name, protein, length):
            return dict(id=name, protein=protein, sequence='A'*length, split='train')
        md = [case('m'+str(i), 'md', 90) for i in range(5)]
        pdb = [case('p0', 'p0', 103), case('p1', 'p1', 109), case('unseen', 'unseen', 103)]
        dev = [dict(id='d'+str(i)) for i in range(128)]
        ids = [c['id'] for c in md[:4]+pdb[:2]]
        t0 = [dict(batch=[dict(case_id=i) for i in ids+['unseen']])]
        t1 = [dict(batch=[dict(case_id=i) for i in ids])]
        out = select_targets(dict(train={'md':md}, PDB_train=pdb), dict(validation={'dev':dev}),
                             {2026:t0,2027:t1}, [dict(protein='dev',sequence='C'*107,length=107)])
        self.assertEqual([x['domain'] for x in out], ['MD_train','PDB_train','PDB_train','development'])
        self.assertNotIn('unseen', [i for r in out for i in r['probe_case_ids']])
        self.assertEqual(out[0]['probe_case_ids'], ['m0','m1','m2','m3'])
        self.assertEqual([x['protein'] for x in out[1:3]], ['p0','p1'])
        self.assertEqual(out[-1]['reference_count'],128)


if __name__ == '__main__': unittest.main()
