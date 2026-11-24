from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from par_two_protein_fit import matched_draw, sample_keys, identity


class TwoProteinFitTests(unittest.TestCase):
    def test_time_factor_changes_only_time_not_noise_rotation_or_sc_coin(self):
        torch.manual_seed(4); target = torch.randn(23,3)*12
        beta = matched_draw(target,2026,7,1,'beta')
        uniform = matched_draw(target,2026,7,1,'uniform')
        torch.testing.assert_close(beta[1],uniform[1],atol=0,rtol=0)
        torch.testing.assert_close(beta[0]-beta[2]*beta[1],uniform[0]-uniform[2]*uniform[1],atol=2e-7,rtol=2e-6)
        self.assertEqual(beta[3:],uniform[3:])
        self.assertNotEqual(float(beta[2]),float(uniform[2]))
        self.assertGreaterEqual(float(beta[2]),float(uniform[2]))

    def test_same_noise_prefix_is_shared_by_both_proteins_and_budgets(self):
        self.assertEqual(sample_keys(2026,32)[:8],sample_keys(2026,8))
        self.assertEqual(len(set(sample_keys(2026,32))),32)
        self.assertNotEqual(sample_keys(2026,8),sample_keys(2027,8))

    def test_pairing_identity_excludes_only_the_controlled_time_value(self):
        event=dict(case_id='train',time=.2,self_condition=True,uniform_time_draw=.1,mixture_coin=.8,noise_rotation_sha256='a')
        row=dict(lr=.001,batch=[event])
        other=dict(lr=.001,batch=[dict(event,time=.1)])
        self.assertEqual(identity(row),identity(other))
        other['batch'][0]['uniform_time_draw']=.12
        self.assertNotEqual(identity(row),identity(other))


if __name__=='__main__':unittest.main()
