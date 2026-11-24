import unittest,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from target_scaling_flow import updates,event,learning_rate
from few_target_flow import learning_rate as eight_rate


class ScalingTests(unittest.TestCase):
    def test_exact_per_case_noise_counters(self):
        for arm,n in [("fixed_total",256),("matched_presentations",1024)]:
            rows=defaultdict(list)
            for step in range(updates(arm)):
                case,counters=event(step);rows[case]+=counters
            self.assertEqual(set(rows),set(range(32)))
            for values in rows.values():self.assertEqual(values,list(range(n)))

    def test_first_eight_cases_match_old_per_case_lr(self):
        for local in range(256):
            for case in range(8):self.assertEqual(learning_rate("matched_presentations",local*32+case),eight_rate("fixed_total",local*8+case))

    def test_fixed_global_lr_and_invalid_steps(self):
        for step in [0,512,2047]:self.assertEqual(learning_rate("fixed_total",step),eight_rate("fixed_total",step))
        with self.assertRaises(ValueError):learning_rate("fixed_total",2048)
        with self.assertRaises(ValueError):updates("bad")


if __name__=="__main__":unittest.main()
