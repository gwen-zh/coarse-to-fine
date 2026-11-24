from pathlib import Path
import sys,unittest,math
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from few_target_flow import updates,event,learning_rate


class FewTargetTests(unittest.TestCase):
    def test_exact_counts_and_no_duplicate_noise_events(self):
        for arm,count in [("fixed_total",1024),("matched_presentations",8192)]:
            samples=defaultdict(list)
            for step in range(updates(arm)):
                case,indices=event(step);samples[case]+=indices
            self.assertEqual(set(samples),set(range(8)))
            for values in samples.values():self.assertEqual(values,list(range(count)))

    def test_learning_rate_matches_single_target_by_exposure(self):
        for exposure in [0,1,256,1024,2047]:
            expected=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*exposure/2047))
            for case in range(8):self.assertEqual(learning_rate("matched_presentations",exposure*8+case),expected)
            self.assertEqual(learning_rate("fixed_total",exposure),expected)

    def test_reject_invalid_settings(self):
        for arm,step in [("bad",0),("fixed_total",2048),("matched_presentations",-1)]:
            with self.assertRaises(ValueError):learning_rate(arm,step)
        with self.assertRaises(ValueError):event(0,case_count=1)


if __name__=="__main__":unittest.main()
