from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from generator_curriculum import training_schedule, event_pools, resolve_event, event_seed, learning_rate


def case(name, split="train"):
    return dict(id=name, sequence="A"*12, target=torch.zeros(12,3,3),
                coarse=torch.zeros(3,3), fine_continuous=torch.zeros(12,8), split=split)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.groups = dict(PDB_train=[case("p1"),case("p2")],
                           train={"a":[case("a1"),case("a2")],"b":[case("b1")]})

    def test_exact_same_unique_events_and_domain_counts(self):
        a,b=[training_schedule(arm,8,4) for arm in ["interleaved","PDB_then_MD"]]
        fa,fb=[sum(rows,[]) for rows in [a,b]]
        self.assertEqual(Counter(fa),Counter(fb))
        self.assertEqual(len(set(fa)),48)
        self.assertEqual(Counter(domain for domain,_ in fa),Counter(PDB=36,MD=12))
        self.assertNotEqual(a,b)
        self.assertTrue(all(domain=="PDB" for row in b[:8] for domain,_ in row))

    def test_case_and_noise_identity_follow_event_not_optimizer_position(self):
        pools=event_pools(self.groups,2026,8,4)
        maps=[]
        for arm in ["interleaved","PDB_then_MD"]:
            result={}
            for batch in training_schedule(arm,8,4):
                for event in batch:
                    result[event]=(resolve_event(self.groups,pools,event)["id"],event_seed(2026,*event))
            maps.append(result)
        self.assertEqual(maps[0],maps[1])

    def test_schedule_and_event_pools_replay_after_resume(self):
        a=training_schedule("PDB_then_MD",8,4)
        self.assertEqual(a[7:],training_schedule("PDB_then_MD",8,4)[7:])
        self.assertEqual(event_pools(self.groups,2027,8,4),event_pools(self.groups,2027,8,4))
        self.assertNotEqual(event_pools(self.groups,2026,8,4),event_pools(self.groups,2027,8,4))

    def test_validation_cases_are_rejected(self):
        bad=dict(PDB_train=[case("val",split="validation")],train=self.groups["train"])
        pools=event_pools(bad,2026,8,4)
        with self.assertRaises(ValueError):resolve_event(bad,pools,("PDB",0))

    def test_lr_and_invalid_schedules(self):
        p=dict(pretrain_updates=8,adapt_updates=4,pretrain_lr=3e-4,
               pretrain_min_lr=3e-5,adapt_lr=1e-4,adapt_min_lr=1e-5)
        self.assertAlmostEqual(learning_rate(p,0),3e-4)
        self.assertAlmostEqual(learning_rate(p,7),3e-5)
        self.assertAlmostEqual(learning_rate(p,8),1e-4)
        self.assertAlmostEqual(learning_rate(p,11),1e-5)
        with self.assertRaises(ValueError):training_schedule("PDB_then_MD",7,4)
        with self.assertRaises(ValueError):training_schedule("other",8,4)


if __name__ == "__main__":
    unittest.main()
