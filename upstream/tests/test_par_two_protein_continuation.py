from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from par_two_protein_continuation import learning_rate,global_step,ADDITIONAL_UPDATES,common_identity


class ContinuationTests(unittest.TestCase):
    def test_floor_does_not_silently_restart_and_restart_has_declared_endpoints(self):
        self.assertEqual({learning_rate(i,'floor') for i in range(ADDITIONAL_UPDATES)},{1e-5})
        values=[learning_rate(i,'restart') for i in range(ADDITIONAL_UPDATES)]
        self.assertAlmostEqual(values[0],1e-4)
        self.assertAlmostEqual(values[-1],1e-5)
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))

    def test_noise_step_range_continues_without_replaying_the_parent(self):
        steps=[global_step(i) for i in range(ADDITIONAL_UPDATES)]
        self.assertEqual(steps,list(range(1024,4096)))
        for i in [-1,ADDITIONAL_UPDATES]:
            with self.assertRaises(ValueError):global_step(i)

    def test_common_pairing_excludes_only_controlled_lr_and_time(self):
        event=dict(case_id='train',time=.4,uniform_time_draw=.2,self_condition=True)
        a=dict(global_update=1025,batch=[event],lr=1e-5)
        b=dict(global_update=1025,batch=[dict(event,time=.2)],lr=1e-4)
        self.assertEqual(common_identity(a),common_identity(b))
        b['batch'][0]['case_id']='other'
        self.assertNotEqual(common_identity(a),common_identity(b))


if __name__=='__main__':unittest.main()
