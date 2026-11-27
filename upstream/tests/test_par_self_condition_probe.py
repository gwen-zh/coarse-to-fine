from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from par_self_condition_probe import trace_coarse,use_self_condition,concatenate_batches
from par_coarse_backbone import sample_coarse_batch


class ToyField(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.tensor(.3));self.calls=[]
    def forward(self,batch):
        self.calls.append('x_sc' in batch)
        state=batch['x_t'];sc=batch.get('x_sc',torch.zeros_like(state))
        return dict(coors_pred=-self.scale*state+.15*sc+.04*batch['t'][:,None,None])


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.atom_diffusion=nn.Module();self.atom_diffusion.nn=ToyField()


def condition(model,features,length,batch_size):
    m=(length+3)//4
    return dict(mask=torch.ones(batch_size,m,dtype=torch.bool),
                x_input=torch.zeros(batch_size,m,3)),torch.zeros(batch_size,m,7)


class SelfConditionProbeTests(unittest.TestCase):
    def trace(self,model,keys,policy):
        with patch('par_self_condition_probe.ar_condition',side_effect=condition):
            return trace_coarse(model,None,10,keys,policy,16)

    def test_always_policy_replays_native_sampler_exactly(self):
        model=ToyModel();keys=[11,12,13]
        with patch('par_coarse_backbone.ar_condition',side_effect=condition):
            native=sample_coarse_batch(model,None,10,keys,16).numpy()
        model.atom_diffusion.nn.calls.clear()
        actual=self.trace(model,keys,'always')
        np.testing.assert_array_equal(actual['coarse'],native)
        self.assertEqual(model.atom_diffusion.nn.calls,[False]+[True]*15)
        self.assertEqual(actual['flow_calls'],16)

    def test_policies_change_only_the_self_conditioning_schedule(self):
        outputs={}
        for policy in ['always','never','delayed_half']:
            model=ToyModel();before=model.atom_diffusion.nn.scale.detach().clone()
            outputs[policy]=self.trace(model,[11,12],policy)
            self.assertEqual(len(model.atom_diffusion.nn.calls),16)
            self.assertIsNone(model.atom_diffusion.nn.scale.grad)
            torch.testing.assert_close(before,model.atom_diffusion.nn.scale,atol=0,rtol=0)
        np.testing.assert_array_equal(outputs['never']['self_condition_used'],[False]*16)
        np.testing.assert_array_equal(outputs['delayed_half']['self_condition_used'],[False]*8+[True]*8)
        np.testing.assert_array_equal(outputs['never']['states_nm'][:9],outputs['delayed_half']['states_nm'][:9])
        np.testing.assert_array_equal(outputs['never']['states_nm'][:2],outputs['always']['states_nm'][:2])
        self.assertGreater(float(np.abs(outputs['never']['coarse']-outputs['delayed_half']['coarse']).max()),1e-5)

    def test_native_units_ragged_centering_and_trace_batch_concatenation(self):
        model=ToyModel();a=self.trace(model,[11,12],'always');b=self.trace(model,[13],'always')
        combined=concatenate_batches([a,b])
        self.assertEqual(combined['states_nm'].shape,(17,3,3,3))
        self.assertEqual(combined['clean_predictions_nm'].shape,(16,3,3,3))
        self.assertEqual(combined['flow_calls'],32);self.assertEqual(combined['AR_calls'],2)
        weights=np.asarray([4,4,2])/10
        np.testing.assert_allclose((combined['coarse']*weights[None,:,None]).sum(1),0,atol=1e-6)
        final_nm=combined['states_nm'][-1]
        expected=10*(final_nm-(final_nm*weights[None,:,None]).sum(1,keepdims=True))
        np.testing.assert_allclose(combined['coarse'],expected,atol=2e-6)

    def test_invalid_policies_and_steps_fail_before_sampling(self):
        for args in [('unknown',0,16,True),('always',16,16,True),('always',0,0,True)]:
            with self.assertRaises(ValueError):use_self_condition(*args)
        self.assertFalse(use_self_condition('always',0,16,False))
        self.assertTrue(use_self_condition('delayed_half',8,16,True))


if __name__=='__main__':unittest.main()
