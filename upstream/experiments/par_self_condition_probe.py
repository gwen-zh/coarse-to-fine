"""Fixed-weight, equal-NFE self-conditioning interventions and trajectory capture.

No target coordinates, learned critic, rejection or reflection correction enter
the sampler. Intermediate references are used only by a separate report stage.
"""
import numpy as np
import torch

from par_coarse_backbone import ar_condition, flow_batch


POLICIES = ['always', 'never', 'delayed_half']


def use_self_condition(policy, step, steps, has_previous):
    if policy not in POLICIES or steps < 1 or not 0 <= step < steps:
        raise ValueError('Fixed policy and valid integration step required')
    return bool(has_previous and (policy == 'always' or
                                 (policy == 'delayed_half' and 2*step >= steps)))


@torch.no_grad()
def trace_coarse(model, features, length, keys, policy, steps=16):
    if policy not in POLICIES or steps < 1 or not keys or length < 4:
        raise ValueError('Valid fixed-policy coarse sampling request required')
    batch, condition = ar_condition(model, features, length, len(keys))
    noise = torch.stack([torch.randn(((length+3)//4,3),
                          generator=torch.Generator().manual_seed(int(key))) for key in keys])
    state = noise.to(condition.device)
    state = state-state.mean(1,keepdim=True)
    states=[state.cpu().numpy().copy()];clean=[];used=[];previous=None
    for step in range(steps):
        t=step/steps
        enabled=use_self_condition(policy,step,steps,previous is not None)
        prediction=model.atom_diffusion.nn(flow_batch(batch,condition,state,t,
                                previous if enabled else None))['coors_pred']
        previous=(state+(1-t)*prediction).detach()
        clean.append(previous.cpu().numpy().copy());used.append(enabled)
        state=state+prediction/steps
        state=state-state.mean(1,keepdim=True)
        states.append(state.cpu().numpy().copy())
    result=state*10.
    counts=result.new_tensor([min(4,length-4*j) for j in range(result.shape[1])])
    result=result-(counts[None,:,None]*result).sum(1,keepdim=True)/length
    return dict(coarse=result.cpu().numpy(),states_nm=np.stack(states),
                clean_predictions_nm=np.stack(clean),self_condition_used=np.asarray(used,dtype=bool),
                seeds=np.asarray(keys,dtype=np.uint32),flow_calls=steps,AR_calls=1)


def concatenate_batches(batches):
    if not batches:raise ValueError('At least one fixed batch required')
    for value in batches[1:]:
        if not np.array_equal(value['self_condition_used'],batches[0]['self_condition_used']):
            raise RuntimeError('Self-conditioning schedule differs between batches')
    return dict(coarse=np.concatenate([b['coarse']for b in batches],0),
                states_nm=np.concatenate([b['states_nm']for b in batches],1),
                clean_predictions_nm=np.concatenate([b['clean_predictions_nm']for b in batches],1),
                self_condition_used=batches[0]['self_condition_used'],
                seeds=np.concatenate([b['seeds']for b in batches]),
                flow_calls=sum(b['flow_calls']for b in batches),AR_calls=len(batches))
