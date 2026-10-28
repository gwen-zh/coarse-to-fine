"""Numerical integration helpers for a frozen coarse velocity predictor."""
import numpy as np
import torch
from latent_flow_head import euler_sample


@torch.no_grad()
def integrate(model, initial, condition, valid, steps, method='euler'):
    if steps<1 or method not in ['euler','midpoint']:
        raise ValueError('Positive steps and known solver required')
    if method=='euler':
        return euler_sample(model,initial,condition,valid,steps)
    state=initial.clone();dt=1./steps
    for i in range(steps):
        t=torch.full((len(state),),i*dt,device=state.device,dtype=state.dtype)
        v=model(state,t,condition,valid)
        midpoint=state+.5*dt*v
        state=state+dt*model(midpoint,t+.5*dt,condition,valid)
    return state


def weighted_rmsd(mobile,reference,weights):
    a,b,w=np.asarray(mobile,float),np.asarray(reference,float),np.asarray(weights,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or w.shape!=(len(a),) or (w<=0).any() or not all(np.isfinite(x).all() for x in [a,b,w]):
        raise ValueError('Finite corresponding points and positive weights required')
    w=w/w.sum();x=a-(a*w[:,None]).sum(0);y=b-(b*w[:,None]).sum(0)
    u,_,vt=np.linalg.svd((x*w[:,None]).T@y);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    return float(np.sqrt((w*np.sum((x@(u@correction@vt)-y)**2,-1)).sum()))
