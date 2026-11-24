"""Paired native coarse FM updates for a frozen PAR backbone and AA adapter."""
import hashlib
import json
import math

import torch
from torch import nn

from par_coarse_backbone import ar_condition,flow_batch


ARMS=["position_only","sequence"]


def conditioning_features(features,arm):
    if arm not in ARMS:raise ValueError("Declared sequence-information control required")
    return torch.zeros_like(features) if arm=="position_only" else features


def adapter_view(factory):
    view=nn.Module()
    view.add_module("residue_projection",factory.residue_projection)
    view.add_module("block_projection",factory.block_projection)
    return view


def draw_key(seed,step,slot):
    return int(hashlib.sha256(f"par-native-coarse-transfer-v1:{seed}:{step}:{slot}".encode()).hexdigest()[:8],16)


def draw(target,seed,step,slot):
    """The same CPU Gaussian, proper rotation, beta/uniform time and SC coin."""
    generator=torch.Generator().manual_seed(draw_key(seed,step,slot))
    target=target.detach().cpu()/10.
    target=target-target.mean(0)
    noise=torch.randn(target.shape,generator=generator);noise=noise-noise.mean(0)
    q,r=torch.linalg.qr(torch.randn(3,3,generator=generator))
    q=q@torch.diag(torch.where(r.diag()<0,-1.,1.))
    q[:,-1]*=torch.linalg.det(q)
    target=target@q
    # Proper Kabsch OT coupling, also used by the official coordinate FM.
    u,_,vh=torch.linalg.svd(noise.T@target)
    correction=torch.eye(3);correction[-1,-1]=torch.linalg.det(u@vh)
    noise=noise@(u@correction@vh)
    coins=torch.rand(3,generator=generator)
    t=coins[0] if coins[1]<.02 else coins[0].pow(1/1.9)
    self_condition=bool(coins[2]<.5)
    state=(1-t)*noise+t*target;truth=target-noise
    h=hashlib.sha256()
    for value in [target,noise,q,coins]:h.update(value.numpy().tobytes())
    return state,truth,t,self_condition,h.hexdigest()


def update(model,optimizer,batch,features,arm,seed,step,p):
    if len(batch)!=4:raise ValueError("Three PDB plus one native MD sample required")
    device=next(model.parameters()).device;model.eval();optimizer.zero_grad(set_to_none=True)
    lr=p["min_lr"]+.5*(p["lr"]-p["min_lr"])*(1+math.cos(math.pi*step/max(1,p["updates"]-1)))
    for group in optimizer.param_groups:group['lr']=lr
    total=0.;events=[];sc_count=0
    for slot,case in enumerate(batch):
        if case['split']!='train':raise ValueError("Development case entered coarse transfer")
        h=conditioning_features(features(case['sequence']).to(device),arm)
        state,truth,t,self_condition,digest=draw(case['coarse'],seed,step,slot)
        state=state[None].to(device);truth=truth[None].to(device);t=t.to(device)
        b,z=ar_condition(model,h,len(case['sequence']))
        x_sc=None
        if self_condition:
            with torch.no_grad():
                preliminary=model.atom_diffusion.nn(flow_batch(b,z.detach(),state,t))['coors_pred']
                x_sc=(state+(1-t)*preliminary).detach()
            sc_count+=1
        prediction=model.atom_diffusion.nn(flow_batch(b,z,state,t,x_sc))['coors_pred']
        loss=(prediction-truth).square().mean()
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite PAR coarse adaptation loss")
        (loss/4).backward();total+=float(loss.detach())/4
        events.append(dict(case_id=case['id'],protein=case['protein'],domain='PDB' if slot<3 else 'MD',
            draw_sha256=digest,self_conditioning=self_condition,time=float(t)))
    params=[v for v in model.parameters() if v.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,p['gradient_clip'])
    if not torch.isfinite(norm):raise RuntimeError("Nonfinite sequence-adapter gradient")
    optimizer.step()
    return dict(update=step+1,loss=total,lr=lr,gradient_norm=float(norm),batch=events,self_condition_count=sc_count)


def identity(row):return {key:row[key] for key in ['batch','lr']}


def frozen_versions(model):
    return dict(parameters={n:(id(v),v._version) for n,v in model.named_parameters() if not v.requires_grad},
        buffers={n:(id(v),v._version) for n,v in model.named_buffers()})


def assert_frozen(model,expected):
    for family,values in [('parameters',model.named_parameters()),('buffers',model.named_buffers())]:
        for name,value in values:
            if name in expected[family] and (id(value),value._version)!=expected[family][name]:
                raise RuntimeError('Frozen PAR state changed: '+name)
            if family=='parameters' and not value.requires_grad and value.grad is not None:
                raise RuntimeError('Frozen PAR parameter received a gradient: '+name)
