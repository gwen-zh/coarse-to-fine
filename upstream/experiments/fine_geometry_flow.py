"""Fine-only flow adaptation with an immutable coarse sampler.

Both control arms execute the same geometry branch; its loss weight is zero in
the flow-only control. Continuous decoder gradients are exact, without STE.
"""
import hashlib

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from audit_continuous_fine_geometry import geometry_loss
from hierarchical_ensemble_flow import component_seed
from multisequence_flow_data import check_case


class FixedCoarseFineFlow(nn.Module):
    def __init__(self, coarse_model, fine_model):
        super().__init__()
        self.coarse_model=coarse_model.eval().requires_grad_(False)
        self.fine_model=fine_model
        self.fine_model.coarse_head.requires_grad_(False)
        self.coarse_dim=coarse_model.coarse_dim
        self.fine_dim=fine_model.fine_dim
        self.config=dict(coarse_sampler="frozen_parent",fine_model=fine_model.config)

    def train(self,mode=True):
        super().train(mode)
        self.coarse_model.eval()
        return self

    def sample_coarse(self,*args,**kwargs):
        return self.coarse_model.sample_coarse(*args,**kwargs)

    def sample_fine(self,*args,**kwargs):
        return self.fine_model.sample_fine(*args,**kwargs)

    def trainable_parameters(self):
        return [v for v in self.fine_model.parameters() if v.requires_grad]


def sample_training_batch(groups,rng):
    cases=[groups["PDB_train"][int(rng.integers(len(groups["PDB_train"])))] for _ in range(3)]
    names=sorted(groups["train"])
    name=names[int(rng.integers(len(names)))]
    cases.append(groups["train"][name][int(rng.integers(len(groups["train"][name])))])
    for case in cases:check_case(case,"train")
    return cases


def draw_seed(seed,step,slot):
    return int(hashlib.sha256(f"fine-geometry-flow-v1:{seed}:{step}:{slot}".encode()).hexdigest()[:8],16)


def fine_flow_loss(model,aa,coarse,fine,generator):
    fine_model=model.fine_model
    normalized=fine_model.normalize_fine(fine.detach())
    noise=torch.randn(normalized.shape,generator=generator,device=normalized.device,dtype=normalized.dtype)
    t=torch.rand((),generator=generator,device=normalized.device,dtype=normalized.dtype)
    state=(1-t)*noise+t*normalized
    velocity=fine_model.fine_velocity(aa,state,t,fine_model.normalize_coarse(coarse.detach()))
    flow=(velocity-(normalized-noise)).square().mean()
    digest=hashlib.sha256(noise.detach().cpu().numpy().tobytes()+t.detach().cpu().numpy().tobytes()).hexdigest()
    return flow,dict(flow=float(flow.detach()),time=float(t),draw_sha256=digest)


def fine_rollout_geometry_loss(model,codec,aa,coarse,seed,nfe=16,checkpoint_velocity=True):
    """Differentiate the FULL inference-matched Euler trajectory and decoder.

    Activation checkpointing recomputes velocity forwards during backward; no
    trajectory prefix or fine state is detached. Native coarse is the fixed
    training condition. This remains an oracle-coarse training objective.
    """
    if nfe<1:raise ValueError("Positive rollout NFE required")
    fine_model=model.fine_model
    condition=fine_model.normalize_coarse(coarse.detach())
    rng=torch.Generator(device=aa.device).manual_seed(component_seed(seed,"fine"))
    state=torch.randn((len(aa),model.fine_dim),generator=rng,device=aa.device)
    digest=hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
    for step in range(nfe):
        def velocity(value,time=step/nfe):
            return fine_model.fine_velocity(aa,value,time,condition)
        if checkpoint_velocity:
            v=checkpoint(velocity,state,use_reentrant=False,preserve_rng_state=False)
        else:v=velocity(state)
        state=state+v/nfe
    fine=fine_model.denormalize_fine(state)
    geometry,bb=geometry_loss(codec,coarse.detach(),fine)
    return geometry,dict(draw_sha256=digest),fine,bb
