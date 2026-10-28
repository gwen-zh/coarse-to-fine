"""Sequence-conditioned continuous residue-frame flow; no token decoder."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
VENDOR=Path(__file__).resolve().parents[1]/'protein-frame-flow'
sys.path.insert(0,str(VENDOR))
from models.flow_model import FlowModel
from data import so3_utils
from openfold.utils import rigid_utils


def frames_from_backbone(backbone):
    ca=backbone[...,1,:];x=F.normalize(backbone[...,2,:]-ca,dim=-1);n=backbone[...,0,:]-ca;y=F.normalize(n-(n*x).sum(-1,keepdim=True)*x,dim=-1);z=torch.cross(x,y,dim=-1)
    return torch.stack([x,y,z],-1),ca


def backbone_from_frames(rotation,translation):
    local=torch.tensor([[-.525,1.363,0.],[0.,0.,0.],[1.526,0.,0.]],device=translation.device,dtype=translation.dtype)
    return torch.einsum('...ij,aj->...ai',rotation,local)+translation[...,None,:]


def centered(x):return x-x.mean(-2,keepdim=True)


def random_prior(batch,n,device,generator=None):
    trans=centered(torch.randn(batch,n,3,device=device,generator=generator))*10.
    q=F.normalize(torch.randn(batch,n,4,device=device,generator=generator),dim=-1)
    return rigid_utils.quat_to_rot(q),trans


def interpolate(rotation,translation,noise_rotation,noise_translation,t):
    rt=so3_utils.geodesic_t(t[:,None,None],rotation,noise_rotation)
    xt=(1-t[:,None,None])*noise_translation+t[:,None,None]*translation
    return rt,xt


class ConditionedNodes(nn.Module):
    def __init__(self,base,width):
        super().__init__();self.base=base;self.projection=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,width),nn.SiLU(),nn.Linear(width,width));self.condition=None
    def forward(self,*args):
        if self.condition is None:raise RuntimeError('Pure sequence features required')
        return self.base(*args)+self.projection(self.condition)


class CoordinateFlow(FlowModel):
    def __init__(self):
        config=OmegaConf.load(VENDOR/'configs/model.yaml').model
        super().__init__(config);self.node_feature_net=ConditionedNodes(self.node_feature_net,config.node_embed_size)
    def forward(self,rotation,translation,t,sequence_features):
        b,n=translation.shape[:2];mask=torch.ones(b,n,device=translation.device)
        self.node_feature_net.condition=sequence_features
        try:
            result=super().forward(dict(res_mask=mask,diffuse_mask=mask,res_idx=torch.arange(n,device=translation.device)[None].expand(b,-1),so3_t=t[:,None],r3_t=t[:,None],trans_t=translation,rotmats_t=rotation))
        finally:self.node_feature_net.condition=None
        return result['pred_rotmats'],centered(result['pred_trans'])


def training_loss(model,rotation,translation,features,t,noise_rotation,noise_translation):
    rt,xt=interpolate(rotation,translation,noise_rotation,noise_translation,t);pr,px=model(rt,xt,t,features);scale=(1-t).clamp_min(.1)[:,None,None]
    trans=((px-translation)/10./scale).square().mean();target_vf=so3_utils.calc_rot_vf(rt,rotation);pred_vf=so3_utils.calc_rot_vf(rt,pr);rot=((pred_vf-target_vf)/scale).square().mean()
    bb=backbone_from_frames(pr,px);truth=backbone_from_frames(rotation,translation)
    atom=((bb-truth)/10.).square().mean((1,2,3));pair=((torch.cdist(px,px)-torch.cdist(translation,translation))/10.).square().mean((1,2));cn=torch.linalg.vector_norm(bb[:,:-1,2]-bb[:,1:,0],dim=-1);geometry=(F.relu(cn-1.8).square()+F.relu(1.-cn).square()).mean(-1)/100.
    auxiliary=((atom+pair+geometry)*(t>.5)).mean();total=trans+rot+.1*auxiliary
    return total,dict(translation=trans.detach(),rotation=rot.detach(),auxiliary=auxiliary.detach())


@torch.no_grad()
def sample(model,features,seed,steps=64):
    if steps<1:raise ValueError('Positive integration steps required')
    b,n=features.shape[:2];device=features.device;rng=torch.Generator(device=device).manual_seed(seed);r,x=random_prior(b,n,device,rng)
    for i in range(steps):
        t=torch.full((b,),i/steps,device=device);pr,px=model(r,x,t,features);fraction=1./(steps-i);x=centered(x+(px-x)*fraction);r=so3_utils.geodesic_t(torch.full((b,1,1),fraction,device=device),pr,r)
    return backbone_from_frames(r,x)
