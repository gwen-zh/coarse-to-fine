"""Coarse-derived proper block frames without a fixed world-axis fallback.

A local tangent is combined with a regularized geometric normal. The normal
anchor is selected from the entire coarse trace by maximum off-tangent distance;
this is a block coordinate system, not a strictly local receptive-field claim.
An exactly collinear trace or fewer than three centers lacks the information
needed for a full SE(3)-equivariant frame and is rejected explicitly.
"""
import torch
from torch.nn import functional as F


class FrameGeometryError(ValueError):
    pass


def block_frames(centers,normal_regularizer=.05,min_relative_width=1e-6):
    if centers.ndim!=2 or centers.shape[1]!=3 or not centers.is_floating_point() or not torch.isfinite(centers).all():
        raise FrameGeometryError('Finite Mx3 coarse centers required')
    m=len(centers)
    if m<3:raise FrameGeometryError('At least three noncollinear coarse centers are required; no world-axis fallback')
    if normal_regularizer<=0 or min_relative_width<=0:raise ValueError('Positive frame regularizers required')
    following=F.pad(centers[1:]-centers[:-1],(0,0,0,1))
    previous=F.pad(centers[1:]-centers[:-1],(0,0,1,0))
    tangent=previous+following
    tangent=torch.cat([following[:1],tangent[1:-1],previous[-1:]],0)
    displacement=centers[None]-centers[:,None]
    distances=displacement.square().sum(-1)
    farthest=distances.argmax(-1);row=torch.arange(m,device=centers.device)
    global_direction=displacement[row,farthest]
    scale=distances.max(-1).values.sqrt()
    if (scale<=torch.finfo(centers.dtype).eps).any():raise FrameGeometryError('Collapsed center trace')
    local_scale=torch.maximum(previous.norm(dim=-1),following.norm(dim=-1))
    choice=torch.where((following.norm(dim=-1)>=previous.norm(dim=-1))[:,None],following,previous)
    choice=torch.where((choice.norm(dim=-1)<scale*1e-7)[:,None],global_direction,choice)
    fallback=tangent.norm(dim=-1)<scale*1e-7
    tangent=torch.where(fallback[:,None],choice,tangent)
    e1=F.normalize(tangent,dim=-1)
    off_tangent=displacement-(displacement*e1[:,None]).sum(-1,keepdim=True)*e1[:,None]
    normal_score=off_tangent.square().sum(-1)
    anchor=normal_score.argmax(-1);g=off_tangent[row,anchor]
    if (g.norm(dim=-1)<=min_relative_width*scale).any():
        raise FrameGeometryError('Center trace is collinear or too narrow to determine a stable normal')
    g=F.normalize(g,dim=-1)
    v=previous.clone();v[0]=following[1]
    v=v-(v*e1).sum(-1,keepdim=True)*e1
    # Positive projection onto the geometric normal prevents cancellation and
    # avoids a sign(v dot g) discontinuity at vanishing local curvature.
    tau=normal_regularizer*torch.where(local_scale>scale*1e-7,local_scale,scale)
    normal=v*(v*g).sum(-1,keepdim=True)+tau.square()[:,None]*g
    e2=F.normalize(normal,dim=-1)
    e3=F.normalize(torch.cross(e1,e2,dim=-1),dim=-1)
    e2=torch.cross(e3,e1,dim=-1)
    frames=torch.stack([e1,e2,e3],-1)
    return frames,dict(global_normal_anchor=anchor,tangent_fallback=fallback)


def vectors_to_local(values,frames,block_size=4):
    basis=frames.repeat_interleave(block_size,0)[:len(values)]
    if values.ndim!=3 or values.shape[1:]!=(3,3) or len(basis)!=len(values):raise ValueError('N x 3 atoms x xyz and matching block frames required')
    return torch.einsum('naj,njk->nak',values,basis)


def vectors_to_world(values,frames,block_size=4):
    basis=frames.repeat_interleave(block_size,0)[:len(values)]
    if values.ndim!=3 or values.shape[1:]!=(3,3) or len(basis)!=len(values):raise ValueError('N x 3 atoms x xyz and matching block frames required')
    return torch.einsum('naj,nkj->nak',values,basis)
