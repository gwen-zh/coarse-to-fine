"""Rigid-invariant native-fragment descriptors for representation diagnostics."""
import numpy as np


def dihedral(a,b,c,d):
    a,b,c,d=[np.asarray(x,dtype=np.float64) for x in (a,b,c,d)]
    axis=c-b
    norm=np.linalg.norm(axis,axis=-1,keepdims=True)
    if np.any(norm<1e-10):raise ValueError('Degenerate dihedral axis')
    axis=axis/norm
    u=a-b;v=d-c
    u=u-(u*axis).sum(-1,keepdims=True)*axis
    v=v-(v*axis).sum(-1,keepdims=True)*axis
    if np.any(np.linalg.norm(u,axis=-1)<1e-10) or np.any(np.linalg.norm(v,axis=-1)<1e-10):raise ValueError('Degenerate dihedral plane')
    return np.arctan2((np.cross(axis,u)*v).sum(-1),(u*v).sum(-1))


def fragment_descriptors(backbone):
    """All interior residues; a 3-residue N/CA/C fragment and central phi/psi.

    Frame uses central CA origin, central CA->C positive x, and N in +xy.
    Rotation/translation removed using the central residue only. Neighbor
    geometry remains unconstrained, including any native connection outliers.
    """
    x=np.asarray(backbone,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(3,3) or len(x)<3 or not np.isfinite(x).all():raise ValueError('Complete finite N/CA/C chain required')
    centers=np.arange(1,len(x)-1)
    origin=x[centers,1]
    ex=x[centers,2]-origin;exnorm=np.linalg.norm(ex,axis=-1,keepdims=True)
    if np.any(exnorm<1e-10):raise ValueError('Degenerate CA-C vector')
    ex=ex/exnorm
    ey=x[centers,0]-origin;ey=ey-(ey*ex).sum(-1,keepdims=True)*ex
    eynorm=np.linalg.norm(ey,axis=-1,keepdims=True)
    if np.any(eynorm<1e-10):raise ValueError('Degenerate N-CA-C frame')
    ey=ey/eynorm;ez=np.cross(ex,ey)
    frame=np.stack([ex,ey,ez],axis=-1)
    fragment=x[centers[:,None]+np.arange(-1,2)[None,:]]-origin[:,None,None,:]
    local=np.einsum('nraj,njk->nrak',fragment,frame)
    phi=dihedral(x[centers-1,2],x[centers,0],x[centers,1],x[centers,2])
    psi=dihedral(x[centers,0],x[centers,1],x[centers,2],x[centers+1,0])
    return dict(residue=centers,slot=centers%4,local=local,angles=np.stack([phi,psi],axis=-1))


def circular_error_degrees(a,b):
    return np.degrees(np.abs((np.asarray(a)-np.asarray(b)+np.pi)%(2*np.pi)-np.pi))


def fit_templates(local,angles,slot,code):
    """TRAIN-only means. Pair index=4*code+slot, unseen pairs fall back to slot."""
    local=np.asarray(local,dtype=np.float64);angles=np.asarray(angles,dtype=np.float64)
    slot=np.asarray(slot,dtype=np.int64);code=np.asarray(code,dtype=np.int64)
    if local.shape!=(len(code),3,3,3) or angles.shape!=(len(code),2) or slot.shape!=code.shape or not len(code):raise ValueError('Mismatched descriptors')
    if np.any(slot<0) or np.any(slot>3) or np.any(code<0) or np.any(code>=1024):raise ValueError('Invalid slot/code')
    def collect(index,size):
        count=np.bincount(index,minlength=size)
        shape_sum=np.zeros((size,3,3,3));angle_sum=np.zeros((size,2,2))
        np.add.at(shape_sum,index,local)
        angle_vector=np.stack([np.sin(angles),np.cos(angles)],axis=-1)
        np.add.at(angle_sum,index,angle_vector)
        return dict(count=count,local=shape_sum/np.maximum(count,1)[:,None,None,None],angles=np.arctan2(angle_sum[...,0],angle_sum[...,1]))
    return dict(slot=collect(slot,4),pair=collect(4*code+slot,4096))


def predict_templates(fit,slot,code,use_code=True):
    slot=np.asarray(slot,dtype=np.int64);pair=4*np.asarray(code,dtype=np.int64)+slot
    shape=fit['slot']['local'][slot].copy();angles=fit['slot']['angles'][slot].copy()
    if use_code:
        seen=fit['pair']['count'][pair]>0
        shape[seen]=fit['pair']['local'][pair[seen]];angles[seen]=fit['pair']['angles'][pair[seen]]
    return shape,angles
