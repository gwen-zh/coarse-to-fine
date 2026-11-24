"""Geometry-only candidate: centroid-seeded transport along coarse tangents."""
import torch
from torch.nn import functional as F
from block_local_frames import block_frames,FrameGeometryError


def centroid_transport_frames(centers):
    initial,_=block_frames(centers);tangents=initial[:,:,0];t=tangents[0];delta=centers-centers[0];scale=delta.norm(dim=-1).max()
    m=len(centers);index=torch.linspace(1.,2.,m,device=centers.device,dtype=centers.dtype);normal=None;selected=None
    for number,weight in enumerate([torch.ones_like(index),index,index.square()]):
        vector=(delta*weight[:,None]).sum(0)/weight.sum();vector=vector-(vector*t).sum()*t
        if float(vector.norm().detach())>float((.001*scale).detach()):normal=F.normalize(vector,dim=-1);selected=number;break
    if normal is None:
        perpendicular=delta-(delta*t).sum(-1,keepdim=True)*t;normal=perpendicular[perpendicular.square().sum(-1).argmax()]
        if normal.norm()<=1e-6*scale:raise FrameGeometryError('No geometric normal exists')
        normal=F.normalize(normal,dim=-1);selected=3
    binormal=F.normalize(torch.cross(t,normal,dim=-1),dim=-1);normal=torch.cross(binormal,t,dim=-1);frames=[torch.stack([t,normal,binormal],-1)];halfturns=0
    for i in range(1,m):
        previous=tangents[i-1];following=tangents[i]
        if float((previous*following).sum().detach())<-.99:previous=-previous;halfturns+=1
        axis=torch.cross(previous,following,dim=-1);cosine=(previous*following).sum().clamp(-1,1)
        moved=normal+torch.cross(axis,normal,dim=-1)+torch.cross(axis,torch.cross(axis,normal,dim=-1),dim=-1)/(1+cosine).clamp_min(1e-8)
        normal=F.normalize(moved-(moved*following).sum()*following,dim=-1);binormal=F.normalize(torch.cross(following,normal,dim=-1),dim=-1);normal=torch.cross(binormal,following,dim=-1);frames.append(torch.stack([following,normal,binormal],-1))
    return torch.stack(frames),dict(seed_rule=selected,near_antiparallel_transports=halfturns)
