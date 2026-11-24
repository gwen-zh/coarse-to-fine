"""Follow-on geometry prototype: transport normals along coarse tangents.

Seed orientation comes from coarse geometry once, then normals are propagated
by proper rotations instead of choosing a new farthest normal anchor per block.
This file does not alter the frozen v1 tokenizer or its training results.
"""
import torch
from torch.nn import functional as F
from block_local_frames import block_frames


def transported_block_frames(centers):
    initial,metadata=block_frames(centers)
    tangent=initial[:,:,0];normal=initial[0,:,1];frames=[initial[0]];halfturns=0
    for i in range(1,len(centers)):
        previous=tangent[i-1];following=tangent[i];cosine=(previous*following).sum().clamp(-1.,1.)
        if float(cosine.detach())<-.99:
            # A half-turn around the previous normal changes t to -t while
            # preserving that normal; then a small proper rotation completes it.
            previous=-previous;halfturns+=1
        axis=torch.cross(previous,following,dim=-1);cosine=(previous*following).sum().clamp(-1.,1.)
        rotated=normal+torch.cross(axis,normal,dim=-1)+torch.cross(axis,torch.cross(axis,normal,dim=-1),dim=-1)/(1.+cosine).clamp_min(1e-8)
        normal=F.normalize(rotated-(rotated*following).sum()*following,dim=-1)
        binormal=F.normalize(torch.cross(following,normal,dim=-1),dim=-1);normal=torch.cross(binormal,following,dim=-1)
        frames.append(torch.stack([following,normal,binormal],-1))
    return torch.stack(frames),dict(seed_normal_anchor=metadata['global_normal_anchor'][0],near_antiparallel_transports=halfturns)
