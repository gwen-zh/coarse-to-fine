"""Same fine networks/book, with a centroid-seeded transported coarse frame."""
import torch
from torch.nn import functional as F
from block_frame_tokenizer import BlockFrameFineTokenizer
from block_centroid_transport_frames import centroid_transport_frames


def stable_frame_context(centers,length,mode,block_size=4,frames=None,metadata=None):
    if frames is None:frames,metadata=centroid_transport_frames(centers)
    if mode not in ['reference','local'] or centers.shape!=((length+block_size-1)//block_size,3):raise ValueError('Invalid stable-frame context')
    basis=frames if mode=='local' else torch.eye(3,device=centers.device,dtype=centers.dtype)[None].expand(len(centers),-1,-1)
    previous=F.pad(centers[1:]-centers[:-1],(0,0,1,0));following=F.pad(centers[1:]-centers[:-1],(0,0,0,1));offsets=torch.cat([torch.einsum('mi,mij->mj',previous,basis),torch.einsum('mi,mij->mj',following,basis)],-1)/10.
    neighbor=torch.cat([torch.einsum('mji,mjk->mik',basis,F.pad(frames[:-1],(0,0,0,0,1,0))).flatten(1),torch.einsum('mji,mjk->mik',basis,F.pad(frames[1:],(0,0,0,0,0,1))).flatten(1)],-1)
    return dict(centers=centers,basis=basis,geometry=torch.cat([offsets,neighbor],-1),frame_metadata=metadata or {})


class StableBlockFrameFineTokenizer(BlockFrameFineTokenizer):
    def __init__(self,mode,block_size=4,fine_dim=8,fine_size=1024,frame_family='centroid_transport'):
        if frame_family!='centroid_transport':raise ValueError('Wrong frame-family checkpoint')
        super().__init__(mode,block_size,fine_dim,fine_size);self.config['frame_family']=frame_family
    def context(self,centers,length):return stable_frame_context(centers,length,self.mode,self.block_size)
