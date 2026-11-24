"""Capacity-matched fine tokenizers: reference-frame vs block-local-frame.

Both use the same fresh E/D topology, 8D fine latent, 1024 EMA book and 18
additional geometric features. Coarse E/D remain frozen and supply centers.
Fine encoding and decoding derive frames from the SAME reconstructed centers.
"""
import torch
from torch import nn
from torch.nn import functional as F
from block_local_frames import block_frames,vectors_to_local,vectors_to_world
from hierarchical_rigid_vq import EMAVectorQuantizer,rigid_backbone
from hierarchical_latent_contract import compose_levels,remove_block_translation
from two_level_autoencoder import pack


def frame_context(centers,length,mode,block_size=4):
    if mode not in ['reference','local']:raise ValueError('Unknown frame convention')
    frames,meta=block_frames(centers)
    basis=frames if mode=='local' else torch.eye(3,device=centers.device,dtype=centers.dtype)[None].expand(len(centers),-1,-1)
    previous=F.pad(centers[1:]-centers[:-1],(0,0,1,0));following=F.pad(centers[1:]-centers[:-1],(0,0,0,1))
    offsets=torch.cat([torch.einsum('mi,mij->mj',previous,basis),torch.einsum('mi,mij->mj',following,basis)],-1)/10.
    neighbor_frames=torch.cat([torch.einsum('mji,mjk->mik',basis,F.pad(frames[:-1],(0,0,0,0,1,0))).flatten(1),torch.einsum('mji,mjk->mik',basis,F.pad(frames[1:],(0,0,0,0,0,1))).flatten(1)],-1)
    if centers.shape!=((length+block_size-1)//block_size,3):raise ValueError('Coarse centers do not match residue length')
    return dict(centers=centers,basis=basis,geometry=torch.cat([offsets,neighbor_frames],-1),frame_metadata=meta)


class BlockFrameFineTokenizer(nn.Module):
    def __init__(self,mode,block_size=4,fine_dim=8,fine_size=1024):
        super().__init__()
        if mode not in ['reference','local']:raise ValueError('Unknown fine frame mode')
        self.mode=mode;self.block_size=block_size;self.fine_dim=fine_dim
        self.config=dict(mode=mode,block_size=block_size,fine_dim=fine_dim,fine_size=fine_size)
        self.encoder=nn.Sequential(nn.Linear(block_size*10,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,block_size*fine_dim))
        inputs=block_size*fine_dim+block_size+2*block_size*(fine_dim+1)+24
        self.decoder=nn.Sequential(nn.Linear(inputs,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,block_size*9))
        self.book=EMAVectorQuantizer(fine_size,fine_dim)
    def context(self,centers,length):return frame_context(centers,length,self.mode,self.block_size)
    def encode(self,target,context):
        residual=remove_block_translation(target,self.block_size)
        values=vectors_to_local(residual,context['basis'],self.block_size)/4.
        packed,mask=pack(values.flatten(1),self.block_size)
        return torch.tanh(self.encoder(torch.cat([packed.flatten(1),mask.to(target.dtype)],-1))).reshape(-1,self.fine_dim)[:len(target)]
    def decode_vectors(self,fine,context,length):
        packed,mask=pack(fine,self.block_size);blocks=torch.cat([packed.flatten(1),mask.to(fine.dtype)],-1)
        inputs=torch.cat([packed.flatten(1),mask.to(fine.dtype),F.pad(blocks[:-1],(0,0,1,0)),F.pad(blocks[1:],(0,0,0,1)),context['geometry']],-1)
        raw=self.decoder(inputs).reshape(-1,9)[:length]
        relative=vectors_to_world(rigid_backbone(raw),context['basis'],self.block_size)
        return compose_levels(context['centers'],relative,self.block_size)
    def forward(self,target,context,quantized=False):
        fine=self.encode(target,context);penalty=fine.sum()*0.;ids=None
        if quantized:q,ids,penalty=self.book(fine)
        else:q=fine
        return dict(backbone=self.decode_vectors(q,context,len(target)),fine_continuous=fine,fine_ids=ids,commitment=penalty)
    def decode_tokens(self,coarse,fine_ids,coarse_tokenizer):
        centers=coarse_tokenizer.coarse_centers(coarse,len(fine_ids));context=self.context(centers,len(fine_ids))
        return self.decode_vectors(self.book.codes[fine_ids],context,len(fine_ids))
