"""Trainable two-level structure tokenizer; explicit scale split, rigid residue head.

No ESM weights or structure latents. Geometry constants match the published
backbone template only. Canonicalization removes arbitrary input orientation;
token-only decoding never reads a reference frame, coordinates or sequence.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_latent_contract import pool_ca, compose_levels, remove_block_translation
from two_level_autoencoder import pack, positions

BB_LOCAL = [[.5256,1.3612,0.],[0.,0.,0.],[-1.5251,0.,0.]]


def orthogonal_frame(x, y):
    eps=1e-6
    e=torch.zeros_like(x);e[...,0]=1.
    x=torch.where((x.norm(dim=-1,keepdim=True)<eps),e,x)
    x=F.normalize(x,dim=-1,eps=eps)
    y=y-(x*y).sum(-1,keepdim=True)*x
    fallback=F.one_hot(x.abs().argmin(-1),3).to(x.dtype)
    fallback=fallback-(fallback*x).sum(-1,keepdim=True)*x
    y=torch.where(y.norm(dim=-1,keepdim=True)<eps,fallback,y)
    y=F.normalize(y,dim=-1,eps=eps)
    z=F.normalize(torch.cross(x,y,dim=-1),dim=-1,eps=eps)
    y=torch.cross(z,x,dim=-1)
    return torch.stack([x,y,z],-1)


def canonical_backbone(x):
    if x.ndim!=3 or x.shape[1:]!=(3,3) or not torch.isfinite(x).all():raise ValueError('Complete finite N/CA/C chain required')
    # First residue fixes orientation; sign matches the local template C direction.
    frame=orthogonal_frame(x[0,1]-x[0,2],x[0,0]-x[0,1])
    return (x-x[:,1].mean(0))@frame


def rigid_backbone(raw):
    if raw.ndim!=2 or raw.shape[1]!=9:raise ValueError('Expected residue translation3 + rotation6')
    bias=raw.new_tensor([1.,0.,0.,0.,1.,0.])
    orientation=raw[:,3:]+bias
    frame=orthogonal_frame(orientation[:,:3],orientation[:,3:])
    return torch.einsum('nij,aj->nai',frame,raw.new_tensor(BB_LOCAL))+4.*raw[:,:3,None].transpose(1,2)


class GlobalMap(nn.Module):
    def __init__(self,inputs,outputs,width=128,layers=4):
        super().__init__();self.input=nn.Linear(inputs+16,width)
        layer=nn.TransformerEncoderLayer(width,4,4*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.layers=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,outputs))
    def forward(self,x):
        h=self.input(torch.cat([x,positions(len(x),x.device,x.dtype)],-1))
        return self.output(self.layers(h[None])[0])


class EMAVectorQuantizer(nn.Module):
    def __init__(self,size,dim,decay=.99,commitment=.25,dead_threshold=.01):
        super().__init__();self.size=size;self.dim=dim;self.decay=decay;self.commitment=commitment;self.dead_threshold=dead_threshold
        self.register_buffer('codes',torch.zeros(size,dim));self.register_buffer('counts',torch.ones(size))
        self.register_buffer('sums',torch.zeros(size,dim));self.register_buffer('initialized',torch.tensor(False))
        self.register_buffer('restarts',torch.tensor(0,dtype=torch.long))
    def nearest(self,z):
        return (z.square().sum(-1,keepdim=True)-2*z@self.codes.T+self.codes.square().sum(-1)).argmin(-1)
    @torch.no_grad()
    def initialize(self,z,iterations=10):
        if len(z)<self.size:raise ValueError('Too few TRAIN latent vectors for codebook initialization')
        centers=z[torch.randperm(len(z),device=z.device)[:self.size]].clone()
        for _ in range(iterations):
            sums=torch.zeros_like(centers);counts=torch.zeros(self.size,device=z.device,dtype=z.dtype)
            for chunk in z.split(4096):
                ids=(chunk.square().sum(-1,keepdim=True)-2*chunk@centers.T+centers.square().sum(-1)).argmin(-1)
                sums.index_add_(0,ids,chunk);counts+=torch.bincount(ids,minlength=self.size)
            centers=torch.where((counts>0)[:,None],sums/counts.clamp_min(1)[:,None],centers)
        self.codes.copy_(centers);self.sums.copy_(centers);self.counts.fill_(1);self.initialized.fill_(True)
    def forward(self,z):
        if not self.initialized:raise RuntimeError('Codebook must be initialized from training data')
        ids=self.nearest(z);q=self.codes.detach().clone()[ids]
        penalty=self.commitment*(z-q).square().mean()
        if self.training:
            with torch.no_grad():
                counts=torch.bincount(ids,minlength=self.size).to(z.dtype)
                sums=torch.zeros_like(self.codes).index_add_(0,ids,z)
                self.counts.mul_(self.decay).add_(counts,alpha=1-self.decay)
                self.sums.mul_(self.decay).add_(sums,alpha=1-self.decay)
                self.codes.copy_(self.sums/self.counts.clamp_min(1e-5)[:,None])
                dead=self.counts<self.dead_threshold
                if dead.any():
                    selected=torch.randint(len(z),(int(dead.sum()),),device=z.device)
                    self.codes[dead]=z[selected];self.sums[dead]=z[selected];self.counts[dead]=1.
                    self.restarts.add_(dead.sum())
        return q+(z-z.detach()),ids,penalty


class HierarchicalRigidVQ(nn.Module):
    def __init__(self,width=128,layers=4,coarse_dim=16,fine_dim=8,coarse_size=256,fine_size=1024,block_size=4):
        super().__init__();self.block_size=block_size;self.coarse_dim=coarse_dim;self.fine_dim=fine_dim
        self.config=dict(width=width,layers=layers,coarse_dim=coarse_dim,fine_dim=fine_dim,coarse_size=coarse_size,fine_size=fine_size,block_size=block_size)
        self.coarse_encoder=GlobalMap(3,coarse_dim,width,layers);self.coarse_decoder=GlobalMap(coarse_dim,3,width,layers)
        self.fine_encoder=nn.Sequential(nn.Linear(block_size*10,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,block_size*fine_dim))
        # own fine, center offsets, own mask, previous/next fine+mask.
        inputs=block_size*fine_dim+6+block_size+2*block_size*(fine_dim+1)
        self.fine_decoder=nn.Sequential(nn.Linear(inputs,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,block_size*9))
        self.coarse_book=EMAVectorQuantizer(coarse_size,coarse_dim);self.fine_book=EMAVectorQuantizer(fine_size,fine_dim)
    def encode_latents(self,x):
        target=canonical_backbone(x);centers=pool_ca(target,self.block_size);residual=remove_block_translation(target,self.block_size)
        c=torch.tanh(self.coarse_encoder(centers/10.))
        packed,mask=pack((residual/4.).flatten(1),self.block_size)
        f=torch.tanh(self.fine_encoder(torch.cat([packed.flatten(1),mask.to(x.dtype)],-1))).reshape(-1,self.fine_dim)[:len(x)]
        return c,f,target
    def decode(self,c,f,length):
        packed,mask=pack(f,self.block_size)
        if c.shape!=(len(mask),self.coarse_dim) or f.shape!=(length,self.fine_dim):raise ValueError('Latent lengths do not match chain')
        centers=self.coarse_decoder(c)*10.;counts=mask.sum(-1).to(c.dtype)
        centers=centers-(centers*counts[:,None]).sum(0)/length
        previous=F.pad(centers[1:]-centers[:-1],(0,0,1,0))/10.
        following=F.pad(centers[1:]-centers[:-1],(0,0,0,1))/10.
        blocks=torch.cat([packed.flatten(1),mask.to(f.dtype)],-1)
        inputs=torch.cat([packed.flatten(1),previous,following,mask.to(f.dtype),F.pad(blocks[:-1],(0,0,1,0)),F.pad(blocks[1:],(0,0,0,1))],-1)
        raw=self.fine_decoder(inputs).reshape(-1,9)[:length]
        backbone=compose_levels(centers,rigid_backbone(raw),self.block_size)
        return dict(backbone=backbone,coarse=centers,fine_residual=remove_block_translation(backbone,self.block_size))
    def forward(self,x,quantized=False,quantization_fraction=1.):
        c,f,target=self.encode_latents(x);loss=c.sum()*0.;ids=None
        if quantized:
            qc,ic,lc=self.coarse_book(c);qf,iff,lf=self.fine_book(f)
            a=quantization_fraction;c_used=(1-a)*c+a*qc;f_used=(1-a)*f+a*qf;loss=lc+lf;ids=(ic,iff)
        else:c_used,f_used=c,f
        result=self.decode(c_used,f_used,len(x))
        result.update(z_c=c,z_f=f,target=target,commitment=loss,ids=ids)
        return result
    @torch.no_grad()
    def tokens(self,x):
        c,f,_=self.encode_latents(x)
        if not self.coarse_book.initialized or not self.fine_book.initialized:raise RuntimeError('Uninitialized codebooks')
        return self.coarse_book.nearest(c),self.fine_book.nearest(f)
    def decode_tokens(self,coarse,fine):
        if coarse.ndim!=1 or fine.ndim!=1:raise ValueError('One unpadded token chain required')
        return self.decode(self.coarse_book.codes[coarse],self.fine_book.codes[fine],len(fine))


def tokenizer_loss(output,weights):
    from two_level_autoencoder import training_loss
    loss,terms=training_loss(output,output['target'],weights)
    # Standalone chain geometry terms; no extra boundary-only weighting.
    loss=loss+output['commitment'];terms=dict(**terms,commitment=output['commitment'])
    return loss,terms
