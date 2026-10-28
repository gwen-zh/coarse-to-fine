"""Trainable continuous two-level backbone AE, no ESM/codebook/old decoder.

Translation centered. Rotation augmentation is a training choice, not an exact
network equivariance claim. Coarse sees block CA centers only; fine sees H(X)
within independent blocks. Decoder reads latents and chain length only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_latent_contract import pool_ca,lift_anchors,remove_block_translation,compose_levels


def positions(n,device,dtype,dim=16):
    p=torch.arange(n,device=device,dtype=dtype)[:,None];f=torch.exp(-math.log(10000)*torch.arange(dim//2,device=device,dtype=dtype)/(dim//2-1))
    return torch.cat([(p*f).sin(),(p*f).cos()],-1)


def block_layout(n,device,block_size=4):
    if n<1:raise ValueError('Empty chain')
    m=(n+block_size-1)//block_size
    return torch.arange(m*block_size,device=device).reshape(m,block_size)<n


def pack(values,block_size=4):
    n=len(values);mask=block_layout(n,values.device,block_size);padded=F.pad(values,(0,0,0,mask.numel()-n))
    return padded.reshape(len(mask),block_size,-1),mask


class GlobalMap(nn.Module):
    def __init__(self,inputs,outputs,width=64):
        super().__init__();self.input=nn.Linear(inputs+16,width)
        layer=nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,outputs))
    def forward(self,x):
        h=self.input(torch.cat([x,positions(len(x),x.device,x.dtype)],-1))
        return self.output(self.context(h[None])[0])


class TwoLevelAutoencoder(nn.Module):
    def __init__(self,coarse_dim=8,fine_dim=4,block_size=4):
        super().__init__();self.coarse_dim=coarse_dim;self.fine_dim=fine_dim;self.block_size=block_size
        self.coarse_encoder=GlobalMap(3,coarse_dim);self.coarse_decoder=GlobalMap(coarse_dim,3)
        self.fine_encoder=nn.Sequential(nn.Linear(block_size*9+block_size,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,block_size*fine_dim))
        self.fine_decoder=nn.Sequential(nn.Linear(block_size*fine_dim+6+block_size,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,block_size*9))
    def encode_coarse(self,centers):return torch.tanh(self.coarse_encoder(centers/10.))
    def encode_fine(self,residual):
        packed,mask=pack((residual/4.).flatten(1),self.block_size)
        z=torch.tanh(self.fine_encoder(torch.cat([packed.flatten(1),mask.to(residual.dtype)],-1)))
        return z.reshape(-1,self.fine_dim)[:len(residual)]
    def encode(self,backbone):
        if backbone.ndim!=3 or backbone.shape[1:]!=(3,3) or not torch.isfinite(backbone).all():raise ValueError('Expected finite complete N/CA/C chain')
        centered=backbone-backbone[:,1].mean(0)
        return self.encode_coarse(pool_ca(centered,self.block_size)),self.encode_fine(remove_block_translation(centered,self.block_size))
    def decode_coarse(self,z_c,length):
        mask=block_layout(length,z_c.device,self.block_size)
        if z_c.shape!=(len(mask),self.coarse_dim):raise ValueError('Coarse latent shape')
        c=self.coarse_decoder(z_c)*10.;counts=mask.sum(-1).to(c.dtype)
        return c-(c*counts[:,None]).sum(0)/length
    def decode_with_anchors(self,centers,z_f,length):
        if z_f.shape!=(length,self.fine_dim):raise ValueError('Fine latent shape')
        packed,mask=pack(z_f,self.block_size)
        if centers.shape!=(len(mask),3):raise ValueError('Anchor shape')
        previous=F.pad(centers[1:]-centers[:-1],(0,0,1,0))/10.
        following=F.pad(centers[1:]-centers[:-1],(0,0,0,1))/10.
        raw=self.fine_decoder(torch.cat([packed.flatten(1),previous,following,mask.to(z_f.dtype)],-1))
        raw=raw.reshape(-1,3,3)[:length]*4.
        output=compose_levels(centers,raw,self.block_size)
        return dict(backbone=output,coarse=centers,fine_residual=remove_block_translation(raw,self.block_size))
    def decode(self,z_c,z_f,length):return self.decode_with_anchors(self.decode_coarse(z_c,length),z_f,length)
    def forward(self,backbone):
        z_c,z_f=self.encode(backbone);output=self.decode(z_c,z_f,len(backbone));output.update(z_c=z_c,z_f=z_f);return output


def geometry_terms(pred,true):
    def bonds(x):return torch.cat([(x[:,0]-x[:,1]).norm(dim=-1),(x[:,1]-x[:,2]).norm(dim=-1),(x[:-1,2]-x[1:,0]).norm(dim=-1),(x[:-1,1]-x[1:,1]).norm(dim=-1)])
    def cosines(x):
        cos=lambda a,b:(F.normalize(a,dim=-1,eps=1e-6)*F.normalize(b,dim=-1,eps=1e-6)).sum(-1)
        return torch.cat([cos(x[:,0]-x[:,1],x[:,2]-x[:,1]),cos(x[:-1,1]-x[:-1,2],x[1:,0]-x[:-1,2]),cos(x[:-1,2]-x[1:,0],x[1:,1]-x[1:,0])])
    b=(bonds(pred)-bonds(true)).square().mean();a=(cosines(pred)-cosines(true)).square().mean()
    ids=torch.arange(len(pred),device=pred.device);mask=(ids[:,None]-ids[None]).abs()>1
    distances=torch.cdist(pred[:,1],pred[:,1]);clash=F.relu(3.-distances[mask]).square().mean() if mask.any() else pred.sum()*0
    return b,a,clash


def training_loss(output,reference,weights):
    target=reference-reference[:,1].mean(0);c=pool_ca(target);r=remove_block_translation(target)
    coarse=(output['coarse']-c).square().mean()/100.;fine=(output['fine_residual']-r).square().mean()/16.;full=(output['backbone']-target).square().mean()/100.
    bond,angle,clash=geometry_terms(output['backbone'],target);regularizer=output['z_c'].square().mean()+output['z_f'].square().mean()
    terms=dict(coarse=coarse,fine=fine,full=full,bond=bond,angle=angle,clash=clash,latent=regularizer)
    return sum(weights[k]*v for k,v in terms.items()),terms
