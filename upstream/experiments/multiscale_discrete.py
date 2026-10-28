"""Matched six-bit geometric side channels and discrete original-fine completion.

Only true-structure encoder sees coordinates. Completer receives three hard
4-way codes, deterministic half-fine visibility and original visible fine IDs.
Geometry encoders never consume pretrained full-context fine latents.
"""
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_tokenizer import GeometryBlock,EMABottleneck,pair_geometry
from half_fine_tokenizer import position_features,retained_positions


def geometry_inputs(bb):
    pair,_=pair_geometry(bb);n=len(bb);idx=torch.arange(n,device=bb.device);sep=(idx[:,None]-idx[None]).abs()
    ca=bb[:,1];distance=torch.cdist(ca,ca)
    masks={'local':sep<=4,'middle':(sep<=4)|((sep<=16)&(distance<=12)),'global':torch.ones_like(sep,dtype=torch.bool)}
    intrinsic=torch.stack([(bb[:,0]-bb[:,1]).norm(dim=-1),(bb[:,2]-bb[:,1]).norm(dim=-1),(bb[:,0]-bb[:,2]).norm(dim=-1)],-1)/3.
    return intrinsic,pair,masks


class Branch(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(3,64);self.blocks=nn.ModuleList([GeometryBlock(64,4) for _ in range(2)]);self.output=nn.Linear(64,16);self.vq=EMABottleneck(4,16)
    def encode(self,intrinsic,pair,mask):
        x=self.input(intrinsic)
        for block in self.blocks:x=block(x,pair,mask)
        return self.output(x)


class MultiScaleDiscrete(nn.Module):
    def __init__(self,mode,vocab=4096):
        super().__init__()
        if mode not in ['local','global','mixed']:raise ValueError(mode)
        self.mode=mode;self.branches=nn.ModuleList([Branch() for _ in range(3)])
        self.input=nn.Linear(128+48+32+1,128)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,vocab))
    @property
    def scales(self):return ['local','middle','global'] if self.mode=='mixed' else [self.mode]*3
    def encode(self,intrinsic,pair,masks):
        features=[];ids=[];commit=intrinsic.new_tensor(0.)
        for branch,scale in zip(self.branches,self.scales):
            q,s,c=branch.vq(branch.encode(intrinsic,pair,masks[scale]));features.append(q);ids.append(s);commit=commit+c
        return torch.cat(features,-1),torch.stack(ids,-1),commit
    def complete(self,visible_ids,keep,coarse,fine_codes):
        if torch.any(visible_ids[~keep]!=0):raise ValueError('Missing fine IDs must be sanitized')
        z=torch.where(keep[:,None],fine_codes[visible_ids],torch.zeros_like(fine_codes[visible_ids]))
        x=torch.cat([F.layer_norm(z,(128,)),coarse*4,position_features(len(z),z.device),keep[:,None].to(z.dtype)],-1)
        return self.head(self.context(self.input(x)[None])[0])


def hard_embeddings(logits,visible_ids,keep,table):
    ids=torch.where(keep,visible_ids,logits.argmax(-1));hard=table[ids]
    soft=logits.softmax(-1)@table
    # Exactly hard forward; surrogate gradient exists only at hidden positions.
    result=hard+torch.where(keep[:,None],torch.zeros_like(soft),soft-soft.detach())
    return result,ids


def nominal_rate(n):return (6*n+12*((n+1)//2))/n
