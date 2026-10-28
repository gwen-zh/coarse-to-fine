"""Small continuous-latent heads conditioned on frozen original-path features."""
import math
import torch
from torch import nn


class LatentHead(nn.Module):
    def __init__(self,condition_dim=1536,latent_dim=128,width=256,layers=4):
        super().__init__();self.width=width
        self.condition=nn.Sequential(nn.LayerNorm(condition_dim),nn.Linear(condition_dim,width))
        self.state=nn.Linear(latent_dim,width)
        self.time=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,width*4,dropout=0.,batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
        self.output=nn.Linear(width,latent_dim);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,state,t,condition,valid):
        frequencies=torch.exp(torch.arange(32,device=t.device,dtype=t.dtype)*(-math.log(10000.)/31))
        phase=t[:,None]*frequencies[None,:]*1000.
        time=self.time(torch.cat([phase.sin(),phase.cos()],-1))
        x=self.condition(condition)+self.state(state)+time[:,None]
        position=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[:,None]
        frequency=torch.exp(torch.arange(self.width//2,device=x.device,dtype=x.dtype)*(-math.log(10000.)/max(1,self.width//2-1)))
        phase=position*frequency[None,:]
        x=x+torch.cat([phase.sin(),phase.cos()],-1)[None]
        return self.output(self.blocks(x,src_key_padding_mask=~valid))


def flow_path(target,noise,t):
    t=t[:,None,None]
    return (1-t)*noise+t*target,target-noise


def protein_mean_mse(prediction,target,valid):
    per_site=(prediction-target).square().mean(-1)
    return ((per_site*valid).sum(-1)/valid.sum(-1)).mean()


@torch.no_grad()
def euler_sample(model,initial,condition,valid,steps=16):
    if steps<1:raise ValueError("Positive number of ODE steps required")
    state=initial.clone()
    for i in range(steps):
        t=torch.full((len(state),),i/steps,device=state.device,dtype=state.dtype)
        state=state+model(state,t,condition,valid)/steps
    return state
