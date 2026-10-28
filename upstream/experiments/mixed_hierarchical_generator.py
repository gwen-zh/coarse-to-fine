"""Continuous coarse flow + discrete fine denoising, frozen learned structure decoder."""
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from hierarchical_rigid_vq import HierarchicalRigidVQ
from latent_flow_head import LatentHead,euler_sample,flow_path,protein_mean_mse
from two_level_autoencoder import pack,positions


class HybridTokenizer(nn.Module):
    """Matched coarse encoder/decoder from warmup, matched final fine components."""
    def __init__(self,model):
        super().__init__();self.model=model
        self.eval().requires_grad_(False)
    @classmethod
    def from_checkpoints(cls,warmup,final,device='cpu'):
        a=torch.load(warmup,map_location='cpu',weights_only=False);b=torch.load(final,map_location='cpu',weights_only=False)
        if a['model_config']!=b['model_config']:raise ValueError('Component model configurations differ')
        model=HierarchicalRigidVQ(**b['model_config']);model.load_state_dict(b['model'])
        for name in ['coarse_encoder','coarse_decoder']:
            model_state={k[len(name)+1:]:v for k,v in a['model'].items() if k.startswith(name+'.')}
            getattr(model,name).load_state_dict(model_state)
        return cls(model.to(device))
    def encode(self,backbone):
        c,f,_=self.model.encode_latents(backbone)
        return c,self.model.fine_book.nearest(f)
    def decode(self,coarse,fine):
        if fine.ndim!=1 or fine.dtype!=torch.long:raise ValueError('Fine IDs must be one unpadded integer sequence')
        if fine.min()<0 or fine.max()>=self.model.fine_book.size:raise ValueError('Fine ID outside codebook')
        return self.model.decode(coarse,self.model.fine_book.codes[fine],len(fine))['backbone']
    def coarse_centers(self,coarse,length):
        m=(length+3)//4
        if coarse.shape!=(m,self.model.coarse_dim):raise ValueError('Coarse latent/length mismatch')
        centers=self.model.coarse_decoder(coarse)*10.
        counts=torch.tensor([min(4,length-4*i) for i in range(m)],device=coarse.device,dtype=coarse.dtype)
        return centers-(centers*counts[:,None]).sum(0)/length


def pooled_sequence(features):
    values,mask=pack(features,4)
    return (values*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]


class ContinuousFineGenerator(nn.Module):
    def __init__(self,condition_dim=1536,coarse_dim=16,width=256,layers=6,fine_size=1024):
        super().__init__();self.coarse_dim=coarse_dim;self.fine_size=fine_size
        self.condition=nn.Sequential(nn.LayerNorm(condition_dim),nn.Linear(condition_dim,width))
        self.coarse_projection=nn.Sequential(nn.Linear(coarse_dim,width),nn.SiLU(),nn.Linear(width,width))
        self.fine_embedding=nn.Embedding(fine_size+1,width);self.type_embedding=nn.Embedding(2,width)
        self.time=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.Linear(width,width))
        layer=nn.TransformerEncoderLayer(width,8,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.layers=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,fine_size))
    def forward(self,features,coarse,fine,mask_rate):
        n=len(features);m=(n+3)//4
        if coarse.shape!=(m,self.coarse_dim) or not coarse.is_floating_point() or fine.shape!=(n,):raise ValueError('Continuous coarse and fine input lengths must match sequence')
        if not torch.isfinite(coarse).all() or fine.min()<0 or fine.max()>self.fine_size:raise ValueError('Invalid coarse or fine inputs')
        h=self.condition(features);c=pooled_sequence(h)+self.coarse_projection(coarse)+self.type_embedding.weight[0]
        f=h+self.fine_embedding(fine)+self.type_embedding.weight[1]
        c=c+positions(m,h.device,h.dtype,h.shape[-1]);f=f+positions(n,h.device,h.dtype,h.shape[-1])
        freq=torch.exp(torch.linspace(0,math.log(1000.),8,device=h.device,dtype=h.dtype));phase=torch.as_tensor(mask_rate,device=h.device,dtype=h.dtype)*freq
        time=self.time(torch.cat([phase.sin(),phase.cos()]))
        output=self.layers((torch.cat([c,f])+time)[None])[0]
        return self.head(output[m:])
    def denoise_loss(self,features,coarse,target,rate=None):
        rate=float(torch.rand((),device=features.device)*.9+.1) if rate is None else float(rate)
        if not 0<rate<=1:raise ValueError('Mask rate outside (0,1]')
        mask=torch.rand(len(target),device=target.device)<rate
        if not mask.any():mask[torch.randint(len(mask),(1,),device=target.device)]=True
        corrupted=torch.where(mask,self.fine_size,target)
        logits=self(features,coarse,corrupted,rate)
        return F.cross_entropy(logits[mask],target[mask])
    @torch.no_grad()
    def sample(self,features,coarse,seed,steps=16,temperature=1.):
        if steps<1 or temperature<=0:raise ValueError('Positive steps/temperature required')
        n=len(features);steps=min(steps,n);rng=torch.Generator(device=features.device).manual_seed(seed)
        fine=torch.full((n,),self.fine_size,device=features.device,dtype=torch.long)
        for i in range(steps):
            remaining=(fine==self.fine_size).nonzero().flatten()
            logits=self(features,coarse,fine,len(remaining)/n)[remaining]/temperature
            probs=logits.softmax(-1);proposals=torch.multinomial(probs,1,generator=rng).flatten()
            confidence=probs.gather(1,proposals[:,None]).flatten()
            count=min(len(remaining),max(1,math.ceil((i+1)*n/steps)-(n-len(remaining))))
            selected=confidence.topk(count).indices;fine[remaining[selected]]=proposals[selected]
        if (fine==self.fine_size).any():raise RuntimeError('Unfilled fine MASK after sampling')
        return fine


class MixedHierarchicalGenerator(nn.Module):
    def __init__(self,mean,std,condition_dim=1536,coarse_dim=16,width=256,coarse_layers=4,fine_layers=6,fine_size=1024):
        super().__init__();mean=torch.as_tensor(mean,dtype=torch.float32);std=torch.as_tensor(std,dtype=torch.float32)
        if mean.shape!=(coarse_dim,) or std.shape!=(coarse_dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise ValueError('Finite train-only normalization required')
        self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.config=dict(condition_dim=condition_dim,coarse_dim=coarse_dim,width=width,coarse_layers=coarse_layers,fine_layers=fine_layers,fine_size=fine_size)
        self.coarse_flow=LatentHead(condition_dim,coarse_dim,width,coarse_layers)
        self.fine=ContinuousFineGenerator(condition_dim,coarse_dim,width,fine_layers,fine_size)
    def normalize(self,c):return (c-self.mean)/self.std
    def denormalize(self,c):return c*self.std+self.mean
    def coarse_loss(self,features,target,tokenizer,geometry_weight=.1):
        truth=self.normalize(target)[None];noise=torch.randn_like(truth);t=torch.rand((1,),device=truth.device)
        state,velocity=flow_path(truth,noise,t);condition=pooled_sequence(features)[None]
        valid=torch.ones(truth.shape[:2],device=truth.device,dtype=torch.bool)
        prediction=self.coarse_flow(state,t,condition,valid);fm=protein_mean_mse(prediction,velocity,valid)
        geometry=fm*0.
        if float(t[0])>=.5 and geometry_weight:
            endpoint=state+(1-t[:,None,None])*prediction
            centers=tokenizer.coarse_centers(self.denormalize(endpoint[0]),len(features))
            with torch.no_grad():reference=tokenizer.coarse_centers(target,len(features))
            geometry=(centers-reference).square().mean()/100.
        return fm+geometry_weight*geometry,dict(flow=fm.detach(),geometry=geometry.detach())
    @torch.no_grad()
    def sample_coarse(self,features,seed,steps=16):
        condition=pooled_sequence(features)[None];rng=torch.Generator(device=features.device).manual_seed(seed)
        initial=torch.randn((1,len(condition[0]),len(self.mean)),device=features.device,generator=rng)
        valid=torch.ones(initial.shape[:2],device=features.device,dtype=torch.bool)
        state=euler_sample(self.coarse_flow,initial,condition,valid,steps=steps)
        return self.denormalize(state[0])
    @torch.no_grad()
    def sample(self,features,seed,coarse_steps=16,fine_steps=16,temperature=1.):
        coarse=self.sample_coarse(features,seed,coarse_steps)
        fine=self.fine.sample(features,self.normalize(coarse),(seed+104729)%(2**32),fine_steps,temperature)
        return coarse,fine
