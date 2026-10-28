"""Sequence-conditioned, two-level masked denoising with an actual fine input track.

Inference receives amino-acid-only features, chain length, RNG and generated
codes. Teacher codes are supervision during training, never sampling inputs.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from two_level_autoencoder import pack,positions


class HierarchicalTokenGenerator(nn.Module):
    def __init__(self,condition_dim=1536,width=256,layers=6,coarse_size=256,fine_size=1024,block_size=4):
        super().__init__();self.coarse_size=coarse_size;self.fine_size=fine_size;self.block_size=block_size
        self.config=dict(condition_dim=condition_dim,width=width,layers=layers,coarse_size=coarse_size,fine_size=fine_size,block_size=block_size)
        self.condition=nn.Sequential(nn.LayerNorm(condition_dim),nn.Linear(condition_dim,width))
        self.coarse_embedding=nn.Embedding(coarse_size+1,width);self.fine_embedding=nn.Embedding(fine_size+1,width)
        self.type_embedding=nn.Embedding(2,width);self.task_embedding=nn.Embedding(2,width)
        self.time=nn.Sequential(nn.Linear(16,width),nn.SiLU(),nn.Linear(width,width))
        layer=nn.TransformerEncoderLayer(width,8,4*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.layers=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.coarse_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,coarse_size))
        self.fine_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,fine_size))
    def forward(self,features,coarse,fine,mask_rate,stage):
        n=len(features);m=(n+self.block_size-1)//self.block_size
        if features.ndim!=2 or coarse.shape!=(m,) or fine.shape!=(n,):raise ValueError('One matching unpadded sequence required')
        if stage not in ['coarse','fine']:raise ValueError('Unknown level')
        if coarse.min()<0 or coarse.max()>self.coarse_size or fine.min()<0 or fine.max()>self.fine_size:raise ValueError('Invalid code/mask ID')
        h=self.condition(features);blocks,valid=pack(h,self.block_size);c=(blocks*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        freq=torch.exp(torch.linspace(0,math.log(1000),8,device=h.device,dtype=h.dtype))
        t=torch.as_tensor(mask_rate,device=h.device,dtype=h.dtype)*freq
        time=self.time(torch.cat([t.sin(),t.cos()]))+self.task_embedding.weight[0 if stage=='coarse' else 1]
        c=c+self.coarse_embedding(coarse)+self.type_embedding.weight[0]+positions(m,h.device,h.dtype,h.shape[-1])
        f=h+self.fine_embedding(fine)+self.type_embedding.weight[1]+positions(n,h.device,h.dtype,h.shape[-1])
        output=self.layers((torch.cat([c,f],0)+time)[None])[0]
        return self.coarse_head(output[:m]),self.fine_head(output[m:])
    def denoise_loss(self,features,true_coarse,true_fine,stage,coarse_condition=None,rate=None):
        device=features.device
        rate=float(torch.rand((),device=device)*.9+.1) if rate is None else float(rate)
        if not 0<rate<=1:raise ValueError('Mask rate outside (0,1]')
        if stage=='coarse':
            mask=torch.rand(len(true_coarse),device=device)<rate
            if not mask.any():mask[torch.randint(len(mask),(1,),device=device)]=True
            coarse=torch.where(mask,self.coarse_size,true_coarse);fine=torch.full_like(true_fine,self.fine_size)
            logits,_=self(features,coarse,fine,rate,stage);target=true_coarse
        else:
            mask=torch.rand(len(true_fine),device=device)<rate
            if not mask.any():mask[torch.randint(len(mask),(1,),device=device)]=True
            coarse=true_coarse if coarse_condition is None else coarse_condition
            fine=torch.where(mask,self.fine_size,true_fine)
            _,logits=self(features,coarse,fine,rate,stage);target=true_fine
        return F.cross_entropy(logits[mask],target[mask]),dict(masked=int(mask.sum()),total=len(mask),rate=rate)
    @torch.no_grad()
    def _fill(self,features,coarse,fine,stage,steps,temperature,rng):
        values=coarse if stage=='coarse' else fine;mask_id=self.coarse_size if stage=='coarse' else self.fine_size
        total=len(values)
        for i in range(min(steps,total)):
            remaining=(values==mask_id).nonzero().flatten()
            if not len(remaining):break
            logits=self(features,coarse,fine,len(remaining)/total,stage)[0 if stage=='coarse' else 1]
            probs=(logits[remaining]/temperature).softmax(-1)
            sampled=torch.multinomial(probs,1,generator=rng).flatten()
            confidence=probs.gather(1,sampled[:,None]).flatten()
            target=math.ceil((i+1)*total/min(steps,total));count=min(len(remaining),max(1,target-(total-len(remaining))))
            chosen=confidence.topk(count).indices
            values[remaining[chosen]]=sampled[chosen]
        if (values==mask_id).any():raise RuntimeError('Sampling left MASK codes')
        return values
    @torch.no_grad()
    def sample_coarse(self,features,seed,steps=16,temperature=1.):
        if steps<1 or temperature<=0:raise ValueError('Positive steps/temperature required')
        rng=torch.Generator(device=features.device).manual_seed(seed);n=len(features)
        c=torch.full(((n+self.block_size-1)//self.block_size,),self.coarse_size,device=features.device,dtype=torch.long)
        f=torch.full((n,),self.fine_size,device=features.device,dtype=torch.long)
        return self._fill(features,c,f,'coarse',steps,temperature,rng)
    @torch.no_grad()
    def sample(self,features,seed,steps=16,temperature=1.):
        if steps<1 or temperature<=0:raise ValueError('Positive steps/temperature required')
        rng=torch.Generator(device=features.device).manual_seed(seed);n=len(features)
        c=torch.full(((n+self.block_size-1)//self.block_size,),self.coarse_size,device=features.device,dtype=torch.long)
        f=torch.full((n,),self.fine_size,device=features.device,dtype=torch.long)
        self._fill(features,c,f,'coarse',steps,temperature,rng)
        self._fill(features,c,f,'fine',steps,temperature,rng)
        return c,f
