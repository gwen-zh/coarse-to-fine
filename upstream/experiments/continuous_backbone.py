"""Fine-stage continuous-state conditioning of the pretrained generator trunk.

The original network source is untouched. Adapters are active only inside this
wrapper call; outside calls (including coarse generation) use base projections.
The first implementation explicitly supports one complete, unpadded chain.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ScopedLoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base.requires_grad_(False);self.active=False
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        value=self.base(x)
        return value+F.linear(F.linear(x,self.a),self.b) if self.active else value


class ContinuousStateInput(nn.Module):
    def __init__(self,width,latent_dim=128,initial_scale=.1):
        super().__init__();self.latent_dim=latent_dim;self.initial_scale=initial_scale
        self.state=nn.Linear(latent_dim,width,bias=False)
        self.time=nn.Sequential(nn.Linear(64,256),nn.SiLU(),nn.Linear(256,width))
        nn.init.normal_(self.state.weight,std=.01)
    def forward(self,state,t):
        if state.ndim!=3 or state.shape[-1]!=self.latent_dim or t.shape!=(state.shape[0],):raise ValueError('State/time dimensions')
        if not torch.isfinite(state).all() or not torch.isfinite(t).all() or torch.any((t<0)|(t>1)):raise ValueError('Invalid state/time')
        f=torch.exp(torch.arange(32,device=t.device,dtype=t.dtype)*(-math.log(10000.)/31))
        phase=1000*t[:,None]*f[None]
        delta=self.initial_scale*(self.state(state)+self.time(torch.cat([phase.sin(),phase.cos()],-1))[:,None])
        # Never inject continuous state into BOS/EOS.
        return F.pad(delta,(0,0,1,1))


class ContinuousBackbone(nn.Module):
    def __init__(self,net,width=1536,latent_dim=128,last_blocks=4,rank=4):
        super().__init__();self.net=net.requires_grad_(False);self.latent_dim=latent_dim
        device=next(net.parameters()).device
        self.input_adapter=ContinuousStateInput(width,latent_dim).to(device)
        self.velocity_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,latent_dim)).to(device)
        nn.init.zeros_(self.velocity_head[-1].weight);nn.init.zeros_(self.velocity_head[-1].bias)
        self.scoped_loras=[]
        if not 0<=last_blocks<=len(net.transformer.blocks):raise ValueError('Invalid LoRA depth')
        for block in list(net.transformer.blocks)[-last_blocks:] if last_blocks else []:
            for container,key in [(block.attn.layernorm_qkv,1),(block.attn,'out_proj')]:
                base=container[key] if isinstance(key,int) else getattr(container,key)
                if not isinstance(base,nn.Linear):raise ValueError('Expected unwrapped pretrained linear')
                layer=ScopedLoRALinear(base,rank)
                if isinstance(key,int):container[key]=layer
                else:setattr(container,key,layer)
                self.scoped_loras.append(layer)
        self._inside=False

    def forward(self,sequence,coarse,state,t,enabled=True):
        # Do not silently treat variable-length/padded batches as full chains.
        if sequence.ndim!=2 or sequence.shape[0]!=1 or coarse.shape!=sequence.shape:raise ValueError('Only one unpadded chain supported at this stage')
        n=sequence.shape[1]-2
        if state.shape!=(1,n,self.latent_dim) or t.shape!=(1,):raise ValueError('State length/dimension mismatch')
        if sequence.dtype!=torch.long or coarse.dtype!=torch.long:raise ValueError('Token dtype')
        if int(coarse[0,0])!=34 or int(coarse[0,-1])!=33 or torch.any((coarse[:,1:-1]<0)|(coarse[:,1:-1]>=32)):raise ValueError('Expected published sampled-coarse IDs with BOS/EOS; no fine feedback')
        if self._inside:raise RuntimeError('Wrapper is not reentrant')
        capture={};handles=[];previous=[x.active for x in self.scoped_loras];self._inside=True
        try:
            for layer in self.scoped_loras:layer.active=bool(enabled)
            delta=self.input_adapter(state,t) if enabled else None
            def inject(module,args,output):
                if args[-1]!=2 or not torch.is_tensor(output) or output.shape[:2]!=sequence.shape:raise ValueError('Unexpected fine encoder interface')
                if delta is not None:
                    if delta.shape!=output.shape:raise ValueError('Injection shape mismatch')
                    return output+delta.to(output.dtype)
                return output
            def capture_head(module,args,kwargs):
                if kwargs.get('scale')!=2 or 'z' not in kwargs:raise ValueError('Unexpected output-head interface')
                capture['hidden']=kwargs['z'][:,1:-1]
            handles.append(self.net.encoder.register_forward_hook(inject))
            handles.append(self.net.output_heads.register_forward_pre_hook(capture_head,with_kwargs=True))
            out=self.net(var_structure_tokens=coarse,sequence_tokens=sequence,auxiliary_embeddings=None,lengths=torch.tensor([n],device=sequence.device),scale=2)
            hidden=capture['hidden']
            return dict(velocity=self.velocity_head(hidden),native_logits=out.structure_logits[:,1:-1,:4096],hidden=hidden)
        finally:
            for handle in handles:handle.remove()
            for layer,active in zip(self.scoped_loras,previous):layer.active=active
            self._inside=False
