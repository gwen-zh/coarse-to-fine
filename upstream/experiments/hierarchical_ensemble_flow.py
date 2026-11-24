"""AA-only hierarchical continuous flows sharing one Transformer backbone.

p(c,f|AA)=p(c|AA)*p(f|c,AA). Sample coarse first, then continuous fine
conditional on that SAME sampled coarse. Snap fine only at the frozen decoder
interface. Noise is a base distribution; encoded latents need not be Gaussian.
"""
import hashlib,math
import torch
from torch import nn
from joint_sequence_predictor import AA_ALPHABET
from two_level_autoencoder import pack,positions


def component_seed(seed,component):
    return int(hashlib.sha256(f'hierarchical-ensemble-flow-v1:{int(seed)}:{component}'.encode()).hexdigest()[:8],16)


def fine_attention_mask(m,n,device):
    mask=torch.zeros((m+n,m+n),dtype=torch.bool,device=device)
    mask[:m,m:]=True
    return mask


def snap_fine(fine,codes):
    if fine.ndim!=2 or codes.ndim!=2 or fine.shape[1]!=codes.shape[1] or not len(fine) or not len(codes) or not torch.isfinite(fine).all() or not torch.isfinite(codes).all():raise ValueError('Finite matching fine vectors and frozen code matrix required')
    distance=fine.square().sum(-1,keepdim=True)-2*fine@codes.T+codes.square().sum(-1)
    ids=distance.argmin(-1)
    return ids,(fine-codes[ids]).square().mean(-1).sqrt()


class HierarchicalEnsembleFlow(nn.Module):
    def __init__(self,coarse_mean,coarse_std,fine_mean,fine_std,width=256,layers=6,coarse_dim=16,fine_dim=8,block_size=4):
        super().__init__()
        if width<8 or width%8 or layers<1 or coarse_dim<1 or fine_dim<1 or block_size<1:raise ValueError('Invalid model configuration')
        for name,value,dim,positive in [('coarse_mean',coarse_mean,coarse_dim,False),('coarse_std',coarse_std,coarse_dim,True),('fine_mean',fine_mean,fine_dim,False),('fine_std',fine_std,fine_dim,True)]:
            value=torch.as_tensor(value,dtype=torch.float32)
            if value.shape!=(dim,) or not torch.isfinite(value).all() or (positive and (value<=0).any()):raise ValueError('Finite TRAIN-only normalization required')
            self.register_buffer(name,value.clone())
        self.coarse_dim=coarse_dim;self.fine_dim=fine_dim;self.block_size=block_size
        self.config=dict(width=width,layers=layers,coarse_dim=coarse_dim,fine_dim=fine_dim,block_size=block_size)
        self.aa_embedding=nn.Embedding(len(AA_ALPHABET),width)
        # Flatten the ordered block: do not average uncontextualized AA IDs.
        self.block_condition=nn.Sequential(nn.Linear(block_size*width+block_size,width),nn.SiLU(),nn.LayerNorm(width))
        self.coarse_input=nn.Linear(coarse_dim,width);self.fine_input=nn.Linear(fine_dim,width)
        self.coarse_to_fine=nn.Linear(coarse_dim,width)
        self.kind=nn.Embedding(2,width);self.mode=nn.Embedding(2,width)
        self.time=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width))
        layer=nn.TransformerEncoderLayer(width,8,4*width,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.coarse_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,coarse_dim))
        self.fine_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,fine_dim))
        for head in [self.coarse_head,self.fine_head]:nn.init.normal_(head[-1].weight,std=.01);nn.init.zeros_(head[-1].bias)
    def normalize_coarse(self,c):return (c-self.coarse_mean)/self.coarse_std
    def denormalize_coarse(self,c):return c*self.coarse_std+self.coarse_mean
    def normalize_fine(self,f):return (f-self.fine_mean)/self.fine_std
    def denormalize_fine(self,f):return f*self.fine_std+self.fine_mean
    def aa_conditions(self,aa):
        if aa.ndim!=1 or aa.dtype!=torch.long or not len(aa) or aa.min()<0 or aa.max()>=len(AA_ALPHABET):raise ValueError('One unpadded AA-ID sequence required')
        h=self.aa_embedding(aa);h=h+positions(len(aa),h.device,h.dtype,h.shape[-1])
        packed,valid=pack(h,self.block_size)
        block=self.block_condition(torch.cat([packed.flatten(1),valid.to(h.dtype)],-1))
        return h,block
    def time_state(self,t,reference):
        t=torch.as_tensor(t,device=reference.device,dtype=reference.dtype)
        if t.numel()!=1 or not torch.isfinite(t).all() or t<0 or t>1:raise ValueError('Scalar flow time in [0,1] required')
        frequency=torch.exp(torch.arange(32,device=reference.device,dtype=reference.dtype)*(-math.log(10000.)/31))
        phase=t.reshape(())*frequency*1000.
        return self.time(torch.cat([phase.sin(),phase.cos()]))
    def _coarse_state(self,c,n):
        if c.shape!=((n+self.block_size-1)//self.block_size,self.coarse_dim) or not c.is_floating_point() or not torch.isfinite(c).all():raise ValueError('Finite matching coarse-state shape required')
    def coarse_velocity(self,aa,state,t):
        residue,block=self.aa_conditions(aa);self._coarse_state(state,len(aa))
        h=block+self.coarse_input(state)+self.kind.weight[0]+self.mode.weight[0]+self.time_state(t,block)
        return self.coarse_head(self.transformer(h[None])[0])
    def fine_hidden(self,aa,state,t,coarse_condition):
        residue,block=self.aa_conditions(aa);self._coarse_state(coarse_condition,len(aa))
        if state.shape!=(len(aa),self.fine_dim) or not state.is_floating_point() or not torch.isfinite(state).all():raise ValueError('Finite matching fine-state shape required')
        time=self.time_state(t,residue);mode=self.mode.weight[1]
        c=block+self.coarse_input(coarse_condition)+self.kind.weight[0]+mode+time
        local=coarse_condition.repeat_interleave(self.block_size,dim=0)[:len(aa)]
        f=residue+self.fine_input(state)+self.coarse_to_fine(local)+self.kind.weight[1]+mode+time
        mask=fine_attention_mask(len(c),len(f),state.device)
        return self.transformer(torch.cat([c,f])[None],mask=mask)[0]
    def fine_velocity(self,aa,state,t,coarse_condition):
        m=(len(aa)+self.block_size-1)//self.block_size
        return self.fine_head(self.fine_hidden(aa,state,t,coarse_condition)[m:])
    def flow_loss(self,aa,target_coarse,target_fine,coarse_jitter_std=.05,generator=None):
        if coarse_jitter_std<0 or not math.isfinite(coarse_jitter_std):raise ValueError('Nonnegative finite paired-conditioning jitter required')
        self._coarse_state(target_coarse,len(aa))
        if target_fine.shape!=(len(aa),self.fine_dim) or not torch.isfinite(target_fine).all():raise ValueError('Expected continuous prequantization fine targets,not IDs')
        c=self.normalize_coarse(target_coarse.detach());f=self.normalize_fine(target_fine.detach())
        nc=torch.randn(c.shape,device=c.device,dtype=c.dtype,generator=generator);nf=torch.randn(f.shape,device=f.device,dtype=f.dtype,generator=generator)
        tc=torch.rand((),device=c.device,dtype=c.dtype,generator=generator);tf=torch.rand((),device=f.device,dtype=f.dtype,generator=generator)
        jitter=torch.randn(c.shape,device=c.device,dtype=c.dtype,generator=generator)
        condition=c+coarse_jitter_std*jitter
        vc=self.coarse_velocity(aa,(1-tc)*nc+tc*c,tc)
        vf=self.fine_velocity(aa,(1-tf)*nf+tf*f,tf,condition)
        lc=(vc-(c-nc)).square().mean();lf=(vf-(f-nf)).square().mean()
        digest=hashlib.sha256(nc.cpu().numpy().tobytes()+nf.cpu().numpy().tobytes()+tc.cpu().numpy().tobytes()+tf.cpu().numpy().tobytes()+jitter.cpu().numpy().tobytes()).hexdigest()
        return lc+lf,dict(coarse_flow=lc.detach(),fine_flow=lf.detach(),coarse_jitter_std=float(coarse_jitter_std),draw_sha256=digest)
    @torch.no_grad()
    def sample_coarse(self,aa,seed,steps=16):
        if steps<1:raise ValueError('Positive coarse solver steps required')
        m=(len(aa)+self.block_size-1)//self.block_size
        rng=torch.Generator(device=aa.device).manual_seed(int(seed)%(2**32))
        z=torch.randn((m,self.coarse_dim),device=aa.device,generator=rng)
        for i in range(steps):z=z+self.coarse_velocity(aa,z,i/steps)/steps
        return self.denormalize_coarse(z)
    @torch.no_grad()
    def sample_fine(self,aa,coarse,seed,steps=16):
        if steps<1:raise ValueError('Positive fine solver steps required')
        self._coarse_state(coarse,len(aa));condition=self.normalize_coarse(coarse)
        rng=torch.Generator(device=aa.device).manual_seed(int(seed)%(2**32))
        z=torch.randn((len(aa),self.fine_dim),device=aa.device,generator=rng)
        for i in range(steps):z=z+self.fine_velocity(aa,z,i/steps,condition)/steps
        return self.denormalize_fine(z)
    @torch.no_grad()
    def sample_latents(self,aa,seed,coarse_steps=16,fine_steps=16):
        c=self.sample_coarse(aa,component_seed(seed,'coarse'),coarse_steps)
        f=self.sample_fine(aa,c,component_seed(seed,'fine'),fine_steps)
        return c,f
    @torch.no_grad()
    def sample_structure(self,aa,decoder,seed,coarse_steps=16,fine_steps=16):
        book=decoder.model.fine_book
        if decoder.training or book.training or decoder.model.coarse_dim!=self.coarse_dim or book.dim!=self.fine_dim or decoder.model.block_size!=self.block_size:raise ValueError('Matching frozen hybrid decoder/codebook required')
        c,f=self.sample_latents(aa,seed,coarse_steps,fine_steps);ids,error=snap_fine(f,book.codes)
        bb=decoder.decode(c,ids)
        return dict(coarse=c,fine_continuous=f,fine_ids=ids,fine_snap_rmse=error,backbone=bb)
