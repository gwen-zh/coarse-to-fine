"""Controlled fine-vocabulary experiment on frozen, previously learned latents.

All K values use the SAME encoder/decoder and train-only latent pool. This tests
capacity conditional on a representation learned with K=1024, not independently
optimized tokenizers at every K. Refitted1024 is the main comparator; the trained
original1024 is a separate historical reference.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from mixed_hierarchical_generator import ContinuousFineGenerator
from regularized_mixed_generator import set_dropout


def nearest(z,codes,chunk=2048):
    if z.ndim!=2 or codes.ndim!=2 or z.shape[1]!=codes.shape[1] or not len(codes):
        raise ValueError('Matching nonempty vector matrices required')
    if not torch.isfinite(z).all() or not torch.isfinite(codes).all():
        raise ValueError('Nonfinite latent/code')
    if not len(z):return torch.empty(0,dtype=torch.long,device=z.device)
    return torch.cat([(v.square().sum(-1,keepdim=True)-2*v@codes.T+codes.square().sum(-1)).argmin(-1) for v in z.split(chunk)])


@torch.no_grad()
def refit_codes(pool,k,seed,iterations=10,chunk=2048):
    if len(pool)<k or iterations<1 or not torch.isfinite(pool).all():
        raise ValueError('Finite training pool at least K; positive fit iterations required')
    rng=torch.Generator(device=pool.device).manual_seed(seed)
    # All K receive nested initial centers from the same training-pool ordering.
    centers=pool[torch.randperm(len(pool),device=pool.device,generator=rng)[:k]].clone()
    trace=[]
    for iteration in range(iterations):
        sums=torch.zeros_like(centers);counts=torch.zeros(k,device=pool.device,dtype=torch.long)
        for values in pool.split(chunk):
            ids=nearest(values,centers,chunk)
            sums.index_add_(0,ids,values);counts+=torch.bincount(ids,minlength=k)
        alive=counts>0
        centers[alive]=sums[alive]/counts[alive,None]
        if (~alive).any():
            centers[~alive]=pool[torch.randint(len(pool),(int((~alive).sum()),),generator=rng,device=pool.device)]
        trace.append(dict(iteration=iteration+1,nonempty=int(alive.sum()),restarted=int((~alive).sum())))
    return centers,trace


def counts_by_slot(ids,k):
    result=torch.zeros(k,4,dtype=torch.long)
    for sequence in ids:
        if sequence.ndim!=1 or (sequence<0).any() or (sequence>=k).any():raise ValueError('Fine IDs outside vocabulary')
        slots=torch.arange(len(sequence))%4
        result+=torch.bincount(sequence.cpu()*4+slots,minlength=k*4).reshape(k,4)
    return result


class CapacityFine(nn.Module):
    def __init__(self,mean,std,k,condition_dim=1536,coarse_dim=16,width=128,layers=2,dropout=.1):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clone())
        if self.mean.shape!=(coarse_dim,) or self.std.shape!=(coarse_dim,) or (self.std<=0).any():raise ValueError('Train-only normalization required')
        self.fine=ContinuousFineGenerator(condition_dim,coarse_dim,width,layers,k)
        set_dropout(self.fine,dropout)
        self.config=dict(k=k,condition_dim=condition_dim,coarse_dim=coarse_dim,width=width,layers=layers,dropout=dropout)
    def normalize(self,c):return (c-self.mean)/self.std
    @torch.no_grad()
    def sample(self,h,c,seed,steps=16):return self.fine.sample(h,self.normalize(c),seed,steps,1.)


def paired_model(mean,std,k,seed,maximum_k=4096,**config):
    if k>maximum_k:raise ValueError('K exceeds common initialization vocabulary')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base=CapacityFine(mean,std,maximum_k,**config)
        result=CapacityFine(mean,std,k,**config)
        source=base.state_dict();target={}
        for name,value in result.state_dict().items():
            old=source[name]
            if name=='fine.fine_embedding.weight':target[name]=torch.cat([old[:k],old[maximum_k:maximum_k+1]],0).clone()
            elif name in ['fine.head.1.weight','fine.head.1.bias']:target[name]=old[:k].clone()
            else:
                if old.shape!=value.shape:raise RuntimeError('Unpaired common parameter '+name)
                target[name]=old.clone()
        result.load_state_dict(target)
    return result


def training_loss(model,h,c,target,jitter=.05,smoothing=.05,rate=None):
    condition=model.normalize(c)
    if jitter:condition=model.normalize((condition+jitter*torch.randn_like(condition))*model.std+model.mean)
    ratio=float(torch.rand(())*.9+.1) if rate is None else float(rate)
    if not 0<ratio<=1:raise ValueError('Invalid mask ratio')
    mask=torch.rand(len(target))<ratio
    if not mask.any():mask[torch.randint(len(target),(1,))]=True
    partial=torch.where(mask,model.fine.fine_size,target)
    logits=model.fine(h,condition,partial,ratio)
    loss=F.cross_entropy(logits[mask],target[mask],label_smoothing=smoothing)
    nll=F.cross_entropy(logits[mask],target[mask])
    return loss,dict(nll=float(nll.detach()),accuracy=float((logits[mask].argmax(-1)==target[mask]).float().mean()),
                     mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest())


def decode_codes(decoder,c,ids,codes):
    if ids.ndim!=1 or ids.dtype!=torch.long or (ids<0).any() or (ids>=len(codes)).any():raise ValueError('Invalid fine IDs')
    return decoder.model.decode(c,codes[ids],len(ids))['backbone']


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
