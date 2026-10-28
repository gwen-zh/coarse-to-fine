"""Mask-source corrector rates with a bounded frozen-hazard local integrator.

For kappa=1-m and corrector (1+b)u_forward-b*u_backward:
mask->tokens total rate is (1+b)kappa_dot/(1-kappa),
token->mask rate is b*kappa_dot/kappa.
Integrated hazards over a denoising interval are A=(1+b)log(m_t/m_s),
B=b*log((1-m_s)/(1-m_t)). Freeze the posterior during this interval and
use the two-status matrix exponential. It is first-order consistent, not an
exact inhomogeneous multi-site CTMC simulation. At b=0 it reproduces the
native finite-step mask transition exactly.
"""
import torch


def status_probabilities(mask_now,mask_next,beta):
    if beta<0:raise ValueError("Nonnegative corrector beta required")
    mt=torch.as_tensor(mask_now,dtype=torch.float64);ms=torch.as_tensor(mask_next,dtype=torch.float64,device=mt.device)
    if not torch.isfinite(mt).all() or not torch.isfinite(ms).all() or torch.any(ms<=0) or torch.any(mt>=1) or torch.any(ms>mt):raise ValueError("Require0<mask_next<=mask_now<1")
    if beta==0:return 1-ms/mt,torch.zeros_like(mt)
    forward=(1+beta)*torch.log(mt/ms)
    backward=beta*(torch.log1p(-ms)-torch.log1p(-mt))
    total=forward+backward;move=-torch.expm1(-total);denominator=torch.where(total>0,total,torch.ones_like(total))
    return forward/denominator*move,backward/denominator*move


def transition_probabilities(log_p,x,mask_now,mask_next,beta,mask_id=32):
    if log_p.shape[:-1]!=x.shape or mask_id>=log_p.shape[-1]:raise ValueError("Shape/vocabulary mismatch")
    unmask,remask=status_probabilities(mask_now,mask_next,beta)
    while unmask.ndim<x.ndim:unmask=unmask.unsqueeze(-1);remask=remask.unsqueeze(-1)
    unmask=unmask.to(log_p.dtype);remask=remask.to(log_p.dtype)
    masked=x==mask_id
    posterior=log_p.exp();posterior[...,mask_id:]=0
    denominator=posterior.sum(-1,keepdim=True)
    # Visible special tokens are restored by the caller and do not use posterior.
    posterior=posterior/denominator.clamp_min(torch.finfo(log_p.dtype).tiny)
    masked_weights=posterior*unmask.unsqueeze(-1);masked_weights[...,mask_id]=1-unmask
    visible=torch.nn.functional.one_hot(x,log_p.shape[-1]).to(log_p.dtype)*(1-remask).unsqueeze(-1)
    visible[...,mask_id]+=remask
    return torch.where(masked.unsqueeze(-1),masked_weights,visible)


def native_raw_weights(log_p,mask_now,mask_next,mask_id=32):
    # Preserve native FP32 operation order and full vocabulary width for RNG replay.
    mt=mask_now[:,None,None];ms=mask_next[:,None,None]
    weights=log_p.exp()*(mt-ms);weights[:,:,mask_id]=ms[:,:,0]
    return weights


def competing_exponential_sample(weights):
    denominator=1e-10-(torch.rand_like(weights)+1e-10).log()
    return (weights/denominator).argmax(-1)
