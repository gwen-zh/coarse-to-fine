"""Continuous coarse geometry objectives; the structural decoder stays frozen.

The alignment rotation is optimized on detached coordinates. Differentiating
squared aligned error with this rotation fixed gives its envelope gradient at
nondegenerate optima, avoiding unstable SVD derivatives. Distances alone cannot
exclude mirror images, so the shape objective also includes proper alignment.
"""
import hashlib
import torch
from latent_flow_head import flow_path,protein_mean_mse
from mixed_hierarchical_generator import pooled_sequence

ARMS=['control','raw_geometry','shape_geometry']


def shape_losses(prediction,reference,counts):
    if prediction.ndim!=2 or prediction.shape[1]!=3 or reference.shape!=prediction.shape:
        raise ValueError('Corresponding Mx3 center matrices required')
    weights=torch.as_tensor(counts,device=prediction.device,dtype=prediction.dtype)
    if weights.shape!=(len(prediction),) or (weights<=0).any():raise ValueError('Positive residue counts required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(reference).all():raise ValueError('Nonfinite block centers')
    reference=reference.detach();w=weights/weights.sum()
    x=prediction-(prediction*w[:,None]).sum(0);y=reference-(reference*w[:,None]).sum(0)
    with torch.no_grad():
        covariance=(x.detach().double()*w.double()[:,None]).T@y.double()
        u,_,vh=torch.linalg.svd(covariance)
        correction=torch.eye(3,device=x.device,dtype=torch.float64)
        correction[-1,-1]=torch.where(torch.linalg.det(u@vh)<0,-1.,1.)
        rotation=(u@correction@vh).to(prediction.dtype)
    aligned=(((x@rotation-y).square().sum(-1))*w).sum()/3./100.
    if len(prediction)<2:return aligned,prediction.sum()*0.
    # Clamp only the zero-distance derivative; self-pairs are excluded.
    def distances(a):return (a[:,None]-a[None]).square().sum(-1).clamp_min(1e-12).sqrt()
    mask=torch.triu(torch.ones(len(prediction),len(prediction),device=x.device,dtype=torch.bool),diagonal=1)
    pairs=(w[:,None]*w[None])[mask]
    distance=((distances(prediction)[mask]-distances(reference)[mask]).square()*pairs).sum()/pairs.sum()/100.
    return aligned,distance


def geometry_flow_loss(generator,decoder,features,target,arm,noise=None,t=None):
    if arm not in ARMS:raise ValueError('Unknown geometry objective')
    truth=generator.normalize(target)[None]
    noise=torch.randn_like(truth) if noise is None else noise
    t=torch.rand((1,),device=truth.device) if t is None else t
    state,velocity=flow_path(truth,noise,t);condition=pooled_sequence(features)[None]
    valid=torch.ones(truth.shape[:2],dtype=torch.bool,device=truth.device)
    prediction=generator.coarse_flow(state,t,condition,valid)
    flow=protein_mean_mse(prediction,velocity,valid)
    endpoint=state+(1-t[:,None,None])*prediction
    raw=flow*0.;aligned=flow*0.;distance=flow*0.;geometry=flow*0.
    coefficient=.1 if arm=='control' else 1.
    if arm!='control' or float(t[0])>=.5:
        centers=decoder.coarse_centers(generator.denormalize(endpoint[0]),len(features))
        with torch.no_grad():reference=decoder.coarse_centers(target,len(features))
        if arm=='control':
            raw=(centers-reference).square().mean()/100.;geometry=raw
        else:
            n=len(features);counts=torch.minimum(torch.full((len(centers),),4,device=centers.device),n-4*torch.arange(len(centers),device=centers.device)).to(centers.dtype)
            w=counts/counts.sum()
            if arm=='raw_geometry':
                raw=((centers-reference).square().sum(-1)*w).sum()/3./100.;geometry=raw
            else:
                aligned,distance=shape_losses(centers,reference,counts);geometry=.5*(aligned+distance)
    draw=hashlib.sha256(noise.detach().cpu().numpy().tobytes()+t.detach().cpu().numpy().tobytes()).hexdigest()
    return flow+coefficient*geometry,dict(flow=flow,geometry=geometry,weighted_geometry=coefficient*geometry,
        raw=raw,aligned=aligned,distance=distance,t=float(t[0]),draw_sha256=draw)
