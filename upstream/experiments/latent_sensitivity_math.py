"""Noise calibration and exact Euclidean Voronoi-face geometry."""
import numpy as np


def covariance_root(covariance):
    covariance=np.asarray(covariance,float)
    if covariance.ndim!=2 or covariance.shape[0]!=covariance.shape[1] or not np.isfinite(covariance).all():raise ValueError("Invalid covariance")
    values,vectors=np.linalg.eigh((covariance+covariance.T)/2)
    if values.min() < -1e-8*max(1.,np.abs(values).max()):raise ValueError("Covariance is not PSD")
    return vectors*np.sqrt(np.maximum(values,0.))[None,:]


def unit_rms_noise(standard_normal,root=None):
    x=np.asarray(standard_normal,float)
    value=x.copy() if root is None else x@np.asarray(root).T
    rms=float(np.sqrt(np.mean(value*value)))
    if not np.isfinite(rms) or rms<=0:raise ValueError("Degenerate noise")
    return value/rms


def nearest_cell_faces(latent,codes,assigned):
    z,c=np.asarray(latent,float),np.asarray(codes,float);ids=np.asarray(assigned)
    if z.ndim!=2 or c.ndim!=2 or z.shape[1]!=c.shape[1] or ids.shape!=(len(z),):raise ValueError("Bad cell geometry shapes")
    if not np.isfinite(z).all() or not np.isfinite(c).all():raise ValueError("Nonfinite cell geometry")
    if ids.dtype.kind not in "iu" or np.any(ids<0) or np.any(ids>=len(c)):raise ValueError("Invalid assignments")
    own=c[ids];distance=(z*z).sum(-1,keepdims=True)-2*z@c.T+(c*c).sum(-1)[None,:]
    own_distance=np.sum((z-own)**2,-1);gap=distance-own_distance[:,None]
    if np.any(gap.min(-1)<-1e-5*np.maximum(1.,own_distance)):raise ValueError("Assignments are not nearest codes")
    separation=np.sqrt(np.maximum(0.,(own*own).sum(-1,keepdims=True)-2*own@c.T+(c*c).sum(-1)[None,:]))
    radius=np.full_like(distance,np.inf);valid=separation>1e-10
    np.divide(np.maximum(gap,0.),2*separation,out=radius,where=valid)
    radius[np.arange(len(z)),ids]=np.inf
    neighbor=radius.argmin(-1);margin=radius[np.arange(len(z)),neighbor]
    if not np.isfinite(margin).all():raise ValueError("No distinct neighboring cell")
    direction=c[neighbor]-own;direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    return margin,neighbor,direction


def aligned_ca_distances(mobile,reference):
    x,y=np.asarray(mobile,float),np.asarray(reference,float)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Invalid CA arrays")
    x=x-x.mean(0);y=y-y.mean(0);u,_,vt=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt)
    return np.linalg.norm(x@(u@fix@vt)-y,axis=-1)
