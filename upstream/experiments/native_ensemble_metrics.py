"""Descriptive native-MD ensemble metrics; histogram bins fitted on TRAIN only.

JS here means divergence in nats (0 to log(2)), not its square root. These
finite-sample descriptors do not establish equilibrium population calibration.
"""
import numpy as np
from structural_metrics import aligned_rmsd


def validate_ca(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=3 or values.shape[-1]!=3 or values.shape[1]<5 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty frame x residue x xyz ensemble required')
    return values


def distance_features(ca):
    ca=validate_ca(ca)
    i,j=np.triu_indices(ca.shape[1],4)
    distance=np.linalg.norm(ca[:,i]-ca[:,j],axis=-1)
    rg=np.sqrt(((ca-ca.mean(1,keepdims=True))**2).sum(-1).mean(-1))
    return distance,rg[:,None]


def train_histogram_edges(features,bins=10):
    features=np.asarray(features,float)
    if features.ndim!=2 or not len(features) or not features.shape[1] or bins<2 or not np.isfinite(features).all():
        raise ValueError('Finite TRAIN-only features required')
    lo=features.min(0);hi=features.max(0);width=np.maximum(hi-lo,1e-3)
    lo=(lo+hi)/2-width/2;hi=lo+width
    core=lo[:,None]+(hi-lo)[:,None]*np.linspace(0,1,bins+1)[None]
    return np.concatenate([np.full((len(lo),1),-np.inf),core,np.full((len(lo),1),np.inf)],axis=1)


def histogram_probabilities(features,edges):
    features=np.asarray(features,float);edges=np.asarray(edges,float)
    if features.ndim!=2 or edges.ndim!=2 or features.shape[1]!=len(edges) or not len(features) or not np.isfinite(features).all():
        raise ValueError('Histogram feature/edge shape mismatch')
    return np.array([np.histogram(features[:,i],bins=edge)[0]/len(features) for i,edge in enumerate(edges)])


def js_divergence(p,q):
    p=np.asarray(p,float);q=np.asarray(q,float)
    if p.shape!=q.shape or np.any(p<0) or np.any(q<0) or not np.isfinite(p).all() or not np.isfinite(q).all() or not np.allclose(p.sum(-1),1) or not np.allclose(q.sum(-1),1):
        raise ValueError('Normalized matching probability distributions required')
    mid=(p+q)/2
    def kl(a):
        mask=a>0;terms=np.zeros_like(a);terms[mask]=a[mask]*np.log(a[mask]/mid[mask]);return terms.sum(-1)
    return (kl(p)+kl(q))/2


def mean_pairwise_rmsd(ca):
    ca=validate_ca(ca)
    return float(np.mean([aligned_rmsd(ca[i],ca[j]) for i,j in zip(*np.triu_indices(len(ca),1))])) if len(ca)>1 else 0.


def rmsf(ca,alignment_reference):
    ca=validate_ca(ca);reference=np.asarray(alignment_reference,float);aligned=[]
    if reference.shape!=ca.shape[1:]:raise ValueError('One fixed TRAIN alignment reference required')
    centered_reference=reference-reference.mean(0)
    for frame in ca:
        x=frame-frame.mean(0);u,_,vt=np.linalg.svd(x.T@centered_reference);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
        aligned.append(x@(u@correction@vt))
    aligned=np.stack(aligned)
    return np.sqrt(((aligned-aligned.mean(0))**2).sum(-1).mean(0))


def compare_ensembles(generated,reference,distance_edges,rg_edges,alignment_reference):
    generated=validate_ca(generated);reference=validate_ca(reference)
    if generated.shape[1:]!=reference.shape[1:]:raise ValueError('Paired sequence residue mapping required')
    gd,gr=distance_features(generated);rd,rr=distance_features(reference)
    matrix=np.array([[aligned_rmsd(sample,ref) for ref in reference] for sample in generated])
    generated_rmsf=rmsf(generated,alignment_reference);reference_rmsf=rmsf(reference,alignment_reference)
    return dict(generated_samples=len(generated),reference_frames=len(reference),mean_sample_nearest_reference_ca_rmsd_A=float(matrix.min(1).mean()),mean_reference_nearest_sample_ca_rmsd_A=float(matrix.min(0).mean()),reference_coverage_at_A={str(x):float((matrix.min(0)<=x).mean()) for x in [.5,1.,2.]},mean_pairwise_ca_rmsd_A=mean_pairwise_rmsd(generated),reference_mean_pairwise_ca_rmsd_A=mean_pairwise_rmsd(reference),js_pairwise_distance_divergence_nats=float(js_divergence(histogram_probabilities(gd,distance_edges),histogram_probabilities(rd,distance_edges)).mean()),js_rg_divergence_nats=float(js_divergence(histogram_probabilities(gr,rg_edges),histogram_probabilities(rr,rg_edges)).mean()),rmsf_mae_A=float(np.abs(generated_rmsf-reference_rmsf).mean()),mean_rg_A=float(gr.mean()),reference_mean_rg_A=float(rr.mean()))
