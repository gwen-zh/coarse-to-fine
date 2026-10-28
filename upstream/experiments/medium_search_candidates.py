"""Fine-center neighborhoods induced by medium centers; no nested-cell assumption."""
import numpy as np


def squared_distances(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Finite compatible code matrices required")
    return np.maximum(0.,(x*x).sum(-1,keepdims=True)-2*x@y.T+(y*y).sum(-1)[None,:])


def candidate_tables(fine,medium,alternatives=4):
    if not 1<=alternatives<len(fine):raise ValueError("Insufficient fine codes")
    distance=squared_distances(fine,fine);np.fill_diagonal(distance,np.inf)
    group=squared_distances(fine,medium).argmin(-1);sizes=np.bincount(group,minlength=len(medium))
    nearest=np.empty((len(fine),alternatives+1),np.int64);cluster=nearest.copy();fallback=np.zeros_like(cluster,dtype=bool)
    for token in range(len(fine)):
        order=np.argsort(distance[token],kind="stable");order=order[order!=token]
        nearest[token]=[token]+order[:alternatives].tolist()
        members=[int(i) for i in order if group[i]==group[token]]
        chosen=members[:alternatives];flags=[False]*len(chosen)
        if len(chosen)<alternatives:
            for i in order:
                if int(i) in chosen:continue
                chosen.append(int(i));flags.append(True)
                if len(chosen)==alternatives:break
        cluster[token]=[token]+chosen;fallback[token]=[False]+flags
    return dict(nearest=nearest,cluster=cluster,fallback=fallback,group=group,sizes=sizes)
