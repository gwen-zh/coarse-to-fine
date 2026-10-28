"""Rank genuine bridge distances; these are not fine-head probabilities."""
import numpy as np


def ranked_candidates(distances, original, k=5):
    d=np.asarray(distances);ids=np.asarray(original)
    if d.ndim!=2 or ids.shape!=(len(d),) or ids.dtype.kind not in "iu":raise ValueError("Bad distance/token shapes")
    if not np.isfinite(d).all() or not 1<=k<=d.shape[1] or np.any(ids<0) or np.any(ids>=d.shape[1]):raise ValueError("Bad distances/IDs/k")
    gap=d[np.arange(len(d)),ids]-d.min(-1)
    if np.any(gap>1e-5*np.maximum(1,np.abs(d.min(-1)))):raise ValueError("Initial tokens do not replay nearest bridge IDs")
    rows=[]
    for row,token in zip(d,ids):
        order=np.argsort(row,kind="stable")
        rows.append([int(token)]+[int(x) for x in order if x!=token][:k-1])
    return np.asarray(rows,dtype=np.int64)
