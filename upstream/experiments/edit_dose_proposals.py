"""Paired, nested independent proposals with a fixed number of token edits."""
import numpy as np
from bounded_multiedit_proposals import maximum_edits,apply_multiedit


def coupled_fixed_edits(topk,original,count,seed,doses=(1,2,4,8),edit_cap=.1):
    pool=np.asarray(topk);initial=np.asarray(original);doses=tuple(doses)
    if pool.ndim!=2 or initial.shape!=(len(pool),) or count<0:raise ValueError('Invalid dimensions/count')
    if not doses or any(not isinstance(k,(int,np.integer)) or k<1 for k in doses) or len(set(doses))!=len(doses):raise ValueError('Invalid doses')
    largest=max(doses)
    if largest>maximum_edits(len(initial),edit_cap):raise ValueError('Dose exceeds acceptance cap; do not silently clip or change cohort')
    alternatives=[]
    for row,token in zip(pool,initial):
        if len(set(row.tolist()))!=len(row) or token not in row:raise ValueError('Pool must be unique and contain initial token')
        alternative=row[row!=token]
        if len(alternative)==0:raise ValueError('No alternatives')
        alternatives.append(alternative)
    rng=np.random.default_rng(seed);streams={k:[] for k in doses}
    for _ in range(count):
        sites=rng.choice(len(initial),size=largest,replace=False)
        tokens=np.asarray([rng.choice(alternatives[i]) for i in sites],dtype=initial.dtype)
        for k in doses:streams[k].append((sites[:k].copy(),tokens[:k].copy()))
    return streams
