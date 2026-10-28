"""Matched local proposal stream, independent of greedy acceptance history."""
import numpy as np


def local_proposals(topk,original,count,seed):
    table=np.asarray(topk);initial=np.asarray(original)
    if table.ndim!=2 or initial.shape!=(len(table),) or count<0:raise ValueError("Invalid proposal shapes/count")
    choices=[]
    for row,token in zip(table,initial):
        alternatives=[int(x) for x in row if x!=token]
        if not alternatives:raise ValueError("No alternative tokens")
        choices.append(alternatives)
    rng=np.random.default_rng(seed);sites=rng.integers(0,len(initial),size=count)
    return [(int(i),choices[i][int(rng.integers(len(choices[i])))]) for i in sites]


def apply_proposal(base,proposal):
    result=np.asarray(base).copy();site,token=proposal;result[site]=token
    return result
