"""Explicit sequence-contiguous segments; no learned spatial grouping claim."""
import numpy as np


def segments(labels,max_length=None):
    if len(labels)==0:raise ValueError('Empty labels')
    if max_length is not None and max_length<1:raise ValueError('Invalid segment cap')
    result=[];start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[start]=='X' or labels[i]!=labels[start] or (max_length is not None and i-start>=max_length):result.append((start,i));start=i
    return result


def pooled_broadcast(values,parts):
    values=np.asarray(values);out=np.empty_like(values);cursor=0
    for start,end in parts:
        if start!=cursor or not start<end<=len(values):raise ValueError('Segments must partition residues')
        out[start:end]=values[start:end].mean(0);cursor=end
    if cursor!=len(values):raise ValueError('Incomplete partition')
    return out
