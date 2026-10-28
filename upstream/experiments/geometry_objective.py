"""Hard-token decoder forward with a soft-probability surrogate gradient.

The straight-through estimator is biased; hard decoded structures must be
evaluated separately. Losses below are backbone proxies, not physical energies.
"""
import torch
from torch.nn import functional as F


def hard_token_embedding(logits,embedding):
    if logits.ndim!=2 or logits.shape[-1]!=embedding.shape[0]:raise ValueError('Logit/codebook shape mismatch')
    probability=logits.softmax(-1)
    ids=logits.argmax(-1)
    soft=probability@embedding
    hard=F.embedding(ids,embedding)
    return hard+(soft-soft.detach()),ids,probability


def backbone_penalties(backbone):
    if backbone.ndim!=3 or backbone.shape[1:]!=(3,3) or len(backbone)<2:
        raise ValueError('Expected N,CA,C coordinates [N,3,3]')
    cn=torch.linalg.vector_norm(backbone[:-1,2]-backbone[1:,0],dim=-1)
    cn_loss=(F.relu(1.1-cn).square()+F.relu(cn-1.7).square()).mean()
    ca=torch.pdist(backbone[:,1])
    clash=F.relu(3.-ca).square().mean()
    return cn_loss,clash


def mix_logit_gradients(geometry_gradient,probability,teacher,geometry_weight=1.,kl_weight=1.,permutation=None):
    if permutation is not None:geometry_gradient=geometry_gradient[:,permutation]
    return geometry_weight*geometry_gradient+kl_weight*(probability-teacher)/len(probability)
