"""Conditional fine-head scoring and exploratory, protein-level statistics."""
import numpy as np
from scipy.stats import rankdata


def sample_tokens(logits,temperature,samples,seed):
    import torch
    if temperature<=0 or samples<1 or not torch.isfinite(logits).all():raise ValueError('Invalid sampling inputs')
    generator=torch.Generator(device=logits.device).manual_seed(seed)
    with torch.no_grad():
        return torch.multinomial((logits/temperature).softmax(-1),samples,replacement=True,generator=generator).T


def fixed_log_score(logits,tokens):
    """Mean log probability under unchanged T=1 logits, not the sampling temperature."""
    import torch
    if tokens.shape[-1]!=logits.shape[0]:raise ValueError('Token/context length differs')
    rows=tokens.reshape(-1,logits.shape[0])
    with torch.no_grad():
        logp=logits.log_softmax(-1)
        return torch.stack([logp.gather(1,row[:,None]).mean() for row in rows]).reshape(tokens.shape[:-1])


def spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.shape!=y.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid correlation vectors')
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])


def protein_interval(values,seed=0):
    """Exploratory bootstrap over proteins; undefined observations remain explicit."""
    values=list(values);valid=np.asarray([v for v in values if v is not None],float)
    if len(valid)==0:return dict(total=len(values),defined=0,mean=None,bootstrap95=None)
    rng=np.random.default_rng(seed)
    means=valid[rng.integers(0,len(valid),size=(5000,len(valid)))].mean(1)
    return dict(total=len(values),defined=len(valid),mean=float(valid.mean()),
                bootstrap95=np.quantile(means,[.025,.975]).tolist())
