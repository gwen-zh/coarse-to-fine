"""Norm and direction diagnostics; zero vectors have no defined cosine."""
import torch


def compare_gradients(ce,kl,geometry):
    if ce.shape!=kl.shape or ce.shape!=geometry.shape:
        raise ValueError('Gradient shapes differ')
    if not all(torch.isfinite(g).all() for g in [ce,kl,geometry]):
        raise ValueError('Nonfinite gradient')
    gradients=dict(ce=ce.flatten().double(),kl=kl.flatten().double(),geometry=geometry.flatten().double())
    gradients['base']=gradients['ce']+gradients['kl']
    norms={name:float(g.norm()) for name,g in gradients.items()}
    result={'norms':norms}
    for name in ['ce','kl','base']:
        denominator=norms['geometry']*norms[name]
        result['geometry_cosine_'+name]=(max(-1.,min(1.,float(gradients['geometry']@gradients[name])/denominator))
            if denominator>0 else None)
    result['geometry_over_base']=norms['geometry']/norms['base'] if norms['base']>0 else None
    return result
