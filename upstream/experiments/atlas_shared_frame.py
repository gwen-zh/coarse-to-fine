"""Candidate single-sequence shared training frame; no parameter changes.

Align structural target labels to a fixed R1 training reference. Generation
still needs only AA/noise; the reference is used for label preparation only.
This is a new target convention, not a silently changed original tokenizer.
"""
import torch
from hierarchical_latent_contract import pool_ca, remove_block_translation
from two_level_autoencoder import pack


def align_backbone_to_training_reference(backbone,reference):
    if backbone.shape!=reference.shape or backbone.ndim!=3 or backbone.shape[1:]!=(3,3) or not torch.isfinite(backbone).all() or not torch.isfinite(reference).all():
        raise ValueError('Complete finite same-sequence N/CA/C structural labels required')
    center=backbone[:,1].mean(0);reference_center=reference[:,1].mean(0)
    x=backbone[:,1]-center;y=reference[:,1]-reference_center
    u,_,vh=torch.linalg.svd(x.T@y);correction=torch.eye(3,device=x.device,dtype=x.dtype);correction[-1,-1]=torch.linalg.det(u@vh)
    rotation=u@correction@vh
    return (backbone-center)@rotation+reference_center


def encode_in_supplied_frame(model,target):
    if target.ndim!=3 or target.shape[1:]!=(3,3) or not torch.isfinite(target).all():raise ValueError('Finite centered structural labels required')
    # Exact existing encoder calculations, with an explicit frame supplied by
    # label preparation instead of reorienting each MD frame at residue one.
    centers=pool_ca(target,model.block_size);residual=remove_block_translation(target,model.block_size)
    c=torch.tanh(model.coarse_encoder(centers/10.))
    packed,valid=pack((residual/4.).flatten(1),model.block_size)
    f=torch.tanh(model.fine_encoder(torch.cat([packed.flatten(1),valid.to(target.dtype)],-1))).reshape(-1,model.fine_dim)[:len(target)]
    return c,f
