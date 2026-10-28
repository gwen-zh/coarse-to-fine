"""Coordinate-level two-scale contract, independent of ESM and quantization.

P averages CA positions in fixed contiguous residue blocks. U places each
block's anchor at every atom in that block. H=I-UP removes ONLY the block CA
centroid, retaining within-residue N/C geometry even for a final singleton.
Thus P(U(c)+H(r))=c. This enforces anchor separation, not all topology or
statistical independence. Coarse anchors alone are not a physical backbone.
"""
import torch


def _layout(n, block_size, device):
    if n < 1 or not isinstance(block_size,int) or block_size < 1:
        raise ValueError('Positive chain length and integer block size required')
    ids=torch.arange(n,device=device)//block_size
    counts=torch.bincount(ids,minlength=(n+block_size-1)//block_size)
    return ids,counts


def pool_ca(backbone,block_size=4):
    """One unpadded chain [N,3atoms,3xyz], coordinates in a common frame."""
    if backbone.ndim!=3 or backbone.shape[1:]!=(3,3) or not backbone.is_floating_point():
        raise ValueError('Expected floating [N, N/CA/C, xyz]')
    ids,counts=_layout(len(backbone),block_size,backbone.device)
    out=backbone.new_zeros(len(counts),3).index_add(0,ids,backbone[:,1])
    return out/counts[:,None].to(backbone.dtype)


def lift_anchors(anchors,length,block_size=4):
    ids,counts=_layout(length,block_size,anchors.device)
    if anchors.shape!=(len(counts),3):raise ValueError('Anchor shape differs from length')
    return anchors[ids,None,:].expand(length,3,3)


def remove_block_translation(residual,block_size=4):
    return residual-lift_anchors(pool_ca(residual,block_size),len(residual),block_size)


def compose_levels(coarse_anchors,raw_fine_residual,block_size=4):
    return lift_anchors(coarse_anchors,len(raw_fine_residual),block_size)+remove_block_translation(raw_fine_residual,block_size)


def split_reference(backbone,block_size=4):
    """Training decomposition only. Generation never calls this with a reference."""
    return pool_ca(backbone,block_size),remove_block_translation(backbone,block_size)
