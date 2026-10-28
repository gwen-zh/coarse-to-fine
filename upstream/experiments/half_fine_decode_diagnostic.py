"""Reference-free decoder-embedding projection for fixed half-fine outputs."""
import torch


def nearest_decoder_tokens(predicted, table, scale, center=None):
    """Nearest ordinary fine embedding in the training-standardized metric.

    This is decoder-input space (1280D in the real model), not encoder128D.
    No reference structure or true hidden token is used to choose a candidate.
    """
    if predicted.ndim!=2 or table.ndim!=2 or predicted.shape[1]!=table.shape[1]:
        raise ValueError('Prediction and decoder table must share their embedding dimension')
    if scale.shape!=(table.shape[1],) or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid channel scale')
    if not torch.isfinite(predicted).all() or not torch.isfinite(table).all():raise ValueError('Nonfinite input')
    center=table.mean(0) if center is None else center
    query=(predicted-center)/scale;codes=(table-center)/scale
    distance=query.square().sum(-1,keepdim=True)+codes.square().sum(-1)[None]-2*query@codes.T
    return distance.argmin(-1)


def decode_variants(predicted, original, keep, table, scale, center=None):
    if predicted.shape!=original.shape or keep.shape!=(len(predicted),) or keep.dtype!=torch.bool:
        raise ValueError('Invalid mask or embedding shapes')
    visible_fixed=torch.where(keep[:,None],original,predicted)
    hidden_ids=nearest_decoder_tokens(predicted[~keep],table,scale,center)
    snapped=visible_fixed.clone();snapped[~keep]=table[hidden_ids]
    return {'continuous':predicted,'visible_fixed':visible_fixed,'hidden_snapped':snapped},hidden_ids
