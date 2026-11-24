"""Fine-decoder-only repair with fixed structure encoders, books and anchors."""
import torch
from torch.nn import functional as F
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_pipeline_io import sha


def enable_fine_decoder_training(tokenizer):
    tokenizer.eval().requires_grad_(False)
    tokenizer.model.fine_decoder.train().requires_grad_(True)
    return list(tokenizer.model.fine_decoder.parameters())


def frozen_structure_state(tokenizer):
    return {key: value.detach().cpu().clone() for key, value in tokenizer.state_dict().items()
            if not key.startswith('model.fine_decoder.')}


def assert_frozen_structure(tokenizer, before):
    now = tokenizer.state_dict()
    for key, value in before.items():
        if not torch.equal(value, now[key].detach().cpu()):
            raise RuntimeError('Frozen structure component changed: '+key)


def repair_loss(prediction, target, geometry=False, block_size=4):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (3,3) or len(target)<5:
        raise ValueError('Matching finite complete N/CA/C chains of at least five residues required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite repair coordinates')
    target = target.detach()
    coordinate = (prediction-target).square().mean()
    def cn(x):
        return (x[:-1,2]-x[1:,0]).norm(dim=-1)
    cn_error = (cn(prediction)-cn(target)).square()
    ids = torch.arange(len(target)-1, device=target.device)
    boundary = ids//block_size != (ids+1)//block_size
    def cosines(x):
        def cosine(a,b):
            return (F.normalize(a,dim=-1,eps=1e-6)*F.normalize(b,dim=-1,eps=1e-6)).sum(-1)
        return torch.cat([cosine(x[:-1,1]-x[:-1,2],x[1:,0]-x[:-1,2]),
                          cosine(x[:-1,2]-x[1:,0],x[1:,1]-x[1:,0])])
    angle = (cosines(prediction)-cosines(target)).square().mean()
    residue_ids = torch.arange(len(target), device=target.device)
    nonlocal_pair = (residue_ids[:,None]-residue_ids[None]).abs()>1
    clash = F.relu(3.-torch.cdist(prediction[:,1],prediction[:,1])[nonlocal_pair]).square().mean()
    terms = dict(coordinate=coordinate, cn=cn_error.mean(), boundary_cn=cn_error[boundary].mean(), angle=angle, clash=clash)
    loss = coordinate
    if geometry:
        loss = loss+terms['cn']+terms['boundary_cn']+.25*angle+.1*clash
    return loss, {key:value.detach() for key,value in terms.items()}


def load_repaired_tokenizer(warmup, final, repair_checkpoint):
    """Only local trusted training checkpoints; a repair never replaces encoders."""
    state = torch.load(repair_checkpoint, map_location='cpu', weights_only=False)
    for path in [warmup, final]:
        if sha(path) != state['parent_checkpoint_sha256'][str(path)]:
            raise RuntimeError('Repair does not match these frozen structural checkpoints')
    tokenizer = HybridTokenizer.from_checkpoints(warmup,final)
    tokenizer.model.fine_decoder.load_state_dict(state['fine_decoder'], strict=True)
    return tokenizer.eval().requires_grad_(False)
