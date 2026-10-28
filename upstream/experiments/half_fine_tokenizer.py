"""Fixed half-fine transmission with an optional discrete geometry side channel.

Encoder may inspect the complete true structure; fusion sees only transmitted
fine tokens, coarse IDs and sequence positions. No structural geometry bypass.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_tokenizer import TwoScaleTokenizer


def retained_positions(length, device=None):
    return torch.arange(length, device=device) % 2 == 0


def sanitize_fine(codes, embeddings, keep, mean_embedding):
    if keep.shape != (len(codes),) or keep.dtype != torch.bool:
        raise ValueError('Expected one boolean visibility per residue')
    visible_codes = torch.where(keep[:, None], codes, torch.zeros_like(codes))
    visible_embeddings = torch.where(keep[:, None], embeddings, mean_embedding.expand_as(embeddings))
    return visible_codes, visible_embeddings


def position_features(length, device, dim=32):
    pos=torch.arange(length,device=device).float()[:,None]
    frequencies=torch.exp(-math.log(10000)*torch.arange(0,dim,2,device=device)/dim)
    return torch.cat(((pos*frequencies).sin(),(pos*frequencies).cos()),-1)


class PartialFineFusion(nn.Module):
    def __init__(self, decoder_dim=1280):
        super().__init__()
        self.input=nn.Linear(128+32+1+32,128)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.context=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128)
        self.output=nn.Linear(128,decoder_dim)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self, fine_codes, fine_embedding, keep, coarse, scale, mean_embedding):
        visible,base=sanitize_fine(fine_codes,fine_embedding,keep,mean_embedding)
        x=torch.cat((F.layer_norm(visible,(128,)),coarse*math.sqrt(32),
                     keep[:,None].to(visible.dtype),position_features(len(visible),visible.device)), -1)
        h=self.context(self.input(x)[None])[0]
        raw=self.output(self.norm(h))
        # Hidden tokens must be reconstructed, not restricted to a tiny repair.
        amplitude=torch.where(keep,.1,4.).to(raw.dtype)[:,None]
        return base+amplitude*scale*torch.tanh(raw)


class HalfFineTokenizer(TwoScaleTokenizer):
    def __init__(self, mode, decoder_dim=1280):
        if mode not in ('none','local','global'):raise ValueError(mode)
        super().__init__('local' if mode=='none' else mode,decoder_dim=decoder_dim)
        self.mode=mode
        self.fusion=PartialFineFusion(decoder_dim)
        if mode=='none':
            for module in (self.input,self.blocks,self.project):module.requires_grad_(False)

    def coarse(self, full_fine_codes, pair, local_mask):
        if self.mode=='none':
            return full_fine_codes.new_zeros(len(full_fine_codes),32),None,full_fine_codes.new_tensor(0.)
        return self.quantizer(self.encode(full_fine_codes,pair,local_mask))

    def forward(self, codes, embeddings, pair, mask, scale, mean_embedding):
        coarse,ids,commitment=self.coarse(codes,pair,mask)
        keep=retained_positions(len(codes),codes.device)
        output=self.fusion(codes,embeddings,keep,coarse,scale,mean_embedding)
        return output,ids,commitment


def hidden_embedding_loss(predicted, teacher, keep, scale):
    return ((predicted[~keep]-teacher[~keep])/scale.clamp_min(1e-6)).square().mean()
