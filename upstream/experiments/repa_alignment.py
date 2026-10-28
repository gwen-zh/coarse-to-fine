"""REPA-style structural latent target, not a geometry-distance guarantee."""
import torch
from torch.nn import functional as F
from coarse_lora import LoRALinear


def alignment_loss(prediction,target,mean,scale):
    target=((target.detach().float()-mean)/scale).detach()
    return (1-F.cosine_similarity(prediction.float(),target,dim=-1,eps=1e-8)).mean()


def install_repa_adapters(model,rank=8):
    model.requires_grad_(False);blocks=model.net.transformer.blocks;layer=max(1,len(blocks)//3)-1
    indices=sorted(set([max(0,layer-1),layer]+list(range(len(blocks)-4,len(blocks)))))
    for index in indices:
        attention=blocks[index].attn
        attention.layernorm_qkv[1]=LoRALinear(attention.layernorm_qkv[1],rank)
        attention.out_proj=LoRALinear(attention.out_proj,rank)
    head=model.net.output_heads.fine_structure_head
    head[-1]=LoRALinear(head[-1],rank)
    width=blocks[layer].attn.out_proj.base.in_features
    projector=torch.nn.Sequential(torch.nn.Linear(width,256),torch.nn.SiLU(),torch.nn.Linear(256,128)).to(next(model.parameters()).device)
    return projector,dict(alignment_block_zero_based=layer,lora_blocks=indices,rank=rank,width=width)
