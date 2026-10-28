"""Original fine tail with a zero-initialized residual-code condition.

ESM block imports are deferred: generator and tokenizer ESM packages must run
in separate processes. All original tail weights are frozen.
"""
import torch
from contextlib import contextmanager
from torch import nn


class ConditionedFineTail(nn.Module):
    def __init__(self,width,heads,blocks,scaling_factor,code_width=128):
        super().__init__()
        from esm.layers.blocks import UnifiedTransformerBlock
        from esm.layers.regression_head import RegressionHead
        self.blocks=nn.ModuleList([UnifiedTransformerBlock(width,heads,use_geom_attn=False,
            residue_scaling_factor=scaling_factor,expansion_ratio=8/3,bias=False) for _ in range(blocks)])
        self.norm=nn.LayerNorm(width,bias=False)
        self.head=RegressionHead(width,4101)
        self.condition_norm=nn.LayerNorm(code_width,elementwise_affine=False)
        self.condition=nn.Linear(code_width,width,bias=False)
        nn.init.zeros_(self.condition.weight)
        for module in [self.blocks,self.norm,self.head]:module.requires_grad_(False)

    def forward(self,prefix,skip,condition=None):
        if prefix.shape!=skip.shape:raise ValueError('Fine prefix/skip shapes differ')
        x=prefix
        if condition is not None:
            if condition.shape[:2]!=(x.shape[0],x.shape[1]-2):raise ValueError('Condition length mismatch')
            delta=self.project_condition(condition)
            x=x+torch.cat([torch.zeros_like(x[:,:1]),delta,torch.zeros_like(x[:,-1:])],dim=1)
        chain=torch.zeros(x.shape[:2],dtype=torch.long,device=x.device)
        for block in self.blocks:x=block(x,None,None,None,chain)
        return self.head(self.norm(x)+skip)[:,1:-1,:4096]

    def project_condition(self,condition):
        return self.condition(self.condition_norm(condition))


@contextmanager
def condition_fine_stack(stack,project_condition,condition,tail_blocks=4):
    """Deploy trained condition only around a fine-stage forward, then remove it."""
    delta=project_condition(condition)
    def inject(module,inputs):
        x=inputs[0]
        if delta.shape[:2]!=(x.shape[0],x.shape[1]-2):raise ValueError('Live condition length mismatch')
        padding=torch.cat([torch.zeros_like(x[:,:1]),delta,torch.zeros_like(x[:,-1:])],1)
        return (x+padding,*inputs[1:])
    hook=stack.blocks[-tail_blocks].register_forward_pre_hook(inject)
    try:yield
    finally:hook.remove()


def load_tail(bundle):
    model=ConditionedFineTail(**bundle['config'])
    model.blocks.load_state_dict(bundle['blocks'],strict=True)
    model.norm.load_state_dict(bundle['norm'],strict=True)
    model.head.load_state_dict(bundle['head'],strict=True)
    return model.eval()


class MidPredictor(nn.Module):
    def __init__(self,width=1536,size=512):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,512),nn.GELU(),nn.Linear(512,size))
    def forward(self,h):return self.net(h)


def residual_target(latent,coarse,coarse_codes,mode):
    if mode not in ('residual','parallel'):raise ValueError('Unknown mid mode')
    if coarse.shape!=latent.shape[:-1] or torch.any((coarse<0)|(coarse>=len(coarse_codes))):
        raise ValueError('Invalid coarse condition')
    return latent-coarse_codes[coarse] if mode=='residual' else latent
