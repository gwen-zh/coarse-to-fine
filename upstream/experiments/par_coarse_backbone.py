"""Pinned PAR coarse-scale transfer with an initially zero AA-conditioning path.

The official source is unmodified. Its checkpoint disables OpenFold pair/IPA
blocks, whose imports otherwise require an unused CUDA extension. Explicit
raise-on-use import guards permit this verified no-pair configuration only;
no active numerical operator is replaced.
"""
import copy
import importlib.machinery
import os
from pathlib import Path
import sys
import types
import typing

import torch
from torch import nn
from torch.nn import functional as F


ROOT=Path("artifacts/par_coarse_backbone_v1")
CHECKPOINT=ROOT/"official_checkpoint/PAR_3scale_by_ratio_60m_200k.ckpt"
GUARDED_MODULES={
    "openfold.model.msa":["MSARowAttentionWithPairBias"],
    "openfold.model.pair_transition":["PairTransition"],
    "openfold.model.structure_module":["InvariantPointAttention"],
    "openfold.model.triangular_multiplicative_update":["TriangleMultiplicationIncoming","TriangleMultiplicationOutgoing"],
}


def guard_unused_openfold(config):
    for branch in [config.model.nn,config.model.atom_diff.nn]:
        if any(branch.get(flag,False) for flag in ["use_attn_pair_bias","update_pair_repr","use_tri_mult"]):
            raise RuntimeError("Optional OpenFold guards support only verified no-pair/no-triangle PAR configs")
    for module_name,names in GUARDED_MODULES.items():
        if module_name in sys.modules:
            if getattr(sys.modules[module_name],"_par_no_pair_guard",False):continue
            raise RuntimeError("Install guards in a fresh isolated PAR process")
        module=types.ModuleType(module_name)
        module._par_no_pair_guard=True
        module.__spec__=importlib.machinery.ModuleSpec(module_name,loader=None)
        for name in names:
            def unavailable(self,*args,_name=name,**kwargs):
                raise RuntimeError("An inactive OpenFold branch was unexpectedly instantiated: "+_name)
            setattr(module,name,type(name,(nn.Module,),{"__init__":unavailable}))
        sys.modules[module_name]=module


def load_official(device="cpu"):
    # Use only this checkpoint's fully specified config; do not fill absent model keys.
    saved=torch.load(CHECKPOINT,map_location="cpu",mmap=True,weights_only=False)
    config=copy.deepcopy(saved["hyper_parameters"]["cfg_exp"])
    guard_unused_openfold(config)
    # Upstream annotates several methods with `from torch import Dict`.
    # Current torch exposes Tensor but not this typing-only alias.
    if not hasattr(torch,"Dict"):torch.Dict=typing.Dict
    from proteinfoundation.protein_autoregressive_modeling.par import PAR
    metadata_dir=(ROOT/"runtime_metadata").resolve();metadata_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("DATA_PATH",str(metadata_dir))
    model=PAR(config,store_dir=str(ROOT/"official_runtime"))
    # The public checkpoint predates removal of this nonlearned level-index
    # buffer in the released flow feature factory. Restore its exact definition
    # rather than dropping a checkpoint key or relaxing strict loading.
    key="atom_diffusion.nn.cond_factory.feat_creators.1.lvl_1L"
    owner=model.atom_diffusion.nn.cond_factory.feat_creators[1]
    if key in saved["state_dict"] and not hasattr(owner,"lvl_1L"):
        levels=torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(owner.num_local_centroids)])[None]
        torch.testing.assert_close(levels,saved["state_dict"][key],atol=0,rtol=0)
        owner.register_buffer("lvl_1L",levels)
    model.load_state_dict(saved["state_dict"],strict=True)
    model.eval().requires_grad_(False)
    return model.to(device),config


def coarse_batch(length,points,device,dtype=torch.float32,batch_size=1):
    if length<4 or points<1:
        raise ValueError("Positive coarse length and at least four residues required")
    mask=torch.ones((batch_size,points),device=device,dtype=torch.bool)
    return dict(x_input=torch.zeros((batch_size,points,3),device=device,dtype=dtype),
        mask=mask,attn_mask=None,cur_scale=torch.tensor(0,device=device,dtype=torch.int32),
        int_pos_idx=torch.linspace(1,length,points,device=device,dtype=dtype)[None].expand(batch_size,-1),
        cur_nres=points,norm_lengths=torch.full((batch_size,),length/256.,device=device,dtype=dtype))


class SequenceConditionedFactory(nn.Module):
    """Preserve official features and add zero-initialized ordered AA blocks."""
    def __init__(self,base,feature_dim=1536,width=128,output_dim=512):
        super().__init__();self.base=base.eval().requires_grad_(False)
        self.feature_dim=feature_dim;self.output_dim=output_dim
        self.residue_projection=nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,width),nn.SiLU())
        self.block_projection=nn.Linear(4*width+4,output_dim)
        nn.init.zeros_(self.block_projection.weight);nn.init.zeros_(self.block_projection.bias)
        self.features=None

    def aa_blocks(self,features):
        if features.ndim!=2 or features.shape[1]!=self.feature_dim or not torch.isfinite(features).all():
            raise ValueError("Finite unpadded AA-only feature matrix required")
        h=self.residue_projection(features.detach());n=len(h);m=(n+3)//4
        padded=F.pad(h,(0,0,0,4*m-n)).reshape(m,-1)
        valid=(torch.arange(4*m,device=h.device)<n).reshape(m,4).to(h.dtype)
        return self.block_projection(torch.cat([padded,valid],-1))[None]

    def forward(self,batch):
        original=self.base(batch)
        if self.features is None:return original
        condition=self.aa_blocks(self.features.to(device=original.device,dtype=original.dtype))
        if original.shape[1:]!=condition.shape[1:]:
            raise ValueError("This initial adapter supports only coarse ceil(N/4) for one AA sequence")
        return original+condition.expand(original.shape[0],-1,-1)*batch["mask"][...,None]


def attach_sequence_conditioner(model,feature_dim=1536,width=128):
    original=model.nn.init_repr_factory
    wrapper=SequenceConditionedFactory(original,feature_dim,width,model.cfg_exp.model.nn.token_dim)
    wrapper=wrapper.to(next(model.parameters()).device)
    model.nn.init_repr_factory=wrapper
    return wrapper


def flow_batch(ar_input,condition,state,time,self_condition=None):
    times=torch.as_tensor(time,device=state.device,dtype=state.dtype).reshape(-1)
    if times.numel()==1:times=times.expand(len(state))
    if times.shape!=(len(state),):raise ValueError("One time per sample required")
    batch=dict(ar_input);batch.update(z_cond=condition,s_cond=ar_input["x_input"],x_t=state,t=times)
    if self_condition is not None:batch["x_sc"]=self_condition
    return batch


def ar_condition(model,features,length,batch_size=1):
    points=(length+3)//4;device=next(model.parameters()).device
    batch=coarse_batch(length,points,device,batch_size=batch_size)
    factory=model.nn.init_repr_factory
    if isinstance(factory,SequenceConditionedFactory):factory.features=features
    # Never retain cross-sample AR KV history for this coarse-only transfer.
    for block in model.nn.transformer_layers:block.mhba.mha.kv_caching(False)
    return batch,model.nn(batch)["cond"]


@torch.no_grad()
def sample_coarse(model,features,length,seed,steps=16):
    """VF Euler diagnostic in native PAR nanometers; return weighted-centered Å.

    Uses pretrained modules, but ceil(N/4) centers and this explicit16-step
    solver are an interface pilot, not a reproduction of published PAR scores.
    """
    return sample_coarse_batch(model,features,length,[seed],steps)[0]


@torch.no_grad()
def sample_coarse_batch(model,features,length,seeds,steps=16):
    if steps<1 or not seeds:raise ValueError("Positive integration budget and sample count required")
    batch,condition=ar_condition(model,features,length,len(seeds));device=condition.device
    noise=torch.stack([torch.randn(((length+3)//4,3),generator=torch.Generator().manual_seed(int(seed))) for seed in seeds])
    state=noise.to(device);state=state-state.mean(1,keepdim=True);self_condition=None
    for i in range(steps):
        b=flow_batch(batch,condition,state,i/steps,self_condition)
        predicted=model.atom_diffusion.nn(b)["coors_pred"]
        self_condition=(state+(1-i/steps)*predicted).detach()
        state=state+predicted/steps;state=state-state.mean(1,keepdim=True)
    result=state*10.
    counts=result.new_tensor([min(4,length-4*j) for j in range(result.shape[1])])
    return result-(counts[None,:,None]*result).sum(1,keepdim=True)/length
