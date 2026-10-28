"""Portable loaders for the compact, trusted project snapshots."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/experiments'))
from hierarchical_rigid_vq import HierarchicalRigidVQ
from mixed_hierarchical_generator import HybridTokenizer,MixedHierarchicalGenerator

def load_tokenizer(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=HierarchicalRigidVQ(**state['model_config']);model.load_state_dict(state['model'],strict=True)
    return HybridTokenizer(model.to(device))

def load_generator(path,device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    weights=state['model']
    model=MixedHierarchicalGenerator(weights['mean'],weights['std'],**state['model_config'])
    model.load_state_dict(weights,strict=True)
    return model.to(device).eval().requires_grad_(False)
