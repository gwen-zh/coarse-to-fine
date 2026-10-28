"""Synthetic architecture/gradient smoke check: no protein data or weights."""
import argparse,json
import torch
from project_model import ROOT
from hierarchical_rigid_vq import HierarchicalRigidVQ
from mixed_hierarchical_generator import HybridTokenizer,MixedHierarchicalGenerator
from hierarchical_latent_contract import pool_ca

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(19);device=torch.device(args.device)
    base=HierarchicalRigidVQ(width=32,layers=1,coarse_size=8,fine_size=16).to(device)
    base.fine_book.initialize(torch.randn(128,8,device=device),2);decoder=HybridTokenizer(base)
    generator=MixedHierarchicalGenerator(torch.zeros(16),torch.ones(16),condition_dim=12,width=32,coarse_layers=1,fine_layers=1,fine_size=16).to(device).eval()
    features=torch.randn(17,12,device=device)
    with torch.no_grad():c,f=generator.sample(features,19,coarse_steps=3,fine_steps=3);bb=decoder.decode(c,f)
    torch.testing.assert_close(pool_ca(bb),decoder.coarse_centers(c,17),atol=2e-5,rtol=1e-6)
    assert torch.isfinite(bb).all()
    print(json.dumps(dict(status='passed',synthetic_only=True,coarse_shape=list(c.shape),fine_shape=list(f.shape),backbone_shape=list(bb.shape),scope='Random initialized models/syntheticfeatures;onlysoftwarecompatibility,nottrainedproteinprediction.'),indent=2))
if __name__=='__main__':main()
