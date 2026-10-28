"""CPU/GPU native reconstruction demo; no external artifacts directory required."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from project_model import ROOT,load_tokenizer
from hierarchical_rigid_vq import canonical_backbone
from hierarchical_latent_contract import pool_ca
from structural_metrics import aligned_rmsd

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/demo_representation')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();torch.set_num_threads(args.threads)
    with np.load(args.input) as z:x=torch.from_numpy(z['backbone'].copy()).float().to(args.device)
    if x.ndim!=3 or x.shape[1:]!=(3,3) or not torch.isfinite(x).all():raise ValueError('Input must contain finite backbone[N,3,3] in N/CA/C order')
    decoder=load_tokenizer(args.checkpoint,args.device);c,f=decoder.encode(x)
    centers=decoder.coarse_centers(c,len(x));bb=decoder.decode(c,f);reference=canonical_backbone(x)
    torch.testing.assert_close(pool_ca(bb),centers,atol=1e-5,rtol=1e-6)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'coordinates.npz',reference=reference.cpu().numpy(),backbone=bb.cpu().numpy(),coarse_centers=centers.cpu().numpy(),coarse=c.cpu().numpy(),fine=f.cpu().numpy())
    result=dict(residues=len(x),coarse_blocks=len(c),fine_tokens=len(f),fine_vocabulary=decoder.model.fine_book.size,ca_reconstruction_rmsd_A=aligned_rmsd(bb[:,1].cpu().numpy(),x[:,1].cpu().numpy()),max_center_component_error_A=float((pool_ca(bb)-centers).abs().max()),scope='Native structure reconstruction,not sequence-only generation.')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,4));arrays=[reference[:,1].cpu().numpy(),centers.cpu().numpy()];i=len(c)//2;start=4*i;stop=min(start+4,len(x));local=bb[start:stop].cpu().numpy()-centers[i].cpu().numpy()
    for index,(a,title,color) in enumerate([(arrays[0],'Original structure','#30343b'),(arrays[1],'Coarse block centers','#cc7d28'),(local.reshape(-1,3),'Fine detail: coarse fixed','#306a9b')]):
        ax=fig.add_subplot(1,3,index+1,projection='3d');ax.plot(*a.T,color=color,lw=1.7)
        if index>0:ax.scatter(*a.T,color=color,s=12)
        if index==2:ax.scatter(0,0,0,color='#cc7d28',s=50)
        extent=a if index==2 else arrays[0];mid=(extent.min(0)+extent.max(0))/2;r=max(float(np.ptp(extent,axis=0).max())*.55,1.)
        for setter,mean in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):setter(mean-r,mean+r)
        ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(22,-55);ax.set_title(title)
    fig.tight_layout();fig.savefig(args.output/'representation.png',dpi=180);fig.savefig(args.output/'representation.pdf');plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
