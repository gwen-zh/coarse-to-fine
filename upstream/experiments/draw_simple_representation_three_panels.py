"""One protein, three structural views; native reconstruction, current tokenizer.

Coarse centers use the real coarse-only decoder. Fine atoms use native fine
IDs together with the SAME coarse latent. No zeroed/empty fine input.
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import ConnectionPatch
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_rigid_vq import canonical_backbone
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

FIG=Path('paper/figures/supervisor_representation_simple_v1')
DATA=Path('artifacts/representation_three_panel_v1')
NAME='109L_1';SEED=2026
BLACK='#30343b';BLUE='#306a9b';ORANGE='#cc7d28';GRAY='#b0b5bc'


def axis3d(ax,points,elev=22,azim=-55,zoom=1.18):
    xyz=np.asarray(points).reshape(-1,3);low=xyz.min(0);high=xyz.max(0)
    center=(high+low)/2.;radius=max(high-low)*.54
    for setter,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center):setter(m-radius,m+radius)
    ax.set_box_aspect((1,1,1),zoom=zoom);ax.set_proj_type('ortho');ax.view_init(elev,azim);ax.set_axis_off()


def backbone_bonds(ax,bb,color,lw=1.7,alpha=1.):
    xyz=np.asarray(bb).reshape(-1,3)
    ax.plot(*xyz.T,color=color,lw=lw,alpha=alpha,solid_capstyle='round')


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    FIG.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    parent=Path('artifacts/mixed_hierarchical_generator_v1/protocol.json')
    p=json.loads(parent.read_text());row=next(r for r in p['targets'] if r['id']==NAME);paths=p['tokenizers'][str(SEED)]
    with np.load(row['reference']) as z:x=torch.from_numpy(z['backbone'].copy()).float()
    decoder=HybridTokenizer.from_checkpoints(paths['warmup'],paths['final'])
    c,f=decoder.encode(x);centers=decoder.coarse_centers(c,len(x));decoded=decoder.decode(c,f);reference=canonical_backbone(x)
    np.testing.assert_allclose(pool_ca(decoded).numpy(),centers.numpy(),atol=1e-5,rtol=0)
    label=Path(f'artifacts/mixed_hierarchical_generator_v1/labels/seed{SEED}/{NAME}.npz')
    with np.load(label) as z:
        np.testing.assert_array_equal(f.numpy(),z['fine']);np.testing.assert_allclose(c.numpy(),z['coarse'],atol=1e-6,rtol=1e-6)
    # Keep the previously fixed middle block; no selection by reconstruction error.
    block=min(max(1,len(c)//2),len(c)-2);start=4*block;stop=min(start+4,len(x));indices=np.arange(start,stop)
    ref=reference.numpy();bb=decoded.numpy();cc=centers.numpy();n=len(x)
    np.savez_compressed(DATA/'coordinates.npz',reference=ref,decoded=bb,coarse_centers=cc,coarse_latent=c.numpy(),fine_ids=f.numpy(),highlighted_residues_zero_based=indices)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
    fig=plt.figure(figsize=(12.6,5.3))
    fig.text(.045,.965,f'{NAME}  |  Coarse and fine representations',fontsize=15,va='top',color=BLACK)
    positions=[[.025,.17,.30,.66],[.355,.17,.30,.66],[.685,.17,.30,.66]]
    axes=[fig.add_axes(pos,projection='3d') for pos in positions]
    for ax,label_text in zip(axes,['a  Original structure','b  Coarse block centers','c  Fine local detail']):
        ax.text2D(.02,1.05,label_text,transform=ax.transAxes,fontsize=13,color=BLACK)
    ax=axes[0]
    ax.plot(*ref[:,1].T,color=BLACK,lw=2.1,solid_capstyle='round')
    ax.plot(*ref[indices,1].T,color=BLUE,lw=4.2,solid_capstyle='round');ax.scatter(*ref[indices,1].T,color=BLUE,s=14,depthshade=False)
    axis3d(ax,ref[:,1])
    ax=axes[1]
    ax.plot(*cc.T,color=ORANGE,lw=1.8,solid_capstyle='round');ax.scatter(*cc.T,color=ORANGE,s=20,depthshade=False)
    ax.scatter(*cc[block],s=135,facecolors='none',edgecolors=BLUE,linewidths=1.7,depthshade=False)
    axis3d(ax,ref[:,1])
    # A shared visual frame for the zoom only; coarse and fine are never
    # independently realigned. Center the selected decoded block at the origin.
    local=bb[indices]-cc[block];native_local=ref[indices]-cc[block]
    ax=axes[2]
    backbone_bonds(ax,native_local,GRAY,lw=1.2,alpha=.9)
    backbone_bonds(ax,local,BLUE,lw=2.)
    ax.scatter(*local.reshape(-1,3).T,color=BLUE,s=24,depthshade=False)
    ax.scatter(0.,0.,0.,color=ORANGE,s=75,marker='o',edgecolors='white',linewidths=.6,depthshade=False)
    for ca in local[:,1]:ax.plot(*np.stack([np.zeros(3),ca]).T,color=ORANGE,lw=.75,ls=':',alpha=.75)
    # Label atoms of one actual residue, not all atoms in the zoom.
    for atom,xyz,offset in zip(['N','CA','C'],local[1],[[.12,.08,.10],[.12,-.08,.10],[.12,.08,.10]]):
        pos=xyz+np.array(offset);ax.text(*pos,atom,fontsize=9,color=BLUE)
    axis3d(ax,np.concatenate([local.reshape(-1,3),native_local.reshape(-1,3),np.zeros((1,3))]),elev=25,azim=-63,zoom=1.12)
    legend=[Line2D([],[],color=GRAY,lw=1.3,label='Reference'),Line2D([],[],color=BLUE,lw=2,label='Decoded N/CA/C'),Line2D([],[],color=ORANGE,marker='o',ls='None',ms=5,label='Fixed block center')]
    ax.legend(handles=legend,loc='lower center',bbox_to_anchor=(.48,-.07),ncol=1,frameon=False,fontsize=8,handlelength=1.5,labelspacing=.35)
    # Sparse captions; interpretation belongs in the spoken explanation.
    fig.text(.175,.135,f'{n} residues',ha='center',fontsize=11,color=BLACK)
    fig.text(.505,.135,f'{len(cc)} centers',ha='center',fontsize=11,color=BLACK)
    fig.text(.835,.105,f'Residues {start+1}–{stop}; same coarse input',ha='center',fontsize=10,color=BLACK)
    fig.text(.045,.035,'Native-structure reconstruction. Each coarse center represents a block of up to four residues.',fontsize=9,color='#666b72')
    fig.canvas.draw()
    # Link the selected real coarse center to the magnified atoms. This is a
    # visual zoom guide; it does not denote a physical bond or generation step.
    projected=proj3d.proj_transform(*cc[block],axes[1].get_proj())
    guide=ConnectionPatch(xyA=(projected[0],projected[1]),coordsA=axes[1].transData,xyB=(.04,.52),coordsB=axes[2].transAxes,arrowstyle='-',ls=(0,(3,4)),lw=.8,color='#a2a7af',alpha=.65,zorder=2)
    fig.add_artist(guide)
    for extension in ['png','pdf','svg']:fig.savefig(FIG/f'Coarse_Fine_One_Protein.{extension}',dpi=250)
    plt.close(fig)
    sources=[__file__,str(parent),row['reference'],str(label),paths['warmup'],paths['final'],'upstream/experiments/mixed_hierarchical_generator.py','upstream/experiments/hierarchical_rigid_vq.py','upstream/experiments/hierarchical_latent_contract.py']
    report=dict(status='complete',protein=NAME,seed=SEED,length=n,coarse_blocks=len(cc),highlighted_residues_one_based=(indices+1).tolist(),coarse_only_path='HybridTokenizer.coarse_centers(c,length)',fine_path='HybridTokenizer.decode(the_same_c,native_fine_ids)',max_block_center_component_error_A=float((pool_ca(decoded)-centers).abs().max()),native_fine_cache_exact=True,example_policy='Same previously fixed protein and middle block; no example selection by outcome.',scope='Single native-structure reconstruction example;fine shown jointly with fixed coarse and decoder context;not evidence of learned topology or sequence-generation improvement.',source_sha256={v:sha(v) for v in sources},figures={str(q):sha(q) for q in FIG.iterdir() if q.suffix in ['.png','.pdf','.svg']})
    atomic_json(DATA/'manifest.json',report)
    (FIG/'talking_points.txt').write_text('This is the same protein in all three panels. The left panel shows the reference structure. The middle panel shows the block centers decoded directly from the coarse representation. The right panel zooms into the marked block and shows the local N/CA/C positions reconstructed using the fine tokens while keeping the coarse input fixed.\n\n中文：左边是真实结构，中间是 coarse 解码得到的块中心，右边放大同一个块，展示固定 coarse 后用 fine 重建出的局部原子位置。\n\nSolid lines between block centers and dotted center-to-CA guides are visual aids, not chemical bonds.\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
