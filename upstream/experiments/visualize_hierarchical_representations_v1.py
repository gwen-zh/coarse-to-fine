"""Descriptive, native-structure representation interventions; no training/selection.

Current HYBRID tokenizer, not the original author's 4096-token model.
A fixed example and interventions are chosen before decoding. Reused dev only.
"""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mixed_hierarchical_generator import HybridTokenizer
from hierarchical_latent_contract import pool_ca
from hierarchical_pipeline_io import sha
from structural_metrics import aligned_rmsd
from diagnostic_io import atomic_json

RUN=Path('artifacts/representation_visualization_v1')
FIG=Path('paper/figures')
SEEDS=[2026,2027]


def rms(x):
    return float(np.sqrt(np.mean(np.sum(np.asarray(x,dtype=float)**2,axis=-1))))


def spatial(ax, xyz, label, color, alpha=1., lw=1.3):
    ax.plot(*xyz.T,color=color,alpha=alpha,lw=lw,label=label)


def style3d(ax, coordinates):
    low=np.min(coordinates,axis=0); high=np.max(coordinates,axis=0)
    middle=(low+high)/2.;radius=max(high-low)/2.*1.08
    for setter,center in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],middle):setter(center-radius,center+radius)
    ax.set_box_aspect((1,1,1));ax.view_init(23,-55);ax.set_axis_off()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    RUN.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    parent=Path('artifacts/mixed_hierarchical_generator_v1/protocol.json')
    p=json.loads(parent.read_text());rows=sorted([r for r in p['targets'] if r['subset']=='validation'],key=lambda r:r['id'])
    example=rows[0]['id']
    inputs=[str(parent),__file__,'upstream/experiments/hierarchical_rigid_vq.py','upstream/experiments/hierarchical_latent_contract.py','upstream/experiments/mixed_hierarchical_generator.py']
    for seed in SEEDS:
        inputs+=list(p['tokenizers'][str(seed)].values())
        inputs+=[f'artifacts/mixed_generator_position_support_v1/train_counts_seed{seed}.npy']
    inputs += [r['reference'] for r in rows]
    manifest=dict(seeds=SEEDS,proteins=len(rows),example_rule='Lexicographically first reused-development ID, seed2026, chosen before decoding',example=example,
        fine_intervention='At residue floor(N/2), replace one fine ID by nearest different code in Euclidean code space among codes observed at this block slot in TRAIN. Coarse unchanged.',
        coarse_intervention='At the middle interior coarse block, replace c_j by 0.5*c_j+0.25*c_(j-1)+0.25*c_(j+1). Fine IDs unchanged.',
        coordinate_change='Same canonical frame, no alignment; rigid alignment would obscure locality. All interventions are descriptive sensitivity tests, not physical conformations or a generation improvement.',
        scope='Native structure encode/decode on all101reuseddev,not sequence generation,not independent confirmation. Fixed4residueblocks enforce centroid separation by construction; not a test of learned topology/statistical independence. Coarse edit may also change local geometry via decoder context.',
        source_sha256={name:sha(name) for name in inputs})
    atomic_json(RUN/'protocol.json',manifest)
    records=[];examples={};pca_data={}
    for seed in SEEDS:
        paths=p['tokenizers'][str(seed)];decoder=HybridTokenizer.from_checkpoints(paths['warmup'],paths['final'])
        counts=np.load(f'artifacts/mixed_generator_position_support_v1/train_counts_seed{seed}.npy');codes=decoder.model.fine_book.codes
        code_np=codes.numpy().astype(float);weights=counts.sum(1).astype(float);weights/=weights.sum()
        mean=(code_np*weights[:,None]).sum(0);centered=code_np-mean
        eig,vec=np.linalg.eigh(centered.T@(centered*weights[:,None]));basis=vec[:,-2:][:,::-1]
        cloud=[];slots=[]
        for row in rows:
            with np.load(row['reference']) as z:bb=torch.from_numpy(z['backbone'].copy()).float()
            c,f=decoder.encode(bb);_,_,target=decoder.model.encode_latents(bb);baseline=decoder.decode(c,f)
            n=len(bb);i=n//2;block=i//4
            distance=(codes-codes[f[i]]).square().sum(-1)
            allowed=torch.from_numpy(counts[:,i%4]>0);allowed[f[i]]=False
            distance=distance.masked_fill(~allowed,float('inf'));replacement=int(distance.argmin())
            if not torch.isfinite(distance[replacement]):raise RuntimeError('No training-supported alternative code')
            edited_f=f.clone();edited_f[i]=replacement;fine_bb=decoder.decode(c,edited_f)
            j=min(max(1,len(c)//2),len(c)-2);edited_c=c.clone();edited_c[j]=.5*c[j]+.25*c[j-1]+.25*c[j+1]
            coarse_bb=decoder.decode(edited_c,f)
            base_center=pool_ca(baseline);fine_center=pool_ca(fine_bb);coarse_center=pool_ca(coarse_bb)
            fine_delta=(fine_bb[:,1]-baseline[:,1]).norm(dim=-1).numpy();coarse_delta=(coarse_bb[:,1]-baseline[:,1]).norm(dim=-1).numpy()
            outside=torch.tensor([abs(k//4-block)>1 for k in range(n)])
            centroid_max=float((fine_center-base_center).abs().max());outside_max=float((fine_bb[outside]-baseline[outside]).abs().max()) if outside.any() else 0.
            if centroid_max>1e-5 or outside_max>1e-6:raise RuntimeError('Unexpected violation of decoder dependency contract')
            cache=Path(f'artifacts/mixed_hierarchical_generator_v1/labels/seed{seed}/{row["id"]}.npz')
            with np.load(cache) as z:
                if not np.array_equal(f.numpy(),z['fine']) or not np.allclose(c.numpy(),z['coarse'],atol=1e-6,rtol=1e-6):raise RuntimeError('Native cache mismatch')
            records.append(dict(seed=seed,target=row['id'],length=n,baseline_ca_rmsd=aligned_rmsd(baseline[:,1].numpy(),bb[:,1].numpy()),fine_edited_residue_zero_based=i,original_fine_id=int(f[i]),replacement_fine_id=replacement,coarse_edited_block_zero_based=j,
                fine_ca_displacement_rms=rms(fine_bb[:,1].numpy()-baseline[:,1].numpy()),coarse_ca_displacement_rms=rms(coarse_bb[:,1].numpy()-baseline[:,1].numpy()),fine_block_center_displacement_rms=rms(fine_center.numpy()-base_center.numpy()),coarse_block_center_displacement_rms=rms(coarse_center.numpy()-base_center.numpy()),max_fine_center_component_error=centroid_max,max_fine_effect_outside_neighbor_blocks=outside_max,
                fine_residues_changed_above1e_5A=int((fine_delta>1e-5).sum()),coarse_residues_changed_above1e_5A=int((coarse_delta>1e-5).sum())))
            cloud.append((code_np[f.numpy()]-mean)@basis);slots.append(np.arange(n)%4)
            if row['id']==example:
                data=dict(reference=target.numpy(),baseline=baseline.numpy(),coarse_centers=base_center.numpy(),fine_edit=fine_bb.numpy(),coarse_edit=coarse_bb.numpy(),fine_displacement=fine_delta,coarse_displacement=coarse_delta,fine_index=i,coarse_block=j)
                examples[seed]=data;np.savez_compressed(RUN/f'example_seed{seed}.npz',**data)
        table=counts.astype(float);joint=table/table.sum();pr=joint.sum(1);ps=joint.sum(0);denom=pr[:,None]*ps[None,:];valid=joint>0
        mi=float((joint[valid]*np.log2(joint[valid]/denom[valid])).sum());purity=float(table.max(1).sum()/table.sum())
        pca_data[seed]=dict(x=np.concatenate(cloud),slot=np.concatenate(slots),mi=mi,purity=purity,explained=float(eig[-2:].sum()/eig.sum()))
    summary=[]
    for seed in SEEDS:
        selected=[r for r in records if r['seed']==seed];keys=['baseline_ca_rmsd','fine_ca_displacement_rms','coarse_ca_displacement_rms','fine_block_center_displacement_rms','coarse_block_center_displacement_rms','fine_residues_changed_above1e_5A','coarse_residues_changed_above1e_5A']
        summary.append(dict(seed=seed,proteins=len(selected),**{key:float(np.mean([r[key] for r in selected])) for key in keys},max_fine_center_component_error=max(r['max_fine_center_component_error'] for r in selected),max_fine_effect_outside_neighbor_blocks=max(r['max_fine_effect_outside_neighbor_blocks'] for r in selected),training_code_slot_mutual_information_bits=pca_data[seed]['mi'],training_code_slot_purity=pca_data[seed]['purity']))
    result=dict(status='complete',summary=summary,targets=records,protocol='artifacts/representation_visualization_v1/protocol.json',scope=manifest['scope'])
    atomic_json(RUN/'results.json',result)
    d=examples[2026];native=d['reference'][:,1];base=d['baseline'][:,1];centers=d['coarse_centers'];orange='#d97706';blue='#2463a5';purple='#8b4cb2'
    plt.rcParams.update({'font.size':10,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig=plt.figure(figsize=(15,9.6));grid=fig.add_gridspec(2,3,hspace=.30,wspace=.25)
    a=fig.add_subplot(grid[0,0],projection='3d');spatial(a,native,'Reference CA','#a1a1aa',lw=2.3);spatial(a,base,'Reconstructed CA',blue);a.scatter(*centers.T,color=orange,s=15,label='Decoded block centers');a.plot(*centers.T,color=orange,lw=.7);style3d(a,native);a.set_title('A  Coarse layout + reconstructed chain');a.legend(loc='lower left',fontsize=8,frameon=False)
    a=fig.add_subplot(grid[0,1]);true_centers=pool_ca(torch.from_numpy(d['reference'])).numpy();dt=np.linalg.norm(true_centers[:,None]-true_centers[None,:],axis=-1);dp=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1);image=np.where(np.triu(np.ones_like(dt),1)>0,dt,dp);im=a.imshow(image,cmap='viridis',origin='lower');fig.colorbar(im,ax=a,shrink=.7,label='Block-center distance (A)');a.set_title('B  Geometry carried by coarse');a.set_xlabel('Block index (upper: reference)');a.set_ylabel('Block index (lower: decoded)')
    a=fig.add_subplot(grid[0,2],projection='3d');residual=d['baseline']-np.repeat(centers,4,axis=0)[:len(base),None,:];col=plt.get_cmap('tab10');start=(d['fine_index']//4)*4
    for k in range(start,min(start+4,len(base))):
        xyz=residual[k];a.scatter(*xyz.T,color=col(k%4),s=[12,25,12]);a.plot(*xyz.T,color=col(k%4),lw=2,label=f'Residue slot {k%4+1}')
    style3d(a,residual[start:start+4].reshape(-1,3));a.set_title('C  Fine: atoms relative to block center');a.legend(loc='lower left',fontsize=8,frameon=False)
    for ax_index,key,title,color in [(0,'coarse_edit','D  Change one coarse block; hold fine fixed',orange),(1,'fine_edit','E  Change one fine token; hold coarse fixed',purple)]:
        a=fig.add_subplot(grid[1,ax_index],projection='3d');spatial(a,base,'Original reconstruction','#a1a1aa',lw=2);spatial(a,d[key][:,1],'After intervention',color,lw=1.5);style3d(a,np.concatenate([base,d[key][:,1]]));a.set_title(title,fontsize=10);a.legend(loc='lower left',fontsize=8,frameon=False)
    a=fig.add_subplot(grid[1,2]);a.plot(np.arange(len(base))+1,d['coarse_displacement'],color=orange,label='One coarse-block edit');a.plot(np.arange(len(base))+1,d['fine_displacement'],color=purple,label='One fine-token edit');start=(d['fine_index']//4-1)*4;end=(d['fine_index']//4+2)*4;a.axvspan(max(start+1,1),min(end,len(base)),color=purple,alpha=.08,label='Fine decoder dependency window');a.set_xlabel('Residue index');a.set_ylabel('CA displacement, same frame (A)');a.set_title('F  Where does each intervention act?');a.legend(fontsize=8,frameon=False)
    fig.suptitle(f'What do the two structural representations control?   |   {example}, N={len(base)}, seed 2026',fontsize=16,y=.98)
    fig.text(.05,.035,'Native-structure reconstruction, not sequence generation. Fixed example chosen by ID before decoding.\nFine cannot move block centers by design; this figure does not establish learned topology or physical validity of edits.',fontsize=10,color='#475569');fig.subplots_adjust(top=.90,bottom=.13)
    for extension in ['png','pdf','svg']:fig.savefig(FIG/f'representation_interventions_en_v1.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.8,5.5));cmap=ListedColormap(['#2463a5','#d97706','#1b9770','#a148a2'])
    for a,seed in zip(axs,SEEDS):
        q=pca_data[seed];stride=max(1,len(q['x'])//7000);xx=q['x'][::stride];ss=q['slot'][::stride]
        a.scatter(xx[:,0],xx[:,1],c=ss,cmap=cmap,vmin=-.5,vmax=3.5,s=4,alpha=.30,rasterized=True);a.set_xlabel('Principal component 1');a.set_ylabel('Principal component 2');a.set_title(f'Seed {seed} | train code-slot MI {q["mi"]:.3f} bits')
        for slot in range(4):a.scatter([],[],color=cmap(slot),label=f'Slot {slot+1}',s=18)
        a.legend(frameon=False,fontsize=8);a.text(.02,.02,f'PC1+PC2: {q["explained"]:.1%} of train-frequency-weighted code variance',transform=a.transAxes,fontsize=8)
    fig.suptitle('Fine tokens also encode position within each 4-residue block',fontsize=15)
    fig.text(.05,.035,'PCA fitted to the frozen fine codebook with TRAIN usage weights; points are native tokens from 101 reused-dev proteins.\nColors denote residue index modulo 4, not secondary structure. Projection is descriptive; clustering alone does not establish function.',fontsize=9,color='#475569');fig.subplots_adjust(top=.82,bottom=.21,wspace=.28)
    for extension in ['png','pdf','svg']:fig.savefig(FIG/f'fine_codebook_position_pca_en_v1.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    for name,digest in manifest['source_sha256'].items():
        if sha(name)!=digest:raise RuntimeError('Input changed during visualization: '+name)
    print(json.dumps(dict(status='complete',summary=summary,figures=['paper/figures/representation_interventions_en_v1.png','paper/figures/fine_codebook_position_pca_en_v1.png']),indent=2))

if __name__=='__main__':main()
