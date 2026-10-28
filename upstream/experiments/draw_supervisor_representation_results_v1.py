"""Supervisor-ready English figures from frozen descriptive representation data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

ROOT=Path('artifacts/representation_motif_atlas_v1')
OUT=Path('paper/figures/supervisor_representation_v1')
BLUE='#235e9b';ORANGE='#c7751b';PURPLE='#8850aa';GRAY='#9ba4b0';INK='#172b42';GREEN='#26826a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titleweight':'bold','axes.titlesize':13,'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#bac4ce'})


def title(fig,main,subtitle):
    fig.text(.045,.965,main,fontsize=23,weight='bold',va='top',color=INK)
    fig.text(.046,.917,subtitle,fontsize=11,color='#586b7c',va='top')


def footer(fig,text):
    fig.text(.046,.035,text,fontsize=9,color='#586b7c',va='bottom',linespacing=1.45)


def style3d(ax,coords,elev=22,azim=-55,scale=1.08):
    coords=np.asarray(coords).reshape(-1,3);lo=coords.min(0);hi=coords.max(0);mid=(hi+lo)/2;r=max(hi-lo)*.5*scale
    for setter,m in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],mid):setter(m-r,m+r)
    ax.set_box_aspect((1,1,1));ax.view_init(elev,azim);ax.set_axis_off()


def ca_pool(bb):
    return np.stack([bb[i:i+4,1].mean(0) for i in range(0,len(bb),4)])


def save(fig,name,pages=None):
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT/f'{name}.{extension}',dpi=200,bbox_inches=None)
    if pages is not None:pages.savefig(fig)
    plt.close(fig)


def overview(old,pages):
    with np.load('artifacts/representation_visualization_v1/example_seed2026.npz') as z:d={k:z[k].copy() for k in z.files}
    base=d['baseline'][:,1];native=d['reference'][:,1];centers=d['coarse_centers']
    fig=plt.figure(figsize=(16,9));title(fig,'What do coarse and fine representations control?','Current hybrid structural tokenizer | Native encode/decode | 101 reused-development proteins, two training seeds')
    grid=fig.add_gridspec(2,3,left=.05,right=.96,top=.85,bottom=.15,wspace=.36,hspace=.55,height_ratios=[1.15,1])
    ax=fig.add_subplot(grid[0,0],projection='3d');ax.plot(*native.T,color=GRAY,lw=2.5,alpha=.8);ax.plot(*base.T,color=BLUE,lw=1.2);ax.plot(*centers.T,color=ORANGE,lw=1);ax.scatter(*centers.T,s=12,color=ORANGE)
    style3d(ax,native);ax.set_box_aspect((1,1,1),zoom=1.15);ax.set_title('A  Decoded coarse layout',loc='left',pad=8)
    ax.legend(handles=[Line2D([],[],color=GRAY,lw=2,label='Reference CA'),Line2D([],[],color=BLUE,label='Reconstructed CA'),Line2D([],[],color=ORANGE,marker='o',ms=4,label='Block centers')],loc='lower left',fontsize=8,frameon=False)
    ax.text2D(.65,.08,'109L_1\n162 residues',transform=ax.transAxes,fontsize=9,color='#586b7c')
    ax=fig.add_subplot(grid[0,1]);truth=ca_pool(d['reference']);ref=np.linalg.norm(truth[:,None]-truth[None,:],axis=-1);pred=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1);image=np.where(np.triu(np.ones_like(ref),1)>0,ref,pred)
    im=ax.imshow(image,cmap='viridis',origin='lower');fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Distance (Å)');ax.set_title('B  Block-center distance map',loc='left');ax.set_xlabel('Block index | upper: reference');ax.set_ylabel('Block index | lower: decoded')
    ax=fig.add_subplot(grid[0,2]);ax.axis('off');ax.set_title('C  Structural decomposition',loc='left')
    for y,color,heading,body in [(.80,ORANGE,'COARSE  ·  continuous z_c','One 16-D vector per 4-residue block\nDecoded into block-center positions'),(.39,PURPLE,'FINE  ·  1,024 discrete codes','One token per residue\nDecoded with adjacent-block context')]:
        ax.add_patch(FancyBboxPatch((0,y-.17),.99,.31,boxstyle='round,pad=.015',transform=ax.transAxes,facecolor=color,alpha=.08,edgecolor='none'))
        ax.text(.04,y+.04,heading,transform=ax.transAxes,fontsize=12,weight='bold',color=color);ax.text(.04,y-.07,body,transform=ax.transAxes,fontsize=10,linespacing=1.6,va='top')
    ax.text(.02,.03,'Fine offsets have zero block CA centroid.',transform=ax.transAxes,fontsize=10,weight='bold')
    ax=fig.add_subplot(grid[1,0]);vals=[s['baseline_ca_rmsd'] for s in old['summary']];bars=ax.bar([0,1],vals,color=[BLUE,'#5893bd'],width=.5);ax.set_xticks([0,1],['Seed 2026','Seed 2027']);ax.set_ylim(0,.5);ax.set_ylabel('CA reconstruction RMSD (Å)');ax.set_title('D  Representation reconstructs well',loc='left')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.012,f'{v:.3f}',ha='center',weight='bold')
    ax.text(.02,.94,'Protein mean, n = 101',transform=ax.transAxes,fontsize=9,color='#586b7c')
    ax=fig.add_subplot(grid[1,1]);x=np.arange(len(base))+1;ax.plot(x,d['coarse_displacement'],color=ORANGE,label='Change one coarse block');ax.plot(x,d['fine_displacement'],color=PURPLE,label='Change one fine token');i=int(d['fine_index']);start=max(1,(i//4-1)*4+1);end=min(len(base),(i//4+2)*4);ax.axvspan(start,end,color=PURPLE,alpha=.08);ax.set_xlabel('Residue index');ax.set_ylabel('CA displacement (Å)');ax.set_title('E  One-level-at-a-time intervention',loc='left');ax.legend(frameon=False,fontsize=8,loc='upper right');ax.text(.02,.025,'Same frame; different perturbation sizes',transform=ax.transAxes,fontsize=8,color='#586b7c',va='bottom')
    ax=fig.add_subplot(grid[1,2]);ax.axis('off');ax.set_title('F  What this establishes',loc='left')
    text=[('Observed','Fine edit affects its block and neighbors.'),('By construction','Fine cannot shift block-center positions.'),('Still to test','Which local shapes do individual codes encode?')]
    for y,(h,b) in zip([.78,.48,.18],text):ax.text(.02,y,h,transform=ax.transAxes,weight='bold',fontsize=11,color=BLUE if h!='By construction' else ORANGE);ax.text(.02,y-.075,b,transform=ax.transAxes,fontsize=10,va='top',wrap=True)
    footer(fig,'Architecture check and reconstruction diagnostic. This does not establish learned global topology or sequence-generation quality.\nExample fixed by protein ID before decoding. Interventions use different magnitudes; their sizes do not measure relative representation importance.')
    save(fig,'01_coarse_fine_roles_en',pages)


def atlas(report,pages,rank=0):
    fig=plt.figure(figsize=(16,9));title(fig,'What native backbone fragments share the same fine token?','Block position controlled | Central N/CA/C aligned | Codes chosen by TRAIN frequency; examples taken from DEV')
    grid=fig.add_gridspec(2,4,left=.045,right=.975,top=.825,bottom=.19,wspace=.10,hspace=.29)
    all_points=[]
    for seed_result in report['seeds']:
        with np.load(ROOT/f'descriptors_seed{seed_result["seed"]}.npz') as z:
            for selected in seed_result['atlas_selection']:
                if selected['rank']!=rank:continue
                all_points.append(z['train_pair_local'][4*selected['code']+selected['slot_zero_based']].reshape(-1,3))
                all_points.append(z['train_slot_local'][selected['slot_zero_based']].reshape(-1,3))
                for example in selected['examples']:all_points.append(z['dev_local'][example['descriptor_index']].reshape(-1,3))
    common_limit=max(4.,float(np.abs(np.concatenate(all_points)).max())*1.05)
    for row,s in enumerate(report['seeds']):
        with np.load(ROOT/f'descriptors_seed{s["seed"]}.npz') as z:d={k:z[k] for k in z.files}
        selections=[r for r in s['atlas_selection'] if r['rank']==rank]
        for col,rec in enumerate(selections):
            ax=fig.add_subplot(grid[row,col],projection='3d');pair=4*rec['code']+rec['slot_zero_based'];template=d['train_pair_local'][pair].reshape(9,3);slot_template=d['train_slot_local'][rec['slot_zero_based']].reshape(9,3);fragments=[]
            for ex in rec['examples']:
                xyz=d['dev_local'][ex['descriptor_index']].reshape(9,3);fragments.append(xyz)
                ax.plot(*xyz.T,color=ORANGE,alpha=.30,lw=.9)
            ax.plot(*slot_template.T,color=GRAY,lw=1.2,ls='--',alpha=.85);ax.plot(*template.T,color=BLUE,lw=2.5);ax.scatter(*template[[1,4,7]].T,color=BLUE,s=15)
            points=np.concatenate(fragments+[template,slot_template]);style3d(ax,points,elev=28,azim=-52,scale=1.03)
            ax.set_xlim(-common_limit,common_limit);ax.set_ylim(-common_limit,common_limit);ax.set_zlim(-common_limit,common_limit);ax.set_box_aspect((1,1,1),zoom=1.24)
            ax.set_title(f'Slot {col+1}  ·  token {rec["code"]}',fontsize=12,pad=5)
            ax.text2D(.04,.02,f'Train: {rec["train_occurrences"]} fragments / {rec["train_proteins"]} proteins\nDev: {rec["dev_occurrences"]} fragments / {rec["dev_proteins"]} proteins',transform=ax.transAxes,fontsize=8,color='#586b7c',va='bottom')
        fig.text(.014,.67 if row==0 else .32,f'SEED {s["seed"]}',rotation=90,fontsize=11,weight='bold',color=BLUE,va='center')
    handles=[Line2D([],[],color=BLUE,lw=2.5,label='Code + slot mean descriptor (train)'),Line2D([],[],color=GRAY,lw=1.5,ls='--',label='Slot-only mean descriptor (train)'),Line2D([],[],color=ORANGE,lw=1.4,alpha=.6,label='Native dev examples; at most 12 proteins per code')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=3,frameon=False,fontsize=9)
    footer(fig,'Each fragment contains three residues (N/CA/C atoms). Central residue sets the frame; neighbor geometry is free to vary.\nMeans are statistical descriptors, not decoded physical structures. Examples are selected by protein ID, not geometric similarity; broad variation remains.')
    save(fig,'02_fine_token_fragment_atlas_en' if rank==0 else 'S1_fine_token_fragment_atlas_second_code_en',pages)


def quant(report,pages):
    fig=plt.figure(figsize=(16,9));title(fig,'Fine tokens: strong position signal, modest gains in a shape probe','TRAIN-only mean descriptors | 101 reused-dev proteins / 15,769 interior residues | Equal protein weighting')
    grid=fig.add_gridspec(2,2,left=.07,right=.96,top=.84,bottom=.19,wspace=.30,hspace=.56)
    ax=fig.add_subplot(grid[0,0]);values=[s['dev_slot_prediction_accuracy']*100 for s in report['seeds']];bars=ax.bar([0,1],values,color=[BLUE,'#5893bd'],width=.5)
    ax.set_xticks([0,1],['Seed 2026','Seed 2027']);ax.axhline(25,color=GRAY,ls='--',lw=1);ax.text(1.35,26,'25% reference',fontsize=9,color='#586b7c',ha='right');ax.set_ylim(0,110);ax.set_xlim(-.5,1.5);ax.set_ylabel('Block-slot prediction accuracy (%)');ax.set_title('A  Token ID reveals position within a block',loc='left')
    for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}%',ha='center',weight='bold')
    ax.text(.02,.98,'Prediction: majority slot for each code in TRAIN',transform=ax.transAxes,fontsize=9,va='top',color='#586b7c')
    ax=fig.add_subplot(grid[0,1]);methods=['slot_only','code_and_slot','shuffled_code_within_slot'];labels=['Slot-only mean','Code + slot mean','Shuffled code'];colors=[GRAY,BLUE,ORANGE];x=np.arange(2);width=.23
    for offset,method,label,color in zip([-1,0,1],methods,labels,colors):
        vals=[s['methods'][method]['local_shape_rmsd_A'] for s in report['seeds']];bars=ax.bar(x+offset*width,vals,width,color=color,label=label)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.014,f'{v:.3f}',fontsize=9,ha='center',rotation=0)
    ax.set_xticks(x,['Seed 2026','Seed 2027']);ax.set_ylim(0,1.52);ax.set_ylabel('Neighbor-atom shape error (Å)');ax.set_title('B  Information beyond block position',loc='left');ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.04))
    ax=fig.add_subplot(grid[1,0]);matrix=[]
    for s in report['seeds']:
        slot=np.array(s['methods']['slot_only']['per_slot'])[:,0];code=np.array(s['methods']['code_and_slot']['per_slot'])[:,0];matrix.append(100*(slot-code)/slot)
    matrix=np.array(matrix);im=ax.imshow(matrix,cmap='RdBu',vmin=-16,vmax=16,aspect='auto');ax.set_xticks(range(4),['Slot 1','Slot 2','Slot 3','Slot 4']);ax.set_yticks(range(2),['Seed 2026','Seed 2027']);ax.set_title('C  Mean-template gain varies across block positions',loc='left')
    for i in range(2):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.2f}%',ha='center',va='center',fontsize=13,color='white' if abs(matrix[i,j])>11 else INK,weight='bold')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Error reduction vs slot-only (%)')
    ax=fig.add_subplot(grid[1,1]);width=.3;x=np.arange(2);colors=[BLUE,PURPLE]
    for offset,key,label,color in zip([-.5,.5],['phi_circular_mae_deg','psi_circular_mae_deg'],['Phi','Psi'],colors):
        means=[];lower=[];upper=[]
        for s in report['seeds']:
            q=s['code_vs_slot_effect'][key];means.append(q['code_minus_slot']);lower.append(q['code_minus_slot']-q['descriptive_paired_bootstrap95'][0]);upper.append(q['descriptive_paired_bootstrap95'][1]-q['code_minus_slot'])
        bars=ax.bar(x+offset*width,means,width,color=color,label=label);ax.errorbar(x+offset*width,means,yerr=[lower,upper],fmt='none',ecolor=INK,capsize=3,lw=1)
        for b,v in zip(bars,means):ax.text(b.get_x()+b.get_width()/2,v-.3,f'{v:+.2f}°',ha='center',va='top',fontsize=10)
    ax.axhline(0,color=GRAY,lw=1);ax.set_xticks(x,['Seed 2026','Seed 2027']);ax.set_ylabel('Code + slot minus slot-only error (degrees)');ax.set_ylim(-10.3,1.7);ax.set_title('D  Backbone angles: small, seed-dependent gains',loc='left');ax.legend(frameon=False,ncol=2,loc='lower right',fontsize=9)
    footer(fig,'Shape probe predicts six neighboring atoms in the central-residue frame; lower error is better. No neural model is trained.\nAngle error bars: descriptive 95% paired bootstrap intervals over reused-dev proteins. This probe omits neighboring-token context and does not measure full representation capacity.')
    save(fig,'03_position_controlled_fine_content_en',pages)


def notes(report,old):
    lines=['Representation results for supervisor — 2026-10-28','===================================================','','Scope: current independently trained hybrid structural tokenizer. These are native-structure representation diagnostics, not sequence-generation improvements. The 101 development proteins have been reused; this is not independent confirmation.','','Suggested presentation order:','1. 01_coarse_fine_roles_en: visualize decoded block centers and one-level interventions.','2. 02_fine_token_fragment_atlas_en: same code, same block slot, native fragments aligned on central N/CA/C. Broad variation is shown rather than selected away.','3. 03_position_controlled_fine_content_en: test single-token local-shape information beyond block position.','','What can be said:','- The coarse decoder reconstructs block-center layout from native coarse latents.','- Fine changes preserve block centers by construction, and act through local decoder context.','- Code IDs predict block position very accurately.','- Position-controlled single-token local-shape gains are modest and differ strongly by seed.','','What cannot be said:','- That coarse has learned global topology from sequence.','- That a fine code denotes a unique secondary-structure class.','- That weak single-token prediction proves the whole fine representation is uninformative.','- That these diagnostics improve sequence-generation accuracy.','','Numerical results (protein means on reused DEV):']
    for s in report['seeds']:
        slot=s['methods']['slot_only'];code=s['methods']['code_and_slot'];null=s['methods']['shuffled_code_within_slot'];reduction=100*(slot['local_shape_rmsd_A']-code['local_shape_rmsd_A'])/slot['local_shape_rmsd_A']
        lines += [f'Seed {s["seed"]}:',f'  Code->slot accuracy: {s["dev_slot_prediction_accuracy"]:.3%}.',f'  Local-shape error: slot-only {slot["local_shape_rmsd_A"]:.6f} A; code+slot {code["local_shape_rmsd_A"]:.6f} A; shuffled {null["local_shape_rmsd_A"]:.6f} A. Reduction {reduction:.3f}%.',f'  Phi MAE: {slot["phi_circular_mae_deg"]:.3f} -> {code["phi_circular_mae_deg"]:.3f} degrees.',f'  Psi MAE: {slot["psi_circular_mae_deg"]:.3f} -> {code["psi_circular_mae_deg"]:.3f} degrees.']
    lines += ['','Spoken English:','I first visualized the decoded block centers and changed one level at a time. This confirms the geometric separation we built into the decoder. I then looked at what individual fine codes contain after controlling for position within each block. The codes carry strong positional information, while a single-code mean-template probe gives modest and seed-dependent local-shape gains. So I would not yet describe the fine codes as clean structural categories. Neighboring-token context may still be important.','','中文讲解：','先展示粗层中心和细层局部排布，再强调 fine 保持中心不动是架构性质。接着展示同 token 的真实局部片段，最后用位置控制的量化结果说明：单 token 的局部形状信息有限且 seed 差异明显。这里不能推出整套 fine 表示没有信息，因为解码还依赖相邻 token 和 coarse 条件。','','Methods:','- Templates and code selections fitted only on 899 TRAIN proteins / 136,345 interior residues.','- Evaluate every interior residue of 101 reused-DEV proteins / 15,769 residues; exclude only chain terminals by definition.','- Same-slot shuffled-code TRAIN control preserves all code/slot counts.','- Atlas picks two most frequent eligible TRAIN codes per slot; primary figure uses rank1, supplementary uses rank2. Eligibility >=20 occurrences across >=5 TRAIN proteins.','- At most12 DEV examples per code, first occurrence in each distinct protein, ordered by protein ID.','- All figures retain the same physical spatial scale across atlas panels.','- Five targeted geometry/template tests pass, including independent MDTraj dihedral parity and rigid-motion invariance.','','Data: artifacts/representation_motif_atlas_v1/results.json','Analysis protocol: artifacts/representation_motif_atlas_v1/protocol.json']
    Path('artifacts/reports/representation_results_for_supervisor_2026-10-28.rst').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True);report=json.loads((ROOT/'results.json').read_text());old=json.loads(Path('artifacts/representation_visualization_v1/results.json').read_text())
    with PdfPages(OUT/'Representation_Results_for_Supervisor.pdf') as pages:
        overview(old,pages);atlas(report,pages,rank=0);quant(report,pages)
    atlas(report,None,rank=1);notes(report,old)
    inputs=[__file__,str(ROOT/'results.json'),str(ROOT/'atlas_selection.json'),'artifacts/representation_visualization_v1/results.json','artifacts/representation_visualization_v1/example_seed2026.npz']+[str(ROOT/f'descriptors_seed{s["seed"]}.npz') for s in report['seeds']]
    manifest=dict(status='complete',main_pdf=str(OUT/'Representation_Results_for_Supervisor.pdf'),pages=3,language='English',format=['png','pdf','svg'],source_sha256={p:sha(p) for p in inputs},figures={str(p):sha(p) for p in OUT.iterdir() if p.suffix in ['.pdf','.png','.svg']})
    atomic_json(OUT/'manifest.json',manifest)
    print(json.dumps({'status':'complete','main_pdf':manifest['main_pdf'],'figures':list(manifest['figures'])},indent=2))

if __name__=='__main__':main()
