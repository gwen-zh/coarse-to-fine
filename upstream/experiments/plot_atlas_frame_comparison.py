"""Display all samples and both fixed-budget seeds of the frame comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_atlas_decoder_repair import load_cache
from analyze_atlas_flow_geometry import align_frames
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

OLD=Path('artifacts/atlas_ensemble_flow_v1')
NEW=Path('artifacts/atlas_shared_frame_flow_v1')
OUT=Path('paper/figures/atlas_frame_comparison_v1')


def main():
    old=json.loads((OLD/'report.json').read_text());new=json.loads((NEW/'report.json').read_text());comparison=json.loads((NEW/'matched_comparison.json').read_text());groups=load_cache()
    train=np.stack([r['target'][:,1].numpy() for r in groups['R1_train']]);reference=np.stack([r['target'][:,1].numpy() for r in groups['R2_dense']]);aligned=align_frames(train,train[0]);mean=aligned.mean(0);matrix=(aligned-mean).reshape(len(aligned),-1);_,s,components=np.linalg.svd(matrix,full_matrices=False);components=components[:2]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.5));pca=axes[0,0];train_projection=matrix@components.T;dev=(align_frames(reference,train[0])-mean).reshape(len(reference),-1)@components.T
    pca.scatter(train_projection[:,0],train_projection[:,1],s=6,color='.8',alpha=.45,label='R1 train');pca.scatter(dev[:,0],dev[:,1],s=15,color='.2',marker='x',label='R2 development')
    for root,color,label in [(OLD,'#999999','Original frame'),(NEW,'#1F719B','Shared frame')]:
        projected=[]
        for seed in [2026,2027]:
            with np.load(root/f'seed{seed}/full_flow_repaired_decoder.npz',allow_pickle=False) as z:ca=z['backbone'][:,:,1].copy()
            projected.extend((align_frames(ca,train[0])-mean).reshape(len(ca),-1)@components.T)
        projected=np.array(projected);pca.scatter(projected[:,0],projected[:,1],s=19,color=color,alpha=.65,label=label)
    pca.set(title='A  All samples in TRAIN-fitted CA projections',xlabel=f'PC1 ({100*s[0]**2/(s*s).sum():.1f}% TRAIN variance)',ylabel=f'PC2 ({100*s[1]**2/(s*s).sum():.1f}% TRAIN variance)');pca.legend(fontsize=8,frameon=False)
    categories=[('Original frame',old,'full_flow_repaired_decoder','#999999'),('Shared frame',new,'full_flow_repaired_decoder','#1F719B'),('True coarse*',new,'oracle_coarse_conditional_fine_repaired_decoder','#8A689F'),('R1 retrieval*',new,'empirical_R1_repaired_decoder','#4B8A6A')]
    native_spread=next(r for r in new['seeds'][0]['arms'] if r['arm']=='full_flow_repaired_decoder')['reference_mean_pairwise_ca_rmsd_A']
    for ax,key,scale,title,ylabel in [(axes[0,1],'mean_sample_nearest_reference_ca_rmsd_A',1.,'B  Proximity to R2 structures','Mean nearest-reference CA RMSD (Å)'),(axes[1,0],'cn_rate',100.,'C  Generated C–N connections','Outliers (%)'),(axes[1,1],'mean_pairwise_ca_rmsd_A',1.,'D  Ensemble spread','Mean pairwise CA RMSD (Å)')]:
        values=[]
        for i,(label,report,arm,color) in enumerate(categories):
            rows=[next(r for r in seed['arms'] if r['arm']==arm) for seed in report['seeds']]
            ys=[scale*(r['geometry'][key] if key=='cn_rate' else r[key]) for r in rows];values.extend(ys)
            ax.bar(i,np.mean(ys),color=color,alpha=.8,width=.62);ax.scatter(i+np.linspace(-.06,.06,len(ys)),ys,color='black',s=15,zorder=3)
            ax.text(i,max(ys)+max(values)*.04,f'{np.mean(ys):.2f}',ha='center',fontsize=9)
        if key=='mean_pairwise_ca_rmsd_A':ax.axhline(native_spread,color='.25',ls='--',lw=1,label='R2 reference');ax.legend(frameon=False,fontsize=8)
        ax.set_xticks(range(4),[c[0] for c in categories]);ax.tick_params(axis='x',labelsize=8);ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.set_ylim(0,max(max(values),native_spread if key=='mean_pairwise_ca_rmsd_A' else 0)*1.2)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same generator and training budget; different target coordinate frames',fontsize=14)
    fig.text(.5,.015,'ATLAS 1b2s_E only: R1 training, reused R2 development; two seeds, 2,048 updates each, 64 raw samples per seed.\n* True-coarse and training-frame retrieval are diagnostic controls. Dots are seeds, not confidence intervals.\nR3 was excluded from development and evaluated separately after model freeze. No unseen-protein claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.96));OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','png','svg']:fig.savefig(OUT/f'Frame_Comparison.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig);atomic_json(NEW/'plot_manifest.json',dict(source_sha256=sha(__file__),comparison_sha256=sha(NEW/'matched_comparison.json'),figure=str(OUT/'Frame_Comparison.pdf'),all_samples_included=True))
    print('FRAME_COMPARISON_FIGURE',OUT/'Frame_Comparison.pdf')


if __name__=='__main__':main()
