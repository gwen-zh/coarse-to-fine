"""Display both seeds and the matched coordinate-only control; reused dev data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hierarchical_pipeline_io import sha
from diagnostic_io import atomic_json

RUN=Path('artifacts/atlas_decoder_repair_v1')
OUT=Path('paper/figures/atlas_decoder_repair_v1')


def main():
    result=json.loads((RUN/'report.json').read_text());baseline=result['baseline']
    specifications=[('R2_dense','boundary_cn_rate',100.,'A  MD development: cross-block C–N','Outliers (%)'),('R2_dense','ca_rmsd',1.,'B  MD development: structure reconstruction','CA RMSD (Å)'),('pdb_validation','cn_rate',100.,'C  PDB development: C–N links','Outliers (%)'),('pdb_validation','ca_rmsd',1.,'D  PDB development: structure reconstruction','CA RMSD (Å)')]
    colors=['#9DA3A6','#B5722C','#286C8C'];fig,axes=plt.subplots(2,2,figsize=(10,6.7))
    for ax,(group,metric,scale,title,ylabel) in zip(axes.flat,specifications):
        values=[[scale*baseline[group]['metrics'][metric]]]
        for arm in ['coordinate','geometry']:values.append([scale*r['groups'][group]['metrics'][metric] for r in result['arms'] if r['arm']==arm])
        for i,ys in enumerate(values):
            ax.bar(i,np.mean(ys),color=colors[i],width=.62,alpha=.8)
            ax.scatter(i+np.linspace(-.07,.07,len(ys)),ys,color='black',s=18,zorder=3)
            ax.text(i,max(ys)+max(max(v) for v in values)*.035,f'{np.mean(ys):.3f}' if scale==1 else f'{np.mean(ys):.2f}',ha='center',fontsize=10)
        ax.set_xticks([0,1,2],['Before repair','Coordinate only','Coordinate + geometry']);ax.tick_params(axis='x',labelsize=8);ax.set_ylabel(ylabel);ax.set_title(title,fontsize=11);ax.set_ylim(0,max(max(v) for v in values)*1.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Own fine-decoder repair with frozen encoders and codebook',fontsize=15)
    fig.text(.5,.015,'Train: ATLAS 1b2s_E R1 (1,001 frames) + 32 PDB replay proteins; 2,048 updates per arm/seed.\nDevelopment: R2 (64 correlated frames) and 16 reused PDB proteins. Dots show both seeds, not confidence intervals.\nOne-protein reconstruction study; R3 remains sealed. No sequence-generation or independent-test claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.96));OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','png','svg']:fig.savefig(OUT/f'Decoder_Repair.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig);atomic_json(RUN/'plot_manifest.json',dict(report_sha256=sha(RUN/'report.json'),source_sha256=sha(__file__),figure=str(OUT/'Decoder_Repair.pdf'),scope='All fixed final checkpoints; reused development only'))
    print('DECODER_REPAIR_FIGURE',OUT/'Decoder_Repair.pdf')


if __name__=='__main__':main()
