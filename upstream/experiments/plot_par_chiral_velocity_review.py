"""Presentation-only plot of the independently checked frozen report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha


def main():
    root=Path('artifacts/par_chiral_velocity_v1')
    report=json.loads((root/'report.json').read_text())
    review=json.loads((root/'review.json').read_text())
    if not review['directional_consistency_passed']or review['report_sha256']!=sha(root/'report.json'):
        raise RuntimeError('Independently reviewed unchanged report required')
    arms=['native','polar','axial'];seeds=['2029','2030']
    values=[[report['effects'][seed]['scores'][arm]['macro_proper_RMSD_A']for arm in arms]for seed in seeds]
    maximum=max(max(row)for row in values)
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.6),sharey=True)
    colors=['#777777','#4c78a8','#e1812c']
    for ax,seed,row in zip(axes,seeds,values):
        ax.bar(np.arange(3),row,color=colors,width=.68)
        ax.set_ylim(0,maximum*1.23)
        for index,value in enumerate(row):
            ax.text(index,value+.075,f'{value:.3f}',ha='center',va='bottom',fontsize=11)
        ax.set_xticks(np.arange(3),['Native\ncontrol','Polar\ncontrol','Cross-product\nresidual'])
        ax.set_title(f'Training-draw seed {seed}',pad=12)
        ax.axhline(2,color='#555555',ls=':',lw=1)
        improvement=review['effects'][seed]['axial_relative_improvement_vs_native_percent']
        ax.text(.04,.94,f'{improvement:.2f}% lower error vs native',transform=ax.transAxes,va='top',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Proper-rotation coarse RMSD (Å)')
    fig.suptitle('Cross-product residual: positive two-target learnability result',fontsize=13,y=.98)
    fig.text(.5,.87,'Two reused TRAIN structures · 64 samples per protein · common parameter/Adam warm start',
             ha='center',fontsize=9,color='#444444')
    fig.text(.5,.025,'Both seeds beat both controls. Per-protein mean ≤ 2 Å target remains unmet; no generalization claim.',
             ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(top=.75,bottom=.21,left=.09,right=.98,wspace=.12)
    directory=Path('paper/figures/par_chiral_velocity_v1');directory.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for ext in ['pdf','png','svg']:
        path=directory/f'PAR_Chiral_Velocity_Reviewed.{ext}'
        fig.savefig(path,dpi=180);outputs.append(path)
    plt.close(fig)
    atomic_json(root/'presentation_review.json',dict(status='rendered_awaiting_visual_inspection',
                report_sha256=sha(root/'report.json'),plot_code_sha256=sha(Path(__file__)),
                note='Explicit common y limit includes every bar and label. Historical numeric sources and original figure preserved.',
                files={str(path):sha(path)for path in outputs}))
    print('Reviewed presentation plot rendered; numerical experiment untouched.')


if __name__=='__main__':main()
