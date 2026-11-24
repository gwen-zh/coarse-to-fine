"""Matched fine-tokenizer probes; both seeds, complete fixed-budget endpoints."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnostic_io import atomic_json
from hierarchical_pipeline_io import sha

OUT=Path('paper/figures/block_frame_probe_v1')


def main():
    roots=[Path('artifacts/block_frame_tokenizer_v1'),Path('artifacts/block_stable_frame_tokenizer_v1')];reports=[json.loads((r/'report.json').read_text()) for r in roots]
    fig,axes=plt.subplots(2,2,figsize=(10.5,7));labels=['Reference\nindependent','Local\nindependent','Reference\ntransport','Local\ntransport'];colors=['#A0A5A8','#286D93','#A0A5A8','#38836B']
    specs=[('PDB_development','ca_rmsd',1.,'A  PDB reconstruction','CA RMSD (Å)'),('PDB_development','cn_rate',100.,'B  PDB C–N geometry','C–N outliers (%)'),('MD_development','ca_rmsd',1.,'C  MD reconstruction','CA RMSD (Å)'),('MD_development','cn_rate',100.,'D  MD C–N geometry','C–N outliers (%)')]
    for ax,(group,key,scale,title,ylabel) in zip(axes.flat,specs):
        all_values=[]
        for family,report in enumerate(reports):
            for a,arm in enumerate(['reference','local']):
                i=2*family+a;ys=[scale*r['evaluation']['groups'][group]['metrics'][key] for r in report['arms'] if r['arm']==arm];all_values.extend(ys);ax.bar(i,np.mean(ys),color=colors[i],width=.65,alpha=.85);ax.scatter(i+np.linspace(-.06,.06,len(ys)),ys,color='black',s=17,zorder=3);ax.text(i,max(ys)+max(all_values)*.045,f'{np.mean(ys):.3f}' if scale==1 else f'{np.mean(ys):.2f}',ha='center',fontsize=9)
        if group=='PDB_development' and key=='cn_rate':ax.axhline(10.,color='.4',ls='--',lw=1,label='Readiness threshold');ax.legend(frameon=False,fontsize=8)
        ax.set_xticks(range(4),labels);ax.tick_params(axis='x',labelsize=8);ax.set(title=title,ylabel=ylabel,ylim=(0,max(all_values)*1.2));ax.axvline(1.5,color='.8',lw=1);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched fine-tokenizer capacity probes: coordinate frame matters',fontsize=14)
    fig.text(.5,.015,'Each arm: 193,604 trainable parameters, 8D fine latent, 1,024-entry EMA book; 1,024 warmup + 4,096 quantized updates.\nThree R1 MD frames + one PDB training protein per update; frozen coarse E/D. Dots are seeds, not confidence intervals.\nEvaluation: 32 reused PDB development proteins and 64 R2 frames. No R3 reuse or generation-quality claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.96));OUT.mkdir(parents=True,exist_ok=True)
    for suffix in ['pdf','png','svg']:fig.savefig(OUT/f'Block_Frame_Probe.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig);atomic_json(Path('artifacts/block_stable_frame_tokenizer_v1/plot_manifest.json'),dict(figure=str(OUT/'Block_Frame_Probe.pdf'),source_sha256=sha(__file__),report_sha256={str(r/'report.json'):sha(r/'report.json') for r in roots}));print('BLOCK_FRAME_FIGURE',OUT/'Block_Frame_Probe.pdf')


if __name__=='__main__':main()
