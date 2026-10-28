"""Read-only final matched-cohort comparison; repeated development, not confirmation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=Path('artifacts/coarse_generalization_v1');source=run/'comparison.json';r=json.loads(source.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
fig,axs=plt.subplots(1,3,figsize=(14.5,4.9));colors={2026:'#286BBC',2027:'#B87924'}
for seed in [2026,2027]:
 b=r['baseline'][str(seed)];rows={x['arm']:x for x in r['rows'] if x['seed']==seed}
 series=[[b['metrics']['ca_rmsd']]+[rows[a]['full_generation_metrics']['ca_rmsd'] for a in ['flow','regression']],
         [100*b['metrics']['cn_rate']]+[100*rows[a]['full_generation_metrics']['cn_rate'] for a in ['flow','regression']],
         [b['spread']]+[rows[a]['full_spread'] for a in ['flow','regression']]]
 for ax,values in zip(axs,series):
  ax.plot([0,1,2],values,'-o',lw=1.8,ms=5,color=colors[seed],label=f'Seed {seed}')
  for x,y in enumerate(values):ax.annotate(f'{y:.3f}',(x,y),xytext=(4,-14) if seed==2026 else (4,8),textcoords='offset points',ha='left' if x==0 else ('right' if x==2 else 'center'),fontsize=8.5,color=colors[seed])
for ax,title,ylabel in zip(axs,['Reference accuracy','C–N connectivity','Sample spread'],['CA RMSD (Å)','C–N outliers (%)','Mean pairwise CA RMSD (Å)']):
 ax.set_title(title,fontweight='bold');ax.set_ylabel(ylabel);ax.set_xticks([0,1,2],['Own baseline','Flow +12 epochs','Point regression']);ax.tick_params(axis='x',labelsize=9)
 ax.set_xlim(-.28,2.30);ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False);ax.margins(y=.15);ax.legend(frameon=False,fontsize=9)
axs[2].set_ylim(0,10.5)
fig.suptitle('Full-corpus coarse training: no consistent overall generation gain',fontweight='bold',fontsize=15,y=.98)
fig.subplots_adjust(left=.065,right=.985,top=.85,bottom=.24,wspace=.29)
fig.text(.065,.12,'899 training proteins; 16 reused development proteins × 8 outputs / seed; structural model and original fine weights fixed.',fontsize=9,color='#536578')
fig.text(.065,.06,'Regression has deterministic coarse centers. Fine-only sample spread does not establish global or thermodynamic ensemble coverage.',fontsize=9,color='#536578')
for ext in ['png','pdf','svg']:fig.savefig(run/f'final_comparison.{ext}',dpi=180,facecolor='white')
plt.close(fig)
(run/'plot_manifest.json').write_text(json.dumps(dict(comparison_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),drawing_script=str(Path(__file__))),indent=2)+'\n')
print('COARSE_GENERALIZATION_PLOT_COMPLETE')
