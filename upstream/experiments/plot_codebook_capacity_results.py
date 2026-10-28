"""Read-only plots of the complete codebook-capacity comparison."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=Path('artifacts/codebook_capacity_v1')
source=run/'comparison.json'
r=json.loads(source.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','font.size':10})
fig,axs=plt.subplots(2,2,figsize=(12,8.2))
colors={2026:'#286BBC',2027:'#B87B24'}
settings=[
 ('Native reconstruction', 'CA RMSD (Å)', lambda a:a['reconstruction']['metrics']['ca_rmsd']),
 ('Predicted fine + native coarse (oracle)', 'CA RMSD (Å)', lambda a:a['fine']['native_metrics']['ca_rmsd']),
 ('Predicted fine + fixed generated coarse', 'CA RMSD (Å)', lambda a:a['fine']['generated_metrics']['ca_rmsd']),
 ('Predicted fine + fixed generated coarse', 'C–N outliers (%)', lambda a:100*a['fine']['generated_metrics']['cn_rate']),
]
for ax,(title,ylabel,getter) in zip(axs.flat,settings):
 for seed in [2026,2027]:
  rows=sorted([a for a in r['rows'] if a['seed']==seed],key=lambda a:a['k'])
  ax.plot([0,1,2],[getter(a) for a in rows],'-o',color=colors[seed],label=f'Seed {seed}',lw=1.8,ms=5)
 ax.set_title(title,fontsize=11.5,fontweight='bold');ax.set_ylabel(ylabel);ax.set_xlabel('Fine vocabulary size K')
 ax.set_xticks([0,1,2],['1024','2048','4096']);ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False)
 ax.legend(frameon=False,fontsize=9)
fig.suptitle('Does a larger fine codebook improve generation?',fontsize=16,fontweight='bold',y=.98)
fig.subplots_adjust(left=.085,right=.98,top=.91,bottom=.13,hspace=.43,wspace=.30)
fig.text(.085,.065,'Fresh train-only codebook refits; same frozen encoder/decoder; same cached generated coarse; fixed 12-epoch fine training.',fontsize=9,color='#536578')
fig.text(.085,.035,'Development reused. Native coarse is an oracle diagnostic. Different K changes fine bitrate, parameter count and compute.',fontsize=9,color='#536578')
for ext in ['png','pdf','svg']:fig.savefig(run/f'capacity_results.{ext}',dpi=180,facecolor='white')
plt.close(fig)
(run/'plot_manifest.json').write_text(json.dumps(dict(comparison_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),drawing_script=str(Path(__file__)),scope='Read-only final development comparison; no model selection or inference.'),indent=2)+'\n')
print('CAPACITY_PLOT_COMPLETE')
