"""Read-only small-set fit/generalization curves, not a quality confirmation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run=Path('artifacts/coarse_smallset_probe_v1');steps=[0,128,512,1024]
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'font.size':10})
fig,axes=plt.subplots(1,2,figsize=(12,5.2));colors={2026:'#286BBC',2027:'#B87924'}
for seed in [2026,2027]:
 for arm in ['flow','regression']:
  rows=[json.loads((run/f'seed{seed}'/arm/f'step{s}'/'evaluation.json').read_text()) for s in steps]
  for ax,subset in zip(axes,['train','validation']):
   ax.plot(steps,[r['groups'][subset]['true_center_rmsd'] for r in rows],color=colors[seed],
           linestyle='-' if arm=='flow' else '--',marker='o' if arm=='flow' else 's',label=f'{arm}, seed {seed}',lw=1.8,ms=4.5)
for ax,title in zip(axes,['Training fit: 8 proteins','Development: 16 reused proteins']):
 ax.set_title(title,fontweight='bold');ax.set_xlabel('Additional optimizer updates');ax.set_ylabel('Weighted block-center RMSD (Å)')
 ax.set_xticks(steps);ax.grid(axis='y',alpha=.25);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8.5,frameon=False)
axes[0].set_yscale('log')
fig.suptitle('Coarse targets can be fitted, but the development gap remains',fontweight='bold',fontsize=14,y=.98)
fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.22,wspace=.26)
fig.text(.075,.11,'Same original coarse checkpoint per seed; 1,024 updates on eight previously seen training proteins; structure and fine fixed.',fontsize=9,color='#536578')
fig.text(.075,.055,'True CA-block-center RMSD after proper alignment, not full-backbone RMSD. Deterministic regression is not an ensemble model.',fontsize=9,color='#536578')
for ext in ['png','pdf','svg']:fig.savefig(run/f'fit_generalization_curves.{ext}',dpi=180,facecolor='white')
plt.close(fig)
(run/'plot_manifest.json').write_text(json.dumps(dict(comparison_sha256=hashlib.sha256((run/'comparison.json').read_bytes()).hexdigest(),drawing_script=str(Path(__file__))),indent=2)+'\n')
print('COARSE_SMALLSET_PLOT_COMPLETE')
