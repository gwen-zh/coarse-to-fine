"""English supervisor-facing research progression, with explicit evidence scopes."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper'/'figures'
STEM='research_evolution_en_v1'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
colors=dict(ink='#223448',muted='#607284',line='#8D9EAE',blue='#286BBC',teal='#168783',orange='#AF7520',purple='#745AA8')
fig,ax=plt.subplots(figsize=(24,20.7));fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
ax.set(xlim=(0,24),ylim=(0,20.7));ax.axis('off');fig.patch.set_facecolor('white')
checks=[]

def text(x,y,s,size=11,color=None,weight='normal',ha='left',zorder=5):
 return ax.text(x,y,s,fontsize=size,color=color or colors['ink'],fontweight=weight,ha=ha,va='center',zorder=zorder)

def card(num,x,y,title,cohort,lines,outcome,kind):
 w,h=6.8,3.7;color=colors[kind]
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.14',fc='#F7F9FC',ec='#D9E2EC',lw=1.1,zorder=1);ax.add_patch(p)
 ax.add_patch(FancyBboxPatch((x+.21,y+h-.64),.43,.43,boxstyle='round,pad=0,rounding_size=.07',fc=color,ec='none',zorder=2))
 text(x+.425,y+h-.425,str(num),size=13,color='white',weight='bold',ha='center')
 items=[text(x+.82,y+h-.425,title,size=13.7,weight='bold'),text(x+.25,y+h-.96,cohort,size=10.0,color=colors['muted'])]
 for i,line in enumerate(lines):items.append(text(x+.25,y+h-1.43-i*.31,line,size=10.7))
 ax.plot([x+.25,x+w-.25],[y+.58,y+.58],color='#DFE6EE',lw=1,zorder=2)
 items.append(text(x+.25,y+.28,outcome,size=9.5,color=color,weight='bold'))
 checks.append((str(num),p,items))
 return dict(left=(x,y+h/2),right=(x+w,y+h/2),top=(x+w/2,y+h),bottom=(x+w/2,y))

def arrow(a,b):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=colors['line'],lw=1.8,zorder=0))

text(.8,19.96,'Research Evolution: What Changed, What Improved, and Why',size=26,weight='bold')
text(.82,19.42,'From auditing the published model to a learned two-level structural interface and sequence-only generation',size=12,color=colors['muted'])
for x,k,label in [(.85,'blue','Architecture / diagnostic'),(6.4,'teal','Benefit within a stated comparison'),(13.1,'orange','Tradeoff / negative result'),(18.6,'purple','New-model research')]:
 ax.add_patch(FancyBboxPatch((x,18.81),.20,.20,boxstyle='round,pad=0,rounding_size=.03',fc=colors[k],ec='none'))
 text(x+.32,18.91,label,size=10.3,color=colors['muted'])

c1=card(1,.8,14.5,'Audit the published model','Architecture and tokenizer diagnostics',[
 'Same residue latent → three parallel codebooks',
 'Fine reuses a pretrained ESM3 tokenizer',
 'Medium usage: 41 / 512 codes in the early audit',
 'Parallel scales do not prove zero conditional information',
 'Question: can representation repairs help?'],
 'OUTCOME: A TESTABLE FAILURE MAP','blue')
c2=card(2,8.6,14.5,'Repair the representation','Medium: 58 holdout  |  RVQ: 40 dev, 3 seeds',[
 'Medium CA RMSD: 8.810 → 4.641 Å (−47.3%)',
 'Code use: 39 → 510 / 512 (same 40 dev)',
 'RVQ vs parallel: 1.565 → 1.228 Å (−21.5%)',
 'RVQ / parallel: 26 bits; original fine: 12 bits',
 'Reconstruction benefit; no generation claim'],
 'MEDIUM: CONFIRMED  |  RVQ: MATCHED DEV COMPARISON','teal')
c3=card(3,16.4,14.5,'Adapt generation with LoRA','58 holdout  |  Modified coarse → RVQ pipeline',[
 'Matched-pipeline RMSD: 13.416 → 12.228 Å',
 '−1.187 Å (−8.9%); 40 / 58 proteins improve',
 'Clash density increases; spread decreases 12.4%',
 'Native fine LoRA: unconfirmed development signal',
 'Benefit belongs to its own modified pipeline'],
 'ACCURACY BENEFIT, WITH GEOMETRY / SPREAD COSTS','teal')
c4=card(4,16.4,9.7,'Add fine-token decoder feedback','Original route  |  64 new proteins × 8 outputs',[
 'Fine edits → decode → gate → accept / reject → repeat',
 'CA–C–N angle error: 4.958 → 4.809° (ungated)',
 'C–N–CA angle error: 5.039 → 4.886° (ungated)',
 'CN outliers: 9.500 → 7.538% (gated metric)',
 'Reference RMSD: 3.249 → 3.251 Å; extra compute'],
 'CONFIRMED: SMALL LOCAL GEOMETRY IMPROVEMENT','teal')
c5=card(5,8.6,9.7,'Test other frozen-interface changes','CE / KL, REPA, continuous / discrete FM, packing',[
 'Lower token loss did not reliably improve structure',
 'Tested REPA / corrector settings: no overall win',
 'Direct segment broadcast: poor reconstruction',
 'These tests do not establish general impossibility',
 'Motivation: learn a structural interface we control'],
 'OUTCOME: DIAGNOSTICS, NOT CONFIRMED UPGRADES','orange')
c6=card(6,.8,9.7,'Build two distinct structural latents','Own trainable structural encoders and decoder',[
 'Coarse: block centers; fine: within-block shape',
 'M = ceil(N / 4); fine remains residue-level',
 'Enforce P(fine residual) = 0',
 'Add neighboring-block context and rigid N / CA / C',
 'Cross-residue connectivity still needs learning'],
 'BUILT: EXPLICIT COARSE / FINE RESPONSIBILITIES','purple')
c7=card(7,.8,4.9,'Diagnose coarse quantization','Same 101 development structures  |  Two seeds',[
 'Full-quant CA RMSD: 2.410 / 2.723 Å',
 'Hybrid CA RMSD: 0.376 / 0.385 Å',
 'CN: 10.15 / 12.20% → 4.27 / 4.59%',
 'Continuous coarse + 1024-code fine; no new updates',
 'Changed information rate; reconstruction only'],
 'DEVELOPMENT RECONSTRUCTION GAIN','teal')
c8=card(8,8.6,4.9,'Train sequence-only generation','899 train / 101 dev  |  60 coarse + 60 fine epochs',[
 'Frozen ESM3 provides amino-acid features only',
 'Continuous coarse flow → masked fine generation',
 'Generation evaluated on 16 dev proteins × 8 / seed',
 'CA RMSD: 10.456 / 10.660 Å; CN: 60.26 / 61.54%',
 'Good reconstruction did not transfer to generation'],
 'TRAINED: SEQUENCE GENERATION QUALITY STILL FAILS','orange')
c9=card(9,16.4,4.9,'Diagnose current generation failures','Two seeds  |  Reused 16-protein development set',[
 '65–66% generated code / slot pairs unseen in train',
 'Slot support: CN −3.66 / −3.62 percentage points',
 'Clashes / residue −28.4% / −39.7%',
 'But CA RMSD worsens by 0.082 / 0.080 Å',
 'More position training / solver steps: no overall win'],
 'CURRENT BOTTLENECK: COARSE + FINE GENERALIZATION','orange')
for a,b in [(c1['right'],c2['left']),(c2['right'],c3['left']),(c3['bottom'],c4['top']),
            (c4['left'],c5['right']),(c5['left'],c6['right']),(c6['bottom'],c7['top']),
            (c7['right'],c8['left']),(c8['right'],c9['left'])]:arrow(a,b)
text(20.2,13.95,'bounded fine repair',size=9.5,color=colors['muted'])
text(4.6,9.15,'learn the interface',size=9.5,color=colors['muted'])

for x,w,title,lines,k in [(.8,10.9,'ESTABLISHED BRANCH',[
 'Original model + constrained fine-token search',
 'Independent confirmation of two ungated local-angle gains.',
 'No demonstrated global-accuracy or thermodynamic-coverage gain.'
 ],'teal'),(12.3,10.9,'NEW-MODEL BRANCH',[
 'Trainable two-level structural model; ≈0.38 Å dev reconstruction.',
 'Sequence → coarse / fine prediction remains the unresolved step.',
 'The original 4096-token search is not yet validated on this model.'
 ],'purple')]:
 p=FancyBboxPatch((x,1.75),w,2.20,boxstyle='round,pad=0,rounding_size=.15',fc='#F0F5FA',ec='#D9E2EC',lw=1);ax.add_patch(p)
 text(x+.3,3.55,title,size=12.5,color=colors[k],weight='bold')
 for i,line in enumerate(lines):text(x+.3,3.06-i*.38,line,size=10.9)
text(.83,1.19,'Arrows show research logic, not that all improvements are combined. Cohorts, tasks, information rates and baselines differ; gains must not be added.',size=10.4,color=colors['muted'])
text(.83,.75,'Confirmed search changes local geometry at an inference cost. New-model reconstruction and sequence generation are reported separately.',size=10.4,color=colors['muted'])
text(.83,.32,'Snapshot: 2026-10-28  |  OpenMM / particle comparisons are planned, not completed. Detailed evidence and limitations accompany this figure.',size=9.8,color=colors['muted'])

fig.canvas.draw();renderer=fig.canvas.get_renderer();problems=[]
for key,patch,items in checks:
 bounds=patch.get_window_extent(renderer)
 for item in items:
  b=item.get_window_extent(renderer)
  if b.x0<bounds.x0+5 or b.x1>bounds.x1-5 or b.y0<bounds.y0+2 or b.y1>bounds.y1-2:problems.append([key,item.get_text()])
if problems:raise RuntimeError('Text exceeds card: '+json.dumps(problems))
for ext,kw in [('png',{'dpi':180}),('pdf',{}),('svg',{})]:fig.savefig(OUT/f'{STEM}.{ext}',facecolor='white',**kw)
fig.savefig(OUT/f'{STEM}_preview.png',facecolor='white',dpi=90)
plt.close(fig)
sources=['artifacts/reports/benefits_2026-10-26.rst','paper/claims_and_gaps.rst',
 'paper/protocols/hierarchical_coarse_repair_zh.rst','paper/protocols/mixed_generator_repair_results_zh.rst',
 'artifacts/mixed_hierarchical_followup_v1/results_zh.rst','artifacts/position_aligned_fine_v1/decision.json',
 'artifacts/coarse_solver_convergence_v1/decision.json','artifacts/reports/supervisor_research_story_2026-10-28.rst']
manifest=dict(date='2026-10-28',scope='Research logic and evidence, not an integrated-model superiority claim.',
              drawing_script=str(Path(__file__).relative_to(ROOT)),
              evidence={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(status='created',bounds_check='passed',files=[str(OUT/f'{STEM}.{ext}') for ext in ['png','pdf','svg']]),indent=2))
