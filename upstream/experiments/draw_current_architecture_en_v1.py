"""Draw the implemented two-level structural model and its training/inference paths.

This is an architecture figure, not a performance claim. The historical joint
training and the currently assembled mixed structural interface are distinguished.
Run from the workspace root with its scientific Python environment.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper' / 'figures'
STEM = 'current_two_level_architecture_en_v1'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.unicode_minus': False,
})
C = dict(ink='#223448', muted='#5C6D7D', line='#718397', panel='#F7F9FC',
         panel_border='#DCE4EC', blue='#286BBC', blue_bg='#EAF2FD',
         teal='#168783', teal_bg='#E8F6F3', purple='#745AA8',
         purple_bg='#F0ECF8', orange='#B87922', orange_bg='#FFF4DF',
         gray='#8593A0', gray_bg='#EEF1F5', white='#FFFFFF',
         output='#245D53', output_bg='#EAF5F0')
fig, ax = plt.subplots(figsize=(24, 22))
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 24), ylim=(0, 22))
ax.axis('off')
text_checks = []


def txt(x, y, text, size=11, color=None, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, text, fontsize=size, color=color or C['ink'],
                   fontweight=weight, ha=ha, va=va, zorder=8, **kw)


def panel(bottom, height, letter, title, subtitle):
    ax.add_patch(FancyBboxPatch((.55, bottom), 22.9, height,
        boxstyle='round,pad=0,rounding_size=.18', facecolor=C['panel'],
        edgecolor=C['panel_border'], linewidth=1.1, zorder=0))
    top = bottom + height
    ax.add_patch(FancyBboxPatch((.9, top-.71), .43, .43,
        boxstyle='round,pad=0,rounding_size=.08', facecolor=C['ink'],
        edgecolor='none', zorder=2))
    txt(1.115, top-.495, letter, size=15, weight='bold', color='white', ha='center')
    txt(1.53, top-.495, title, size=16.7, weight='bold')
    txt(.95, top-1.03, subtitle, size=10.9, color=C['muted'])


def node(key, x, y, w, h, title, body=(), kind='gray', status=None, title_size=11.5, body_size=10.3):
    edge = C[kind] if kind in C else C['line']
    fill = C.get(kind+'_bg', C['white'])
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=.11', facecolor=fill,
        edgecolor=edge, linewidth=1.25, zorder=4)
    ax.add_patch(patch)
    lines = [title] + list(body)
    gap = .255 if h < 1.65 else .31
    center = y+h/2 - (.05 if status else 0)
    start = center + (len(lines)-1)*gap/2
    entries=[]
    for i, line in enumerate(lines):
        obj=txt(x+w/2, start-i*gap, line, size=title_size if i==0 else body_size,
                weight='bold' if i==0 else 'normal', color=edge if i==0 else C['ink'], ha='center')
        entries.append(obj)
    if status:
        badge_w = max(.66, len(status)*.085+.16)
        ax.add_patch(FancyBboxPatch((x+w-badge_w-.1, y+h-.13), badge_w, .23,
            boxstyle='round,pad=0,rounding_size=.06', facecolor=edge,
            edgecolor='white', linewidth=.6, zorder=9))
        badge_text = txt(x+w-badge_w/2-.1, y+h-.015, status, size=7.4, color='white', weight='bold', ha='center')
        badge_text.set_zorder(10)
    b=dict(x=x,y=y,w=w,h=h,l=(x,y+h/2),r=(x+w,y+h/2),
           t=(x+w/2,y+h),b=(x+w/2,y),c=(x+w/2,y+h/2))
    text_checks.append((key, patch, entries))
    return b


def arrow(points, color=None, dashed=False, lw=1.45, mutation=11):
    color=color or C['line']
    path=MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=mutation,
        color=color, linewidth=lw, linestyle=(0,(4,3)) if dashed else '-',
        capstyle='round', joinstyle='round', zorder=3))


def tag(x,y,label,color=None,size=9.3,ha='center'):
    return txt(x,y,label,size=size,color=color or C['muted'],ha=ha,
               bbox=dict(boxstyle='round,pad=.12',fc=C['panel'],ec='none'))


# Overall title and visual legend.
txt(.8, 21.45, 'Two-Level Protein Conformation Generator', size=27, weight='bold')
txt(.82, 20.94, 'Current implementation  |  structure learning, generator training, and sequence-only inference', size=12, color=C['muted'])
for x,color,label in [(.87,C['blue'],'TRAIN = gradient-trained module'),
                      (6.3,C['orange'],'EMA = learned codebook update'),
                      (11.65,C['gray'],'FIXED = weights held constant')]:
    ax.add_patch(FancyBboxPatch((x,20.34),.21,.21,boxstyle='round,pad=0,rounding_size=.035',fc=color,ec='none'))
    txt(x+.34,20.445,label,size=10.1,color=C['muted'])
arrow([(18.15,20.445),(18.85,20.445)], color=C['line'])
txt(18.98,20.445,'data / conditioning',size=9.5,color=C['muted'])

# A. Current structure interface, with historically accurate training provenance.
panel(13.6,6.25,'A','STRUCTURE MODEL TRAINING & CURRENT LATENT INTERFACE',
      'Training history: 20 continuous warm-up epochs, then 80 quantized epochs. The current hybrid interface is shown below.')
a_x=node('A_input',.95,15.25,2.1,2.15,'INPUT',
         ['Reference backbone X','N residues; N / CA / C','Shape: N × 3 × 3'],kind='gray',body_size=10.1)
a_p=node('A_centers',3.55,17.20,2.7,1.25,'Block CA centers',
         ['P(X): M × 3','M = ceil(N / 4)'],kind='blue')
a_h=node('A_residual',3.55,15.20,2.7,1.25,'Local residual',
         ['H(X) = X − lift(P(X))','Shape: N × 3 × 3'],kind='teal',body_size=9.8)
a_ec=node('A_encoder_coarse',6.75,17.20,2.8,1.25,'Coarse encoder E_c',
          ['Global Transformer','Block-level context'],kind='blue',status='TRAIN')
a_ef=node('A_encoder_fine',6.75,15.20,2.8,1.25,'Fine encoder E_f',
          ['Block-local MLP','Local backbone shape'],kind='teal',status='TRAIN')
a_c=node('A_coarse',10.05,17.20,2.6,1.25,'Continuous z_c',
         ['Shape: M × 16','Coarse quantization bypassed'],kind='blue',body_size=9.15)
a_f=node('A_fine',10.05,15.20,2.6,1.25,'Fine z_f → Q_f',
         ['N × 8 → IDs → code vectors','1024 codes, 8 dimensions'],kind='orange',status='EMA',body_size=9.8)
a_dc=node('A_decoder_coarse',13.2,17.20,3.5,1.25,'Coarse decoder D_c',
          ['Predict block centers','Shape: M × 3'],kind='blue',status='TRAIN')
a_df=node('A_decoder_fine',13.2,15.20,3.5,1.25,'Fine decoder D_f',
          ['Fine + neighboring block context','Per-residue rotation / translation'],kind='teal',status='TRAIN',body_size=9.55)
a_head=node('A_rigid',17.25,15.20,2.85,3.25,'Rigid composition',
            ['Ideal N / CA / C template','Apply R_i and t_i','Center the local residual','Add block centers','P(residual) = 0'],kind='purple',title_size=11.3,body_size=9.8)
a_out=node('A_output',20.65,16.02,2.25,1.60,'OUTPUT',
           ['Reconstructed X_hat','Shape: N × 3 × 3'],kind='output',body_size=10)
arrow([a_x['r'],(3.30,a_x['r'][1]),(3.30,a_p['l'][1]),a_p['l']],color=C['blue'])
arrow([a_x['r'],(3.30,a_x['r'][1]),(3.30,a_h['l'][1]),a_h['l']],color=C['teal'])
for left,right,color in [(a_p,a_ec,'blue'),(a_ec,a_c,'blue'),(a_c,a_dc,'blue'),
                         (a_h,a_ef,'teal'),(a_ef,a_f,'teal'),(a_f,a_df,'teal')]:
    arrow([left['r'],right['l']],color=C[color])
arrow([a_dc['b'],a_df['t']],color=C['blue'],lw=1.15,mutation=9)
tag(16.1,16.82,'center offsets',size=8.4)
arrow([a_dc['r'],(17.25,a_dc['r'][1])],color=C['blue'])
arrow([a_df['r'],(17.25,a_df['r'][1])],color=C['teal'])
arrow([a_head['r'],a_out['l']],color=C['purple'])
loss_a=node('A_loss',3.55,14.12,19.35,.62,
            'Structural loss: center + local/full reconstruction + bond / angle / clash + latent / commitment terms',
            kind='orange',title_size=10.8)
arrow([a_x['b'],(a_x['b'][0],14.43),loss_a['l']],color=C['orange'],dashed=True,lw=1.2)
arrow([a_out['b'],(a_out['b'][0],14.74)],color=C['orange'],dashed=True,lw=1.2)
txt(.95,13.84,'Hybrid checkpoint assembly: E_c / D_c from continuous warm-up; E_f / Q_f / D_f from the quantized endpoint. The trained 256-code coarse book is not used here.',
    size=9.2,color=C['muted'])

# B. Supervised generator training. Teacher structures supply labels, never deployment inputs.
panel(7.1,6.1,'B','GENERATOR TRAINING  |  LEARNED STRUCTURE MODEL FIXED',
      'Export fixed structural labels, train the coarse flow model, then freeze it and train the conditional fine model.')
b_aa=node('B_sequence',.95,10.45,2.10,1.3,'AA sequence',
          ['Length: N'],kind='gray')
b_esm=node('B_esm',3.55,10.45,2.55,1.3,'ESM3 features',
           ['Amino-acid input only','h: N × 1536'],kind='gray',status='FIXED')
b_h=node('B_features',6.60,10.45,2.15,1.3,'Sequence condition h',
         ['Pool to M blocks for G_c','Keep N residues for G_f'],kind='gray',title_size=10.1,body_size=9.0)
b_x=node('B_structure',.95,8.40,2.10,1.35,'Training backbone',
         ['Supervision only'],kind='gray',title_size=10.9)
b_teacher=node('B_teacher',3.55,8.40,2.55,1.35,'E_c + E_f + Q_f',
               ['Frozen label extraction'],kind='gray',status='FIXED',body_size=9.5)
b_labels=node('B_labels',6.60,8.40,2.15,1.35,'Training labels',
              ['z_c*: M × 16','s_f*: N token IDs'],kind='gray',body_size=9.7)
b_noise=node('B_flow_path',9.25,10.45,1.80,1.3,'Flow path',
             ['Noise + z_c*','State z_t, time t'],kind='blue',title_size=10.5,body_size=9.2)
b_mask=node('B_mask',9.25,8.40,1.80,1.35,'Mask fine IDs',
            ['s_f* → partial IDs','Random mask ratio'],kind='teal',title_size=10.0,body_size=8.9)
b_gc=node('B_coarse_generator',11.55,10.45,4.15,1.3,'Coarse flow model G_c',
          ['Pooled h + (z_t, t)','Predict a continuous velocity field'],kind='blue',status='TRAIN',body_size=10.1)
b_gf=node('B_fine_generator',11.55,8.40,4.15,1.35,'Fine masked model G_f',
          ['h + coarse condition + partial fine IDs','Predict 1024-way token logits'],kind='teal',status='TRAIN',body_size=9.9)
b_lc=node('B_coarse_loss',17.20,10.45,5.70,1.3,'L_coarse = flow MSE + 0.1 L_center',
          ['Velocity target: z_c* − noise','Decoded endpoint center MSE / 100, for t ≥ 0.5'],kind='orange',title_size=11,body_size=9.65)
b_lf=node('B_fine_loss',17.20,8.40,5.70,1.35,'L_fine = masked-token cross-entropy',
          ['Target: native fine IDs s_f* at masked positions','Structural encoders, codebook and decoder stay fixed'],kind='orange',title_size=11,body_size=9.65)
for left,right in [(b_aa,b_esm),(b_esm,b_h),(b_x,b_teacher),(b_teacher,b_labels)]:
    arrow([left['r'],right['l']])
arrow([b_labels['r'],b_mask['l']],color=C['teal'])
arrow([b_labels['r'],(8.98,b_labels['r'][1]),(8.98,b_noise['l'][1]),b_noise['l']],color=C['blue'])
arrow([b_h['t'],(b_h['t'][0],12.02),(13.625,12.02),b_gc['t']],color=C['line'])
tag(10.55,12.02,'pooled sequence condition',size=8.8)
arrow([b_h['b'],(b_h['b'][0],10.06),(13.625,10.06),b_gf['t']],color=C['line'])
tag(11.40,10.06,'sequence condition h',size=8.7)
arrow([b_labels['b'], (b_labels['b'][0],8.05), (13.625,8.05), b_gf['b']],color=C['blue'])
tag(11.4,8.05,'coarse condition: native / generated*',size=8.6)
for left,right,color in [(b_noise,b_gc,'blue'),(b_mask,b_gf,'teal'),(b_gc,b_lc,'blue'),(b_gf,b_lf,'teal')]:
    arrow([left['r'],right['l']],color=C[color])
txt(.95,7.68,'Order: G_c (60 epochs) → freeze G_c → G_f (60 epochs). *Coarse condition: native labels initially; later, native labels or a frozen G_c sample.',
    size=9.8,color=C['muted'])
txt(.95,7.35,'Real structures are used only to make training targets and evaluate outputs. ESM3 contributes sequence features; it is not the structural encoder or decoder.',
    size=9.6,color=C['muted'])

# C. Deployment. Every tensor entering the system originates from sequence or noise.
panel(.95,5.75,'C','GENERATION / INFERENCE  |  SEQUENCE ONLY',
      'Input: amino-acid sequence + sampling seed. All trained weights are fixed. No reference structure is supplied.')
c_in=node('C_input',.95,3.35,2.00,1.6,'INPUT',
          ['AA sequence','N residues'],kind='gray')
c_esm=node('C_esm',3.45,3.35,2.40,1.6,'Frozen ESM3',
           ['AA-only features','h: N × 1536'],kind='gray')
c_gc=node('C_gc',6.35,3.35,3.00,1.6,'Coarse generation',
          ['Gaussian noise → flow','16 Euler steps'],kind='blue',status='FIXED',title_size=11.2,body_size=10)
c_c=node('C_coarse',9.85,3.35,2.00,1.6,'Continuous z_c',
         ['M × 16','Block-level layout'],kind='blue',title_size=10.4,body_size=9.6)
c_gf=node('C_gf',12.35,3.35,3.35,1.6,'Fine generation',
          ['All MASK → N fine IDs','16 denoising rounds'],kind='teal',status='FIXED',body_size=10.3)
c_d=node('C_decoder',16.20,3.35,3.15,1.6,'Structural decoding',
         ['1024 × 8 fine-code lookup','D_c / D_f + rigid output'],kind='purple',status='FIXED',title_size=11,body_size=9.65)
c_out=node('C_output',19.85,3.35,3.05,1.6,'OUTPUT',
           ['N / CA / C backbone','Coordinates: N × 3 × 3'],kind='output',body_size=10.1)
for left,right,color in [(c_in,c_esm,'line'),(c_esm,c_gc,'blue'),(c_gc,c_c,'blue'),
                         (c_c,c_gf,'blue'),(c_gf,c_d,'teal'),(c_d,c_out,'purple')]:
    arrow([left['r'],right['l']],color=C[color])
arrow([c_esm['t'],(c_esm['t'][0],5.32),(c_gf['t'][0],5.32),c_gf['t']],color=C['line'])
tag(10.50,5.32,'same sequence features h',size=9.2)
noise=node('C_noise',6.45,2.05,2.80,.70,'Sampling seed → noise',kind='blue',title_size=10.2)
arrow([noise['t'],c_gc['b']],color=C['blue'])
arrow([c_c['b'],(c_c['b'][0],2.94),(c_d['b'][0],2.94),c_d['b']],color=C['blue'])
tag(14.35,2.94,'coarse latent also enters the decoder directly',size=9.0)
txt(12.0,1.64,'Repeat with different noise / sampling seeds → K backbone conformations',
    size=12.0,weight='bold',ha='center',color=C['output'])
txt(12.0,1.22,'Rigid templates constrain intra-residue geometry; inter-residue C–N connectivity is still learned.',
    size=10.5,ha='center',color=C['muted'])

# Scope footer: do not accidentally imply an unimplemented repair or thermodynamic model.
txt(.80,.54,'Implemented two-level interface: coarse controls block centers; fine controls zero-mean local shape. No separate mid level is used.',size=10,color=C['muted'])
txt(.80,.22,'Architecture snapshot: 2026-10-28  •  OpenMM relaxation and particle search are not yet connected to this pipeline.',size=9.5,color=C['muted'])

# Check node text against its intended bounding rectangle before export.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
problems=[]
for name, patch, entries in text_checks:
    bounds=patch.get_window_extent(renderer)
    for entry in entries:
        box=entry.get_window_extent(renderer)
        if box.x0 < bounds.x0+4 or box.x1 > bounds.x1-4 or box.y0 < bounds.y0+2 or box.y1 > bounds.y1-2:
            problems.append((name,entry.get_text(),[round(v,1) for v in box.bounds],[round(v,1) for v in bounds.bounds]))
if problems:
    raise RuntimeError('Node labels exceed bounds: '+json.dumps(problems,ensure_ascii=False))

for extension, kwargs in [('png',{'dpi':180}),('pdf',{}),('svg',{})]:
    fig.savefig(OUT/f'{STEM}.{extension}',facecolor='white',**kwargs)
fig.savefig(OUT/f'{STEM}_preview.png',dpi=90,facecolor='white')
plt.close(fig)

sources=[
    'upstream/experiments/hierarchical_rigid_vq.py',
    'upstream/experiments/hierarchical_latent_contract.py',
    'upstream/experiments/mixed_hierarchical_generator.py',
    'upstream/experiments/train_hierarchical_tokenizer.py',
    'upstream/experiments/train_mixed_hierarchical_generator.py',
    'paper/protocols/mixed_hierarchical_generation_zh.rst',
]
manifest=dict(figure=STEM,architecture_date='2026-10-28',language='English',
              source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
              drawing_script=str(Path(__file__).relative_to(ROOT)),
              scope='Implemented mixed structural interface and trained generator; not a performance claim or thermodynamic ensemble guarantee.',
              files=[f'{STEM}.{ext}' for ext in ['png','pdf','svg']])
(OUT/f'{STEM}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'created','files':manifest['files'],'node_label_bounds':'passed'},indent=2))
