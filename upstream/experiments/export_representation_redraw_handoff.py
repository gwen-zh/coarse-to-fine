"""Export the exact three-panel example for an external redraw; no model changes."""
import csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa
from hierarchical_rigid_vq import orthogonal_frame

OUT=Path('artifacts/representation_redraw_handoff_v1')
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path('artifacts/representation_three_panel_v1/coordinates.npz')
with np.load(SOURCE) as z:data={k:z[k].copy() for k in z.files}
p=json.loads(Path('artifacts/pdb_training_v2/manifest.json').read_text());row=next(x for x in p['targets'] if x['id']=='109L_1')
structure=MMCIFParser(QUIET=True).get_structure('109L',row['raw']);chain=next(structure.get_models())['A'];residues=[r for r in chain if is_aa(r,standard=True)]
with np.load('artifacts/pdb_training_v2/backbone/109L_1.npz') as z:original=z['backbone'].astype(np.float32)
parsed=np.array([[r[a].coord for a in ['N','CA','C']] for r in residues],dtype=np.float32)
np.testing.assert_array_equal(parsed,original)
x=torch.from_numpy(original);frame=orthogonal_frame(x[0,1]-x[0,2],x[0,0]-x[0,1]).numpy();origin=x[:,1].mean(0).numpy()
np.testing.assert_allclose((original-origin)@frame,data['reference'],atol=1e-5,rtol=0)
resnum=[int(r.id[1]) for r in residues];assert resnum==list(range(1,163))
highlight={int(k) for k in data['highlighted_residues_zero_based']}

def table(name,header,rows):
 with (OUT/name).open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)

def pdb_line(serial,atom,resname,resid,xyz,occupancy=1.,bfactor=0.,element=''):
 element=element or atom[0]
 return f'ATOM  {serial:5d} {atom:>4s} {resname:>3s} A{resid:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}\n'

# Native full atoms are transformed into the SAME frame as coarse/fine.
lines=['HEADER    T4 LYSOZYME 109L CHAIN A; CANONICAL FIGURE FRAME\n'];serial=1
for i,residue in enumerate(residues):
 for atom in residue:
  xyz=(atom.coord.astype(float)-origin)@frame
  lines.append(pdb_line(serial,atom.name,residue.resname,resnum[i],xyz,atom.occupancy or 1.,atom.bfactor or 0.,atom.element));serial+=1
lines+=['TER\n','END\n'];(OUT/'reference_full_atoms.pdb').write_text(''.join(lines))
lines=['HEADER    DECODED N/CA/C BACKBONE; COARSE AND FINE JOINT DECODING\n'];serial=1
for i,residue in enumerate(residues):
 for j,atom in enumerate(['N','CA','C']):
  lines.append(pdb_line(serial,atom,residue.resname,resnum[i],data['decoded'][i,j]));serial+=1
lines+=['TER\n','END\n'];(OUT/'decoded_backbone.pdb').write_text(''.join(lines))
table('reference_ca.csv',['residue_number','residue_name','x','y','z','highlight'],[[resnum[i],r.resname,*[float(v) for v in data['reference'][i,1]],int(i in highlight)] for i,r in enumerate(residues)])
table('coarse_centers.csv',['block_number','first_residue','last_residue','x','y','z','highlight'],[[i+1,4*i+1,min(4*i+4,len(residues)),*[float(v) for v in xyz],int(any(k in highlight for k in range(4*i,min(4*i+4,len(residues)))))] for i,xyz in enumerate(data['coarse_centers'])])
block=min(highlight)//4;center=data['coarse_centers'][block]
local=[]
for i in sorted(highlight):
 for j,atom in enumerate(['N','CA','C']):
  ref=data['reference'][i,j];pred=data['decoded'][i,j]
  local.append([resnum[i],residues[i].resname,atom,*[float(v) for v in ref],*[float(v) for v in pred],*[float(v) for v in ref-center],*[float(v) for v in pred-center]])
table('fine_local_atoms.csv',['residue_number','residue_name','atom','reference_x','reference_y','reference_z','decoded_x','decoded_y','decoded_z','local_reference_x','local_reference_y','local_reference_z','local_decoded_x','local_decoded_y','local_decoded_z'],local)
table('fine_tokens.csv',['residue_number','fine_token_id'],[[resnum[i],int(k)] for i,k in enumerate(data['fine_ids'])])
np.savez_compressed(OUT/'exact_coordinates.npz',**data)
brief='''Please redraw this as a simple scientific figure with three panels, using the attached coordinates.

PROTEIN
T4 lysozyme, PDB 109L, chain A. Dataset ID 109L_1 (the suffix is the polymer entity ID, not the chain name). There are 162 modeled residues, numbered 1-162. The deposited sequence is 164 residues; unmodeled residues 163-164 are not shown. Source: https://www.rcsb.org/structure/109L

PANEL A — Original structure
Use reference_ca.csv to draw the native C-alpha trace, connecting CA atoms in residue order. Use dark gray; highlight residues 81-84 in blue. If a molecular cartoon is preferred, reference_full_atoms.pdb contains the same native protein atoms in the exact shared figure frame. Use a real molecular renderer for a cartoon rather than inventing helices/sheets.

PANEL B — Coarse block centers
Use coarse_centers.csv: 41 actual centers from the trained coarse-only decoder. Connect centers in block order and draw orange points. Blocks 1-40 have four residues each; block41 contains residues161-162. Highlight block21, corresponding to residues81-84. Use the same camera, spatial scale and orientation as panelA. Center connections are visual guides, not chemical bonds. Do not replace these learned decoded coordinates by centers re-averaged from the native PDB.

PANEL C — Fine local detail, with coarse fixed
Use fine_local_atoms.csv for residues81-84 (ASN,ALA,LYS,LEU): 12 N/CA/C atoms. Blue is the joint decoder reconstruction from native fine tokens and the SAME coarse latent as panelB; light gray is the native reference. Use local_decoded_x/y/z and local_reference_x/y/z to put the fixed block center at (0,0,0). Mark that center in orange. Draw backbone bonds in the order N_i-CA_i-C_i-N_(i+1); label N/CA/C on one residue. Optional dotted lines from the orange center to the four decoded CA atoms indicate offsets, not bonds. A magnified view may use a different camera from A/B, but do not realign blue and gray independently. The coarse input is held fixed; fine is not independently decoded into a complete protein.

MODEL CONTEXT
Current independently trained hybrid tokenizer, seed2026: coarse latent shape41x16 (continuous), fine token sequence length162 with vocabulary1024. Fine vectors are retrieved from the codebook and decoded together with coarse and neighboring-token context. This is native structure encode/decode, not sequence-only generation or the original author's4096-token model.

STYLE
Plain white background, three large structural views, short labels. No decorative modules, dashboards, extra charts, improvement claims, or invented protein geometry. Title suggestion: T4 lysozyme (109L). Panel labels: Original structure / Coarse block centers / Fine local detail. Connect the highlighted center in B to the magnified block in C with a subtle zoom guide.

COORDINATES
All files use the SAME canonical frame, in angstroms. exact_coordinates.npz and CSV preserve numerical coordinates; PDB files are rounded to0.001A. reference_full_atoms.pdb contains experimentally observed native atoms; decoded_backbone.pdb contains ONLY the model's N/CA/C atoms. Model-generated oxygen/side-chain atoms are not supplied.
'''
(OUT/'redraw_brief_for_claude.txt').write_text(brief)
figure=Path('paper/figures/supervisor_representation_simple_v1/Coarse_Fine_One_Protein.svg')
(OUT/'current_figure.svg').write_bytes(figure.read_bytes())
metadata=dict(protein='T4 lysozyme',pdb='109L',auth_chain='A',dataset_id='109L_1',modeled_residues=162,deposited_sequence_length=164,highlighted_residues=[81,82,83,84],highlighted_block_one_based=21,units='angstrom',source_url='https://www.rcsb.org/structure/109L',source_coordinate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),files={str(q.name):hashlib.sha256(q.read_bytes()).hexdigest() for q in OUT.iterdir() if q.is_file() and q.name!='manifest.json'})
(OUT/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
zip_path=OUT.parent/'representation_redraw_for_claude.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for q in sorted(OUT.iterdir()):
  if q.is_file():z.write(q,arcname=q.name)
print(json.dumps({'zip':str(zip_path),'files':[q.name for q in sorted(OUT.iterdir()) if q.is_file()],'size_kb':round(zip_path.stat().st_size/1024,1),'native_pdb_backbone_verified_against_original':True},indent=2))
