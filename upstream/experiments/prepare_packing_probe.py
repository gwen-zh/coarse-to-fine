import argparse,hashlib,json
from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser,DSSP
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1
from diagnostic_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(exist_ok=True,parents=True);names=json.loads(Path('artifacts/development/objective_geometry_diagnosis/protocol.json').read_text())['geometry_targets'];corpus=Path('artifacts/pdb_training_v2');rows={r['id']:r for r in json.loads((corpus/'manifest.json').read_text())['targets']};targets=[]
    for name in names:
        row=rows[name];assert row['subset']=='validation';raw=Path(row['raw']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==row['raw_sha256'];model=next(MMCIFParser(QUIET=True).get_structure(name,str(raw)).get_models());residues=[r for r in model[row['auth_chain']] if is_aa(r,standard=True)];assert ''.join(seq1(r.resname) for r in residues)==row['sequence'];dssp=DSSP(model,str(raw),dssp='/usr/bin/mkdssp');labels=[];missing=[]
        for residue in residues:
            key=(row['auth_chain'],residue.id)
            if key not in dssp:
                labels.append('X');missing.append(str(key));continue
            value=dssp[key][2];labels.append('H' if value in 'HGI' else 'E' if value in 'EB' else 'C')
        assert len(labels)==row['length'];targets.append(dict(target=name,length=row['length'],ss3=''.join(labels),dssp_missing_residues=missing,latent=str(corpus/'tokens'/(name+'.npz')),reference=str(corpus/row['backbone']),raw_sha256=row['raw_sha256']));print('PACKING_DSSP_READY',name,flush=True)
    code=['spatial_packing.py','run_packing_probe.py'];protocol=dict(version=1,scope='Oracle compression/reconstruction only on24 reused validation proteins. True DSSP boundaries require real structures; unassigned residues retained as explicit singletonX, never removed or called coil; not a sequence-only generator, learned codebook or deployable packing method. Sequence-contiguous SSE grouping, not general3D spatial packing. No independent confirmation cohort access.',targets=targets,chunks=4,weights='artifacts/tokenizer/decoder_checkpoint_110.pth',codes='artifacts/development/residual_tokens/residual_bridge.pt',arms=['original','fixed4_snap','dssp_snap','dssp_cap4_snap','dssp_embedding_average'],notes='Frozen original fine-code centers and decoder embeddings. Segment mean latent nearest fine code, broadcast IDs; continuous decoder-embedding averaging diagnoses a different bottleneck without training. Count actual segments and boundary side information; no assumed4x end-to-end compute gain. No new embedding/codebook fitting.',code_sha256={str(Path('upstream/experiments')/f):hashlib.sha256((Path('upstream/experiments')/f).read_bytes()).hexdigest() for f in code});atomic_json(a.run/'protocol.json',protocol)


if __name__=='__main__':main()
