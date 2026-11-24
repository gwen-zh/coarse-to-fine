"""Strict native-MD backbone extraction with explicit trajectory roles and units."""
import re
from pathlib import Path
import numpy as np

RESIDUE_AA = dict(zip(
    'ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),
    'ACDEFGHIKLMNPQRSTVWY'))
RESIDUE_AA.update(HID='H', HIE='H', HIP='H', CYX='C')
ROLES = {'R1': 'train', 'R2': 'development', 'R3': 'sealed_test'}


def backbone_indices(topology, expected_sequence):
    residues = [r for r in topology.residues if r.is_protein or r.name in RESIDUE_AA]
    if not residues or len({r.chain.index for r in residues}) != 1:
        raise ValueError('One complete protein chain required')
    sequence = ''.join(RESIDUE_AA.get(r.name, 'X') for r in residues)
    if sequence != expected_sequence or 'X' in sequence:
        raise ValueError('Topology sequence differs from frozen selection')
    indices = []
    for residue in residues:
        names = {}
        for atom in residue.atoms:
            names.setdefault(atom.name, []).append(atom.index)
        if any(len(names.get(name, [])) != 1 for name in ['N', 'CA', 'C']):
            raise ValueError('Exactly one N, CA, and C per selected residue required')
        indices.append([names[name][0] for name in ['N', 'CA', 'C']])
    return np.asarray(indices, dtype=np.int64)


def backbone_angstrom(xyz_nm, indices):
    values = np.asarray(xyz_nm)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Finite trajectory coordinates in nm required')
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.min() < 0 or indices.max() >= values.shape[1]:
        raise ValueError('Invalid N/CA/C topology indices')
    return np.asarray(values[:, indices, :] * 10., dtype=np.float32)


def uniform_frame_indices(total_frames, maximum=16):
    if total_frames < 2 or maximum < 2:
        raise ValueError('At least two frames needed for a trajectory diagnostic')
    return np.linspace(0, total_frames-1, min(total_frames, maximum), dtype=np.int64)


def diagnostic_trajectories(raw_root):
    paths = list(Path(raw_root).rglob('*.xtc'))
    result = {}
    for path in paths:
        match = re.search(r'(?:^|_)R([123])(?:_|\.|$)', path.name, re.IGNORECASE)
        if match is None:
            raise ValueError('Unknown ATLAS trajectory naming: '+path.name)
        replicate = 'R'+match.group(1)
        if replicate in result:
            raise ValueError('Duplicate ATLAS replicate')
        result[replicate] = path
    if set(result) != set(ROLES):
        raise ValueError('Expected exactly three native MD replicates')
    # R3 is never opened by diagnostic code. Downloading is not evaluating.
    return {replicate: result[replicate] for replicate in ['R1', 'R2']}
