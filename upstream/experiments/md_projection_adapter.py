"""C-alpha-only bridge to official MD projections, with explicit Angstrom units.

This is sufficient for the contact-map projection, not for the benchmark's full
atom physical filters. Never silently apply this to those filters.
"""
import mdtraj as md
import numpy as np
from Bio.SeqUtils import seq3


def ca_trajectory(sequence, ca_angstrom):
    coords=np.asarray(ca_angstrom)
    if coords.ndim!=3 or coords.shape[1:]!=(len(sequence),3) or len(coords)==0:
        raise ValueError('Expected nonempty [samples, sequence_length, 3] CA coordinates')
    if not np.isfinite(coords).all():raise ValueError('Nonfinite coordinates must be counted and reported before projection')
    if any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in sequence):raise ValueError('Unsupported residue')
    topology=md.Topology();chain=topology.add_chain()
    for aa in sequence:
        residue=topology.add_residue(seq3(aa).upper(),chain)
        topology.add_atom('CA',md.element.carbon,residue)
    return md.Trajectory(np.asarray(coords/10.,dtype=np.float32),topology)
