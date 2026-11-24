import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import mdtraj as md
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from atlas_md_io import backbone_indices, backbone_angstrom, uniform_frame_indices, diagnostic_trajectories


class NativeMDInputTests(unittest.TestCase):
    def test_atom_names_control_backbone_order_and_nm_to_angstrom(self):
        top=md.Topology();chain=top.add_chain()
        for name in ['ALA','GLY']:
            res=top.add_residue(name,chain)
            for atom in ['C','H','N','CA']:
                top.add_atom(atom,md.element.hydrogen if atom=='H' else md.element.carbon,res)
        indices=backbone_indices(top,'AG')
        np.testing.assert_array_equal(indices,[[2,3,0],[6,7,4]])
        xyz=np.arange(24,dtype=np.float32).reshape(1,8,3)/10.
        result=backbone_angstrom(xyz,indices)
        np.testing.assert_allclose(result[0,0],[[6,7,8],[9,10,11],[0,1,2]],atol=1e-6)
    def test_incomplete_or_mismatched_topology_fails(self):
        top=md.Topology();res=top.add_residue('ALA',top.add_chain())
        for atom in ['N','CA']:top.add_atom(atom,md.element.carbon,res)
        with self.assertRaises(ValueError):backbone_indices(top,'A')
        with self.assertRaises(ValueError):backbone_indices(top,'G')
    def test_sealed_replicate_excluded_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for r in [1,2,3]:Path(tmp,f'example_A_R{r}.xtc').touch()
            selected=diagnostic_trajectories(tmp)
            self.assertEqual(set(selected),{'R1','R2'})
            self.assertNotIn('R3',selected)
            Path(tmp,'duplicate_R1.xtc').touch()
            with self.assertRaises(ValueError):diagnostic_trajectories(tmp)
    def test_frame_selection_is_fixed_ordered_and_includes_endpoints(self):
        indices=uniform_frame_indices(10001,16)
        self.assertEqual(len(indices),16)
        self.assertEqual((indices[0],indices[-1]),(0,10000))
        self.assertTrue(np.all(np.diff(indices)>0))
        np.testing.assert_array_equal(uniform_frame_indices(3,16),[0,1,2])


if __name__=='__main__':unittest.main()
