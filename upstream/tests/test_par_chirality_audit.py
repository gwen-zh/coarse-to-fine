from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from audit_par_coarse_chirality import TRANSFORMS, defect


class ChiralityAuditTests(unittest.TestCase):
    def test_transform_determinants_and_orthogonality(self):
        for name, values in TRANSFORMS.items():
            matrix = torch.tensor(values)
            torch.testing.assert_close(matrix.T @ matrix, torch.eye(3), atol=0, rtol=0)
            expected = -1. if name.startswith('reflect') else 1.
            self.assertEqual(float(torch.linalg.det(matrix)), expected)

    def test_radial_field_is_equivariant_under_rotations_and_reflections(self):
        x = torch.randn(4, 23, 3)
        for values in TRANSFORMS.values():
            matrix = x.new_tensor(values)
            absolute, relative = defect(-x, -(x @ matrix), matrix)
            torch.testing.assert_close(absolute, torch.zeros(4), atol=0, rtol=0)
            torch.testing.assert_close(relative, torch.zeros(4), atol=0, rtol=0)

    def test_cross_product_distinguishes_proper_rotations_from_reflection(self):
        x = torch.randn(4, 23, 3)
        def field(values):
            return torch.cross(values, values.roll(1, dims=1), dim=-1)
        original = field(x)
        for name, values in TRANSFORMS.items():
            matrix = x.new_tensor(values)
            absolute, relative = defect(original, field(x @ matrix), matrix)
            if name.startswith('reflect'):
                torch.testing.assert_close(relative, torch.full((4,), 2.), atol=1e-6, rtol=1e-6)
            else:
                torch.testing.assert_close(absolute, torch.zeros(4), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
