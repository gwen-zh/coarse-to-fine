"""Run portable, synthetic-data tests without cluster artifacts or ESM weights."""
import os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
patterns=['test_hierarchical_latent_contract.py','test_hierarchical_rigid_vq.py','test_mixed_hierarchical_generator.py','test_coarse_geometry_supervision.py']
suite=unittest.TestSuite()
for pattern in patterns:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'upstream/tests'),pattern=pattern))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
