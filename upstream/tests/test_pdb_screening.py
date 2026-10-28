import sys
from pathlib import Path
import unittest

from Bio import Align

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from prepare_pdb_training import close_match


class PdbScreeningTests(unittest.TestCase):
    def test_sequence_match_requires_identity_and_coverage(self):
        aligner=Align.PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,
                                     open_gap_score=-5,extend_gap_score=-0.5)
        seq='ACDEFGHIKLMNPQRSTVWY'
        self.assertTrue(close_match(seq,seq,aligner))
        self.assertTrue(close_match('HHHHHH'+seq,seq,aligner))
        self.assertFalse(close_match('AAAAAA','DDDDDD',aligner))
        self.assertFalse(close_match('WWWWWWACDEYYYYYYYYYYYY',seq,aligner))


if __name__=='__main__':unittest.main()
