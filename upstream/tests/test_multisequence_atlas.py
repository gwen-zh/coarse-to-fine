import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from Bio.Align import PairwiseAligner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import prepare_multisequence_atlas as data


class MDDataTests(unittest.TestCase):
    def test_exact_duplicate_and_length_bound(self):
        aligner = PairwiseAligner(mode="global", match_score=2, mismatch_score=-1,
                                 open_gap_score=-3, extend_gap_score=-.5)
        aa = "ACDEFGHIKLMNPQRSTVWY" * 4
        self.assertEqual(data.sequence_identity(aligner, aa, aa), 1.)
        self.assertEqual(data.sequence_identity(aligner, aa[:8], aa), 0.)

    def test_download_never_opens_R3_and_test_proteins_cannot_be_indexed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "protein_A"
            out.mkdir()
            archive = out / "protein_A_protein.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in ["protein_A.pdb", "protein_A_prod_R1_fit.xtc", "protein_A_prod_R2_fit.xtc", "protein_A_prod_R3_fit.xtc"]:
                    bundle.writestr(name, b"native-data-placeholder")
            selection = dict(selected=[dict(name="protein_A", official_split="train", sequence="ACDE",
                url="https://example.invalid", expected_bytes=archive.stat().st_size),
                dict(name="test_A", official_split="test")])
            (root / "selection.json").write_text(json.dumps(selection))
            original_open = zipfile.ZipFile.open
            def guarded_open(bundle, name, *args, **kwargs):
                filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
                if "_R3_" in filename:
                    raise AssertionError("R3 member was opened")
                return original_open(bundle, name, *args, **kwargs)
            with patch.object(data, "RUN", root), patch.object(zipfile.ZipFile, "open", guarded_open):
                data.download(selection, 0)
                with self.assertRaises(ValueError):
                    data.download(selection, 1)
            self.assertTrue((out / "complete.json").exists())
            self.assertFalse((out / "raw/protein_A_prod_R3_fit.xtc").exists())

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "unsafe_A"
            out.mkdir()
            archive = out / "unsafe_A_protein.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside.pdb", b"invalid")
            p = dict(selected=[dict(name="unsafe_A", official_split="train", sequence="ACDE",
                url="https://example.invalid", expected_bytes=archive.stat().st_size)])
            with patch.object(data, "RUN", root), self.assertRaises(RuntimeError):
                data.download(p, 0)
            self.assertFalse((root / "outside.pdb").exists())


if __name__ == "__main__":
    unittest.main()
