import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

path = Path(__file__).resolve().parents[1] / "Hierarchical_VQ_VAE/esm/models/tokenizer_patch.py"
spec = importlib.util.spec_from_file_location("tokenizer_patch_under_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TokenizerPatchTests(unittest.TestCase):
    def setup_modules(self):
        encoder, decoder = nn.Module(), nn.Module()
        encoder.medium_codebook = nn.Module()
        encoder.medium_codebook.embeddings = nn.Parameter(torch.zeros(4, 2))
        encoder.fine = nn.Embedding(8, 2)
        decoder.medium_embed = nn.Embedding(9, 3)
        decoder.fine = nn.Embedding(13, 3)
        patch = dict(format_version=1, level="medium", changes_token_ids=True,
                     encoder_weight=torch.ones(4, 2), decoder_weight=torch.full((9, 3), 2.0))
        return encoder, decoder, patch

    def test_only_medium_weights_change(self):
        encoder, decoder, patch = self.setup_modules()
        original = encoder.fine.weight.clone(), decoder.fine.weight.clone()
        result = module.apply_medium_patch(encoder, decoder, patch)
        self.assertTrue(result["changes_token_ids"])
        self.assertTrue(torch.equal(encoder.medium_codebook.embeddings, patch["encoder_weight"]))
        self.assertTrue(torch.equal(decoder.medium_embed.weight, patch["decoder_weight"]))
        self.assertTrue(torch.equal(encoder.fine.weight, original[0]))
        self.assertTrue(torch.equal(decoder.fine.weight, original[1]))

    def test_bad_second_tensor_does_not_mutate_first(self):
        encoder, decoder, patch = self.setup_modules()
        patch["decoder_weight"] = torch.ones(1, 1)
        with self.assertRaises(ValueError):
            module.apply_medium_patch(encoder, decoder, patch)
        self.assertTrue((encoder.medium_codebook.embeddings == 0).all())

    def test_safe_file_roundtrip_and_nonfinite_rejection(self):
        encoder, decoder, patch = self.setup_modules()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.pt"
            torch.save(patch, path)
            module.apply_medium_patch(encoder, decoder, path)
        patch["encoder_weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            module.apply_medium_patch(encoder, decoder, patch)


if __name__ == "__main__":
    unittest.main()
