import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

path = Path(__file__).resolve().parents[1] / 'Hierarchical_VQ_VAE/esm/models/residual_bridge.py'
spec = importlib.util.spec_from_file_location('residual_bridge_test_module', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResidualBridgeTests(unittest.TestCase):
    def test_encoding_prefixes_and_safe_roundtrip(self):
        centers = [torch.tensor([[0.], [10.]]), torch.tensor([[0.], [2.]])]
        fine = torch.tensor([[0.], [10.], [12.]])
        bridge = module.ResidualTokenBridge(centers, fine)
        tokens = bridge.encode(torch.tensor([[[12.]]]))
        self.assertEqual(tokens.tolist(), [[[1, 1]]])
        self.assertEqual(bridge.fine_tokens(tokens[..., :1]).item(), 1)
        self.assertEqual(bridge.fine_tokens(tokens).item(), 2)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'bridge.pt'
            torch.save(dict(version=1, centers=centers, fine_codes=fine,
                            mode='residual', sizes=[2, 2], metadata={'generator_trained': False}), p)
            loaded, metadata = module.load_residual_bridge(p)
            self.assertFalse(metadata['generator_trained'])
            self.assertTrue(torch.equal(loaded.encode(torch.tensor([[[12.]]])), tokens))

    def test_invalid_tokens_and_codebooks(self):
        bridge = module.ResidualTokenBridge([torch.zeros(2, 1)], torch.zeros(3, 1))
        for bad in [torch.tensor([[2]]), torch.tensor([[-1]]), torch.zeros(1, 1), torch.zeros(1, 2, dtype=torch.long)]:
            with self.assertRaises(ValueError):
                bridge.fine_tokens(bad)
        with self.assertRaises(ValueError):
            module.ResidualTokenBridge([torch.tensor([[float('nan')]])], torch.zeros(3, 1))


if __name__ == '__main__':
    unittest.main()
