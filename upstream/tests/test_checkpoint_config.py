import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ConfigTests(unittest.TestCase):
    def test_released_checkpoint_fallback_includes_model_defaults(self):
        path = Path(__file__).resolve().parents[1] / "hierarchical_ConfGen/slm/utils/checkpoint_utils.py"
        spec = importlib.util.spec_from_file_location("checkpoint_utils_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configs = []

        class StopBeforeWeights(Exception):
            pass

        def capture(cfg):
            configs.append(cfg)
            raise StopBeforeWeights

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "released/mp_rank_00_model_states.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            with patch.object(module.hydra.utils, "instantiate", side_effect=capture):
                with self.assertRaises(StopBeforeWeights):
                    module.load_state_dict_from_lightning_ckpt(checkpoint, device="cpu")
        cfg = configs[0]
        self.assertEqual(cfg.optimizer._target_, "torch.optim.AdamW")
        self.assertEqual(cfg._target_, "slm.models.model.MaskedDiffusionLanguageModeling")
        self.assertEqual(cfg.net._target_, "slm.models.net.CustomizedESM3")
        self.assertNotIn("config", cfg.net)
        self.assertFalse(cfg.net.pretrained)
        self.assertIsNone(cfg.scheduler)


if __name__ == "__main__":
    unittest.main()
