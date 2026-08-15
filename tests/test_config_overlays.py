import unittest

from src.utils.utils import load_any_config


class ConfigOverlayTest(unittest.TestCase):
    def test_architecture_overlays_resolve_from_base(self):
        cfg = load_any_config("config/model_improvement/E_arch04_combined.yaml")
        self.assertTrue(cfg["tasks"]["mask"]["mask_embed_head"])
        self.assertEqual(cfg["model_parameters"]["transformer"]["n_heads"], 8)
        self.assertEqual(cfg["model_parameters"]["interaction_embedder"]["output_size"], 8)
        self.assertEqual(cfg["model_artefacts"]["log_dir"], "model_improvement/E_arch04_combined")


if __name__ == "__main__":
    unittest.main()
