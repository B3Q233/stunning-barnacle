"""公有模型注册表（models/registry.py）单元测试"""
import unittest

import torch

from models.registry import (
    AVAILABLE_MODELS,
    get_dataset_cls,
    get_model_cls,
    get_model_entry,
    load_model_config,
)


class PublicRegistryTest(unittest.TestCase):
    """公有注册表同时登记 lightgcn / mf / wmf。"""

    def test_all_models_registered(self):
        self.assertEqual(
            sorted(AVAILABLE_MODELS),
            ["lightgcn", "mf", "wmf"],
        )

    def test_mf_entry(self):
        entry = get_model_entry("mf")
        self.assertEqual(entry["config_path"], "models/mf/config.yaml")
        self.assertIs(get_model_cls("mf").__name__, "MatrixFactorization")
        self.assertIs(get_dataset_cls("mf").__name__, "MFDataset")

    def test_wmf_entry(self):
        entry = get_model_entry("wmf")
        self.assertEqual(entry["config_path"], "models/wmf/config.yaml")
        self.assertIs(get_model_cls("wmf").__name__, "WMFModel")
        self.assertIs(get_dataset_cls("wmf").__name__, "WMFDataset")

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_model_entry("unknown_model")
        self.assertIn("unknown_model", str(ctx.exception))
        self.assertIn("wmf", str(ctx.exception))

    def test_load_model_config_wmf(self):
        cfg = load_model_config("wmf")
        self.assertEqual(cfg["model"]["factors"], 100)
        self.assertEqual(cfg["model"]["alpha"], 40)
        self.assertEqual(cfg["data"]["dataset"], "ml100k")

    def test_load_model_config_overrides(self):
        cfg = load_model_config("wmf", overrides={"factors": 200})
        self.assertEqual(cfg["model"]["factors"], 200)


class AttackRegistryShimTest(unittest.TestCase):
    """各攻击 registry.py 与公有注册表指向同一实现（单源）。"""

    def test_random_shim(self):
        from attacks.random.registry import get_model_cls as random_get
        self.assertIs(random_get, get_model_cls)
        self.assertIn("wmf", AVAILABLE_MODELS)

    def test_bandwagon_shim(self):
        from attacks.bandwagon.registry import AVAILABLE_MODELS as bw_models
        self.assertIs(bw_models, AVAILABLE_MODELS)

    def test_pgd_shim(self):
        from attacks.pgd.registry import get_model_entry as pgd_get
        self.assertIs(pgd_get, get_model_entry)
        self.assertEqual(pgd_get("mf")["model_cls"],
                         "models.mf.model:MatrixFactorization")
        self.assertIn("wmf", AVAILABLE_MODELS)

    def test_tpa_shim_keeps_active_model_name(self):
        from attacks.tpa.registry import active_model_name
        cfg = {"surrogate": {"enabled": True, "model_name": "wmf"}}
        self.assertEqual(active_model_name(cfg), "wmf")
        cfg2 = {"model": {"name": "mf"}}
        self.assertEqual(active_model_name(cfg2), "mf")


if __name__ == "__main__":
    unittest.main()
