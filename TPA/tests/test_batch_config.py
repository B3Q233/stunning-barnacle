"""Batch 配置校验与公共工具单测（CPU）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.generator import load_batch_config, validate_batch_config
from attacks.batch.utils import deep_merge, flatten_experiment, group_name


def _base_cfg():
    return {
        "attack": {"name": "bandwagon"},
        "experiment": {"dataset": "ml100k", "seed": 42},
        "model": {"name": "lightgcn", "overrides": {}},
        "classification": {"k": 10, "popular_ratio": 0.2,
                           "checkpoint": "models/lightgcn/checkpoints/best.pt"},
        "warm_start": {"enabled": True,
                       "checkpoint": "models/lightgcn/checkpoints/best.pt"},
        "training": {"epochs": 5, "device": "cpu"},
        "batch": {"tiers": ["popular", "normal", "cold"], "per_tier": 3,
                  "strategy": "random", "seed": 42},
        "override": {},
    }


class ValidateBatchConfigTest(unittest.TestCase):

    def test_valid_passes(self):
        validate_batch_config(_base_cfg())

    def test_missing_experiment_raises(self):
        cfg = _base_cfg()
        del cfg["experiment"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_missing_batch_raises(self):
        cfg = _base_cfg()
        del cfg["batch"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_zero_per_tier_raises(self):
        cfg = _base_cfg()
        cfg["batch"]["per_tier"] = 0
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_unknown_tier_raises(self):
        cfg = _base_cfg()
        cfg["batch"]["tiers"] = ["hot"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)


class BatchConfigIOTest(unittest.TestCase):

    def test_load_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "batch.yaml"
            p.write_text(json.dumps(_base_cfg()), encoding="utf-8")
            cfg = load_batch_config(p)
        self.assertEqual(cfg["experiment"]["dataset"], "ml100k")


class UtilsTest(unittest.TestCase):

    def test_group_name(self):
        self.assertEqual(group_name(_base_cfg()),
                         "bandwagon_ml100k_lightgcn_top10")

    def test_flatten_experiment(self):
        flat = flatten_experiment(_base_cfg())
        self.assertEqual(flat["dataset"], "ml100k")
        self.assertEqual(flat["seed"], 42)
        self.assertNotIn("experiment", flat)
        self.assertNotIn("batch", flat)

    def test_deep_merge_nested_and_priority(self):
        attack_default = {"attack": {"ratio": 0.03, "filler_size": 20},
                          "training": {"epochs": 30}}
        batch = {"training": {"epochs": 10}}
        override = {"attack": {"filler_size": 40}}
        merged = deep_merge(deep_merge(attack_default, batch), override)
        self.assertEqual(merged, {
            "attack": {"ratio": 0.03, "filler_size": 40},
            "training": {"epochs": 10},
        })
        self.assertEqual(attack_default["training"]["epochs"], 30)
