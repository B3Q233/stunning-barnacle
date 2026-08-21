"""Generator：分层采样 + 四层合并原子配置单测（CPU）。"""
import tempfile
import unittest
from pathlib import Path

from attacks.batch.generator import (
    atomic_run_tag,
    build_atomic_config,
    config_rel_path,
    generate_configs,
    sample_targets,
    write_configs,
)

from tests.test_batch_config import _base_cfg


def _categories():
    return {"popular": [1, 2], "normal": [5, 6], "cold": [9]}


class SampleTargetsTest(unittest.TestCase):

    def test_random_fixed_seed_deterministic(self):
        categories = {"popular": [0, 1, 2, 3, 4],
                      "normal": [10, 11, 12], "cold": [20, 21, 22, 23]}
        a = sample_targets(categories, ["popular", "normal", "cold"], 2,
                           "random", 42)
        b = sample_targets(categories, ["popular", "normal", "cold"], 2,
                           "random", 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a["popular"]), 2)
        self.assertTrue(all(i in categories["popular"] for i in a["popular"]))
        self.assertEqual(len(a["cold"]), 2)

    def test_first_takes_head(self):
        out = sample_targets({"cold": [20, 21, 22]}, ["cold"], 2,
                             "first", 42)
        self.assertEqual(out["cold"], [20, 21])

    def test_empty_tier_skipped(self):
        out = sample_targets({"cold": []}, ["cold"], 2, "random", 42)
        self.assertEqual(out["cold"], [])


class AtomicConfigTest(unittest.TestCase):

    def test_merged_atomic_config(self):
        cfg = _base_cfg()
        cfg["override"] = {"attack": {"filler_size": 40}}
        atomic = build_atomic_config(cfg, 251, "cold", "2026-08-21-15-30")
        self.assertEqual(atomic["attack"]["ratio"], 0.03)       # P4 默认
        self.assertEqual(atomic["attack"]["filler_size"], 40)   # P2 override
        self.assertEqual(atomic["training"]["epochs"], 5)       # P3 batch
        self.assertEqual(atomic["attack"]["target_items"],
                         {"strategy": "specified", "ids": [251]})
        self.assertEqual(atomic["run_tag"], "2026-08-21-15-30-cold-item251")
        self.assertEqual(atomic["output"]["dir"],
                         "attacks/batch/output/2026-08-21-15-30/runs")
        self.assertEqual(atomic["dataset"], "ml100k")
        self.assertNotIn("experiment", atomic)
        self.assertNotIn("batch", atomic)
        self.assertNotIn("override", atomic)

    def test_run_tag_and_rel_path(self):
        cfg = _base_cfg()
        self.assertEqual(atomic_run_tag(cfg, "cold", 251, "2026-08-21-15-30"),
                         "2026-08-21-15-30-cold-item251")
        self.assertEqual(config_rel_path(cfg, "cold", 251),
                         "bandwagon_ml100k_lightgcn_top10/cold/item251.yaml")


class GenerateAndWriteTest(unittest.TestCase):

    def test_generate_configs_count_and_names(self):
        cfg = _base_cfg()
        entries = generate_configs(cfg, _categories(), "2026-08-21-15-30")
        # per_tier=3，各层池 2/2/1 → 每层取满可用数量，共 2+2+1=5
        self.assertEqual(len(entries), 5)
        rels = [rel for rel, _ in entries]
        self.assertEqual(len(set(rels)), 5)
        self.assertIn("bandwagon_ml100k_lightgcn_top10/cold/item9.yaml", rels)

    def test_write_configs_hierarchical(self):
        cfg = _base_cfg()
        entries = generate_configs(cfg, _categories(), "t")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_configs(entries, Path(tmp))
            self.assertEqual(len(paths), 5)
            rel = "bandwagon_ml100k_lightgcn_top10/cold/item9.yaml"
            self.assertTrue((Path(tmp) / rel).exists())
            import yaml
            content = yaml.safe_load((Path(tmp) / rel).read_text(encoding="utf-8"))
            self.assertEqual(content["attack"]["target_items"]["ids"], [9])
            self.assertNotIn("batch", content)
            self.assertNotIn("override", content)
