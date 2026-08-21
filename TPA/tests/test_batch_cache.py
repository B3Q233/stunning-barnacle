"""公共分类缓存归一化单测（fixture，CPU）。"""
import tempfile
import unittest
from pathlib import Path

from attacks.batch.runner import (
    attack_cache_path, ensure_classify_cache, normalize_cache)
from attacks.batch.utils import read_json
from attacks.batch.generator import build_atomic_base

from tests.test_batch_config import _base_cfg


class NormalizeCacheTest(unittest.TestCase):

    def test_override_dataset_drives_attack_cache_path(self):
        cfg = _base_cfg()
        cfg["override"] = {"dataset": "yelp2018"}
        base = build_atomic_base(cfg)
        self.assertEqual(base["dataset"], "yelp2018")
        p = attack_cache_path(base)
        self.assertTrue(p.as_posix().endswith(
            "attacks/bandwagon/data/rec_freq/yelp2018/lightgcn_top10.json"))

    def test_ordinary_mapped_to_normal_and_meta(self):
        cfg = _base_cfg()
        attack_cache = {
            "categories": {"popular": [1, 5], "ordinary": [31, 42],
                           "cold": [251, 987]},
            "summary": {"num_items": 1000},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            normalize_cache(cfg, attack_cache, cache_dir)
            rec = read_json(cache_dir / "rec_freq.json")
            self.assertEqual(rec["popular"], [1, 5])
            self.assertEqual(rec["normal"], [31, 42])
            self.assertEqual(rec["cold"], [251, 987])
            meta = read_json(cache_dir / "meta.json")
            for key in ("dataset", "model", "topk", "checkpoint", "generated_at"):
                self.assertIn(key, meta)
            self.assertEqual(meta["dataset"], "ml100k")
            self.assertEqual(meta["topk"], 10)

    def test_ensure_reads_existing_cache(self):
        cfg = _base_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            rec_path = cache_dir / "rec_freq.json"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            rec_path.write_text(
                '{"popular": [1], "normal": [2], "cold": [3]}',
                encoding="utf-8")
            got = ensure_classify_cache(cfg, cache_dir=cache_dir)
            self.assertEqual(got["cold"], [3])

    def test_attack_cache_path(self):
        p = attack_cache_path(_base_cfg())
        self.assertTrue(p.as_posix().endswith(
            "attacks/bandwagon/data/rec_freq/ml100k/lightgcn_top10.json"))
