"""Runner 调度单测：dry-run、目录整理、logs（stub run_atomic，不训练模型）。"""
import tempfile
import unittest
from pathlib import Path

import attacks.batch.runner as runner
from attacks.batch.utils import read_json

from tests.test_batch_config import _base_cfg
from tests.test_batch_generator import _categories


class RunnerTest(unittest.TestCase):

    def test_write_meta_fields(self):
        cfg = _base_cfg()
        entries = [("a", {"run_tag": "x"}), ("b", {"run_tag": "y"})]
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "meta.json"
            runner.write_meta(cfg, "2026-08-21-15-30", entries, meta_path)
            meta = read_json(meta_path)
        self.assertEqual(meta["batch_tag"], "2026-08-21-15-30")
        self.assertEqual(meta["attack"], "bandwagon")
        self.assertEqual(meta["dataset"], "ml100k")
        self.assertEqual(meta["model"], "lightgcn")
        self.assertEqual(meta["topk"], 10)
        self.assertEqual(meta["tiers"], ["popular", "normal", "cold"])
        self.assertEqual(meta["per_tier"], 3)
        self.assertEqual(meta["total_runs"], 2)
        self.assertEqual(meta["seed"], 42)

    def test_dry_run_writes_configs_and_meta_without_execution(self):
        cfg = _base_cfg()
        calls = []
        original = runner.run_atomic
        runner.run_atomic = lambda atomic, stage: calls.append(stage)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_root = Path(tmp)
                runner.run_batch(cfg, "t", out_root, _categories(),
                                 dry_run=True)
                self.assertEqual(calls, [])
                self.assertTrue((out_root / "configs").exists())
                self.assertTrue((out_root / "meta.json").exists())
                config_files = list((out_root / "configs").rglob("*.yaml"))
                # per_tier=3，层池 2/2/1 → 2+2+1=5 个原子配置
                self.assertEqual(len(config_files), 5)
        finally:
            runner.run_atomic = original

    def test_run_moves_staging_and_writes_logs(self):
        cfg = _base_cfg()

        def fake_run(atomic, stage):
            if stage == "model":
                src = runner.staging_dir(runs_root, atomic)
                src.mkdir(parents=True, exist_ok=True)
                (src / "history.json").write_text(
                    '{"history": [], "best": {}}', encoding="utf-8")

        original = runner.run_atomic
        runner.run_atomic = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_root = Path(tmp)
                runs_root = out_root / "runs"
                runner.run_batch(cfg, "t", out_root, _categories())
                moved = (runs_root / "bandwagon_ml100k_lightgcn_top10"
                         / "cold" / "item9")
                self.assertTrue(moved.exists())
                self.assertTrue((moved / "history.json").exists())
                self.assertTrue((out_root / "logs" / "runner.log").exists())
        finally:
            runner.run_atomic = original
