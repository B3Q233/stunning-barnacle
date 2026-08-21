"""结果整合单测（fixture 分层 runs 目录，CPU）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.aggregate import (
    build_results_rows, scan_runs, tier_summary,
    write_results_csv, write_summary_md)

from tests.test_batch_config import _base_cfg


def _make_run(root, group, tier, item, best):
    d = root / group / tier / f"item{item}"
    d.mkdir(parents=True)
    (d / "history.json").write_text(
        json.dumps({"history": [], "best": best}, ensure_ascii=False),
        encoding="utf-8")


def _best_entry(value):
    """BestTracker.best_results() 的真实条目格式。"""
    return {"epoch": 1, "value": value, "metrics": {}, "checkpoint": "x.pt"}


class AggregateTest(unittest.TestCase):

    def test_scan_and_rows(self):
        cfg = _base_cfg()
        group = "bandwagon_ml100k_lightgcn_top10"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, group, "cold", 251,
                      {"target_hr@10": _best_entry(0.5),
                       "target_ndcg@10": _best_entry(0.4),
                       "recall@10": _best_entry(0.3),
                       "ndcg@10": _best_entry(0.2)})
            _make_run(root, group, "cold", 987,
                      {"target_hr@10": _best_entry(0.7),
                       "target_ndcg@10": _best_entry(0.6),
                       "recall@10": _best_entry(0.31),
                       "ndcg@10": _best_entry(0.21)})
            _make_run(root, group, "popular", 32,
                      {"target_hr@10": _best_entry(0.2),
                       "target_ndcg@10": _best_entry(0.18),
                       "recall@10": _best_entry(0.32),
                       "ndcg@10": _best_entry(0.22)})
            self.assertEqual(len(scan_runs(root, group)), 3)
            rows = build_results_rows(root, group, cfg, 10)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["attack"], "bandwagon")
            self.assertEqual(rows[0]["dataset"], "ml100k")
            self.assertEqual(rows[0]["model"], "lightgcn")
            self.assertEqual(rows[0]["tier"], "cold")
            summary = tier_summary(rows, 10)
            self.assertAlmostEqual(summary["cold"]["target_hr@10"]["mean"], 0.6)
            self.assertAlmostEqual(summary["cold"]["target_ndcg@10"]["mean"], 0.5)
            self.assertEqual(summary["cold"]["target_hr@10"]["n"], 2)
            write_results_csv(rows, 10, root / "results.csv")
            write_summary_md(
                "2026-08-21-15-30", summary,
                {"recall@10": 0.35, "ndcg@10": 0.25}, 10,
                root / "summary.md")
            csv_text = (root / "results.csv").read_text(encoding="utf-8")
            self.assertIn("target_ndcg@10", csv_text)
            md_text = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("0.6000", md_text)
            self.assertIn("Clean Model Utility", md_text)

    def test_missing_history_skipped(self):
        cfg = _base_cfg()
        group = "bandwagon_ml100k_lightgcn_top10"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / group / "cold").mkdir(parents=True)
            self.assertEqual(build_results_rows(root, group, cfg, 10), [])
