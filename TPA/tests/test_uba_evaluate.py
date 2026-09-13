"""UBA 目标用户群指标单测（论文 Table 1 口径，CPU）。

与 evaluation/attack_eval.py 的 target_hr@K（全合格用户口径）区分：
本指标只在 attack.uba.target_users 选出的用户上统计，且必须过滤这些用户在**干净**
训练集里的已交互物品。
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from attacks.uba.evaluate import (
    compute_target_user_metrics,
    format_target_user_report,
    save_target_user_report,
)


def make_scores(rows):
    return torch.tensor(rows, dtype=torch.float32)


class ComputeTargetUserMetricsTest(unittest.TestCase):

    SCORES = make_scores([
        [5.0, 4.0, 3.0, 2.0, 9.0],    # u0：item4 rank1，但训练集已交互 → 排除
        [9.0, 8.0, 6.0, 1.0, 7.0],    # u1：item4 rank3（k=3 命中）
        [1.0, 2.0, 3.0, 9.0, 0.5],    # u2：item4 rank5（未命中）
        [0.0, 0.0, 0.0, 0.0, 0.0],    # u3：未在目标用户列表里
    ])
    USER_IDS = [0, 1, 2, 3]
    CLEAN = {0: {4}, 1: set(), 2: set(), 3: set()}

    def test_hits_and_eligible_filter(self):
        out = compute_target_user_metrics(
            self.SCORES, self.USER_IDS, self.CLEAN, [0, 1, 2, 3], 4, 3)
        self.assertEqual(out["n_target_users"], 4)
        self.assertEqual(out["n_evaluated"], 3)   # u0 已交互目标物品 → 排除
        self.assertEqual(out["hit_users"], 1)     # 只有 u1 命中
        self.assertAlmostEqual(out["hr@k"], 1 / 3)
        self.assertAlmostEqual(out["ndcg@k"], (1 / math.log2(4)) / 3)
        self.assertAlmostEqual(out["mean_rank"], 3.0)

    def test_seen_items_are_filtered(self):
        """过滤已见物品后排名会前移：u1 的 9.0/8.0 是训练集已见物品。"""
        clean = {1: {0, 1}, 2: set()}
        out = compute_target_user_metrics(
            self.SCORES, self.USER_IDS, clean, [1, 2], 4, 3)
        # u1: 过滤掉 item0(9.0)、item1(8.0) 后剩 item2=6.0, item3=1.0, item4=7.0
        #     → item4 排名第 1（6.0 < 7.0）→ 命中且 NDCG=1
        self.assertEqual(out["hit_users"], 1)
        self.assertAlmostEqual(out["hr@k"], 0.5)
        self.assertAlmostEqual(out["ndcg@k"], 1.0 / 2)

    def test_no_users_in_eval_set_returns_zeros(self):
        out = compute_target_user_metrics(
            self.SCORES, self.USER_IDS, self.CLEAN, [99], 4, 3)
        self.assertEqual(out["n_evaluated"], 0)
        self.assertEqual(out["hr@k"], 0.0)
        self.assertEqual(out["ndcg@k"], 0.0)
        self.assertIsNone(out["mean_rank"])
        self.assertIsNone(out["mean_rank_all"])

    def test_rank_all_uses_strict_greater(self):
        scores = make_scores([
            [5.0, 4.0, 3.0, 2.0, 9.0],
            [1.0, 8.0, 6.0, 7.0, 5.0],
        ])
        out = compute_target_user_metrics(
            scores, [0, 1], {0: set(), 1: set()}, [0, 1], 4, 3)
        # u0: 目标 9.0 最高 → rank 1；u1: 目标 5.0，严格更高有 3 个 → rank 4
        self.assertAlmostEqual(out["mean_rank_all"], 2.5)

    def test_k_larger_than_items_does_not_crash(self):
        out = compute_target_user_metrics(
            self.SCORES, self.USER_IDS, self.CLEAN, [1, 2], 4, 99)
        self.assertEqual(out["k"], 99)
        self.assertGreaterEqual(out["hr@k"], 0.0)


class ReportFormatTest(unittest.TestCase):

    METRICS = {
        "k": 10,
        "clean": {"n_evaluated": 50, "hit_users": 1, "hr@k": 0.02,
                  "ndcg@k": 0.01, "mean_rank_all": 900.0},
        "poisoned": {"n_evaluated": 50, "hit_users": 13, "hr@k": 0.26,
                     "ndcg@k": 0.12, "mean_rank_all": 300.0},
    }

    def test_report_contains_both_phases_and_delta(self):
        md = format_target_user_report(self.METRICS)
        self.assertIn("UBA 目标用户群指标（Top-10", md)
        self.assertIn("| clean | 50 | 1 |", md)
        self.assertIn("| poisoned | 50 | 13 |", md)
        self.assertIn("+0.2400", md)

    def test_save_writes_md_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            md = save_target_user_report(self.METRICS, out)
            self.assertEqual(md.name, "target_user_metrics.md")
            payload = json.loads(
                (out / "target_user_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["poisoned"]["hit_users"], 13)

    def test_missing_phase_does_not_crash(self):
        md = format_target_user_report({"k": 10, "clean": None, "poisoned": None})
        self.assertIn("UBA 目标用户群指标", md)


if __name__ == "__main__":
    unittest.main()
