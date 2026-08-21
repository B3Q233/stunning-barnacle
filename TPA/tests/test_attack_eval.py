"""evaluation.attack_eval 单元测试（unittest，无第三方依赖）"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.attack_eval import (
    aggregate_target_metrics,
    build_attack_eval_metrics,
    compute_target_metrics,
    format_report,
    save_report,
)


def make_scores(rows):
    return torch.tensor(rows, dtype=torch.float32)


class ComputeTargetMetricsTest(unittest.TestCase):
    """3 用户 × 5 物品；目标物品 4。"""

    SCORES = make_scores([
        [5.0, 4.0, 3.0, 2.0, 9.0],    # u0：item4 rank1（但训练集已交互，被排除）
        [9.0, 8.0, 6.0, 1.0, 7.0],    # u1：item4 rank3（k=3 命中）
        [1.0, 2.0, 3.0, 9.0, 0.5],    # u2：item4 rank5（未命中）
    ])
    USER_IDS = [0, 1, 2]

    def test_hit_and_eligible_filter(self):
        clean = {0: {4}, 1: set(), 2: set()}  # u0 训练集已交互目标 → 排除
        out = compute_target_metrics(self.SCORES, self.USER_IDS, clean, [4], 3)
        m = out[4]
        self.assertEqual(m["n_elig"], 2)
        self.assertAlmostEqual(m["hr@k"], 0.5)
        self.assertAlmostEqual(m["ndcg@k"], 0.25)      # dcg = 1/log2(4) = 0.5，/2
        self.assertEqual(m["hit_users"], 1)
        self.assertAlmostEqual(m["mean_rank"], 3.0)
        self.assertAlmostEqual(m["mean_rank_all"], 4.0)  # (3 + 5) / 2
        # 旧别名兼容
        self.assertEqual(m["exposure"], m["hr@k"])
        self.assertEqual(m["ndcg"], m["ndcg@k"])

    def test_no_eligible_users(self):
        clean = {0: {4}, 1: {4}, 2: {4}}
        m = compute_target_metrics(self.SCORES, self.USER_IDS, clean, [4], 3)[4]
        self.assertEqual(m["n_elig"], 0)
        self.assertEqual(m["hr@k"], 0.0)
        self.assertEqual(m["ndcg@k"], 0.0)
        self.assertIsNone(m["mean_rank"])
        self.assertIsNone(m["mean_rank_all"])

    def test_all_eligible_hit_at_rank_one(self):
        scores = make_scores([[9.0, 1.0, 0.5], [8.0, 2.0, 0.5]])
        out = compute_target_metrics(scores, [0, 1], {0: set(), 1: set()}, [0], 3)
        self.assertEqual(out[0]["hr@k"], 1.0)
        self.assertAlmostEqual(out[0]["ndcg@k"], 1.0)  # rank1：dcg=1，IDCG=1

    def test_mean_rank_all_uses_strict_greater_count(self):
        # rank = 严格高于目标分的候选数 + 1（论文 rank_ui 定义；并列不算高于）
        scores = make_scores([
            [5.0, 4.0, 3.0, 2.0, 9.0],   # target=4: >9 的有 0 个 → rank 1
            [1.0, 8.0, 6.0, 7.0, 5.0],   # target=4: >5 的有 3 个 → rank 4
        ])
        out = compute_target_metrics(scores, [0, 1], {0: set(), 1: set()}, [4], 3)
        self.assertAlmostEqual(out[4]["mean_rank_all"], 2.5)  # (1 + 4) / 2

    def test_ties_not_counted_above(self):
        # 并列值不算"严格高于"：target=0 与另一物品同分 → rank=1
        scores = make_scores([[1.0, 1.0, 0.5]])
        out = compute_target_metrics(scores, [0], {0: set()}, [0], 3)
        self.assertEqual(out[0]["mean_rank_all"], 1.0)

    def test_multiple_targets_equal_single_calls(self):
        torch.manual_seed(7)
        scores = torch.randn(8, 12)
        user_ids = list(range(8))
        clean = {0: {1}, 3: {5}, 7: {2}}
        targets = [1, 5, 9]
        combined = compute_target_metrics(
            scores, user_ids, clean, targets, k=4)
        for t in targets:
            single = compute_target_metrics(
                scores, user_ids, clean, [t], k=4)[t]
            self.assertEqual(combined[t], single, f"target {t} 结果不一致")

    def test_chunk_size_independent(self):
        torch.manual_seed(11)
        scores = torch.randn(9, 15)
        user_ids = list(range(9))
        clean = {1: {4}, 5: {7}}
        targets = [2, 6, 10]
        a = compute_target_metrics(
            scores, user_ids, clean, targets, k=5, chunk_size=1)
        b = compute_target_metrics(
            scores, user_ids, clean, targets, k=5, chunk_size=1024)
        self.assertEqual(a, b)


class AggregateTargetMetricsTest(unittest.TestCase):
    def test_single_target(self):
        tm = {5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2}}
        agg = aggregate_target_metrics(tm, [5], 10)
        self.assertEqual(agg, {"target_hr@10": 0.5, "target_ndcg@10": 0.25})

    def test_multiple_targets_equal_mean(self):
        tm = {
            5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2},
            7: {"hr@k": 0.8, "ndcg@k": 0.6, "n_elig": 5},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertAlmostEqual(agg["target_hr@10"], 0.65)
        self.assertAlmostEqual(agg["target_ndcg@10"], 0.425)

    def test_skips_target_without_eligible_users(self):
        tm = {
            5: {"hr@k": 0.5, "ndcg@k": 0.25, "n_elig": 2},
            7: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertEqual(agg, {"target_hr@10": 0.5, "target_ndcg@10": 0.25})

    def test_all_no_eligible_returns_zero(self):
        tm = {
            5: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
            7: {"hr@k": 0.0, "ndcg@k": 0.0, "n_elig": 0},
        }
        agg = aggregate_target_metrics(tm, [5, 7], 10)
        self.assertEqual(agg, {"target_hr@10": 0.0, "target_ndcg@10": 0.0})


class BuildAttackEvalMetricsTest(unittest.TestCase):

    def test_device_kwarg_accepted(self):
        scores = make_scores([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        user_ids = [0, 1]
        user_items = {0: {1}, 1: {3}}
        test_pos = {0: {2}, 1: {0}}
        metrics_cfg = [
            {"target_ndcg@3": "upper"}, {"target_hr@3": "upper"},
            {"recall@3": "upper"}, {"ndcg@3": "upper"},
        ]
        a, _ = build_attack_eval_metrics(
            scores, user_ids, user_items, test_pos,
            clean_user_items={}, targets=[2], ks=[3],
            metric_names=["target_ndcg@3", "target_hr@3",
                          "recall@3", "ndcg@3"],
            device="cpu")
        self.assertIn("recall@3", a)
    """与 ComputeTargetMetricsTest 同夹具；整体指标按测试计划手算。"""

    SCORES = make_scores([
        [5.0, 4.0, 3.0, 2.0, 9.0],
        [9.0, 8.0, 6.0, 1.0, 7.0],
        [1.0, 2.0, 3.0, 9.0, 0.5],
    ])
    USER_IDS = [0, 1, 2]
    USER_ITEMS = {0: {0}, 1: set(), 2: set()}   # 整体指标过滤用（训练集）
    TEST_POS = {0: {1}, 1: {4}, 2: {0}}
    CLEAN = {0: {4}, 1: set(), 2: set()}        # 目标人群过滤用（干净训练集）
    TARGETS = [4]

    def test_with_target_metrics(self):
        names = ["target_ndcg@3", "target_hr@3", "recall@3", "ndcg@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, self.TARGETS, [3], names,
        )
        self.assertEqual(set(res), set(names))
        self.assertAlmostEqual(res["target_ndcg@3"], 0.25)
        self.assertAlmostEqual(res["target_hr@3"], 0.5)
        self.assertAlmostEqual(res["recall@3"], 2 / 3)
        # u0 ndcg = 1/log2(3)，u1 ndcg = 0.5，u2 = 0 → 均值
        self.assertAlmostEqual(
            res["ndcg@3"], (1.0 / math.log2(3) + 0.5) / 3)
        self.assertEqual(set(details), {4})
        self.assertEqual(details[4]["n_elig"], 2)

    def test_without_target_metrics(self):
        names = ["recall@3", "ndcg@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, self.TARGETS, [3], names,
        )
        self.assertEqual(set(res), set(names))
        self.assertEqual(details, {})

    def test_empty_targets_no_crash(self):
        names = ["target_ndcg@3", "recall@3"]
        res, details = build_attack_eval_metrics(
            self.SCORES, self.USER_IDS, self.USER_ITEMS, self.TEST_POS,
            self.CLEAN, [], [3], names,
        )
        self.assertEqual(res["target_ndcg@3"], 0.0)
        self.assertEqual(details, {})


class FormatReportTest(unittest.TestCase):
    REPORT = {
        "k": 3,
        "model_utility": {
            "clean": {"recall@3": 0.6, "ndcg@3": 0.4},
            "poisoned": {"recall@3": 0.58, "ndcg@3": 0.39},
        },
        "target_metrics": {
            "clean": {4: {"hr@k": 0.0, "ndcg@k": 0.0, "mean_rank_all": 900.0}},
            "poisoned": {4: {"hr@k": 0.5, "ndcg@k": 0.25, "mean_rank_all": 300.0}},
        },
    }

    def test_title_and_conclusion(self):
        md = format_report(self.REPORT, title="PGD（投影梯度上升投毒）攻击对比报告")
        self.assertIn("# PGD（投影梯度上升投毒）攻击对比报告（Top-3）", md)
        self.assertIn("## 结论", md)
        self.assertIn("投毒后目标物品曝光提升", md)
        self.assertIn("recall@3", md)

    def test_missing_recall_does_not_crash(self):
        report = dict(self.REPORT)
        report["model_utility"] = {
            "clean": {"ndcg@3": 0.4},
            "poisoned": {"ndcg@3": 0.39},
        }
        md = format_report(report)
        self.assertIn("## 结论", md)

    def test_no_utility_skips_utility_section(self):
        report = dict(self.REPORT)
        report["model_utility"] = {"clean": None, "poisoned": None}
        md = format_report(report)
        self.assertNotIn("模型效用（测试集", md)
        self.assertIn("## 结论", md)


class SaveReportTest(unittest.TestCase):
    def test_name_mapping_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            md = save_report(FormatReportTest.REPORT, out, name="pgd")
            self.assertEqual(md.name, "pgd_comparison.md")
            self.assertTrue((out / "pgd_comparison.json").exists())
            self.assertIn(
                "PGD（投影梯度上升投毒）攻击对比报告（Top-3）",
                md.read_text(encoding="utf-8"),
            )
            json.loads((out / "pgd_comparison.json").read_text(encoding="utf-8"))

    def test_tpa_and_template_default_to_attack_name(self):
        for name in ("tpa", "attack_imp_direct_poison"):
            with tempfile.TemporaryDirectory() as d:
                out = Path(d)
                save_report(FormatReportTest.REPORT, out, name=name)
                self.assertTrue((out / "attack_comparison.md").exists())


if __name__ == "__main__":
    unittest.main()
