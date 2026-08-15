"""WMF 评估指标（步骤④）单元测试：expected percentile rank 手算用例"""
import math
import unittest

import numpy as np
import torch

from evaluation.metrics import (
    compute_metrics,
    expected_percentile_rank,
    rank_values,
)


class ExpectedPercentileRankTest(unittest.TestCase):
    """论文 Eq.(8)：rank̄ = Σ r^t_ui·rank_ui / Σ r^t_ui，越低越好。"""

    def test_hand_computed_basic(self):
        # 3 用户 × 5 物品；无训练集过滤
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],   # u0: item1(4.0) 高于它的只有 item0 -> 1/5
            [1.0, 2.0, 3.0, 4.0, 5.0],   # u1: item0(1.0) -> 4/5；item4(5.0) -> 0/5
            [3.0, 3.0, 3.0, 3.0, 3.0],   # u2: 无测试项，不参与
        ])
        test_pos = {0: {1}, 1: {0, 4}}
        rank = expected_percentile_rank(scores, {}, test_pos)
        # 按观测单元平均：u0 一条(0.2)、u1 两条(0.8 与 0.0) -> (0.2+0.8+0.0)/3
        self.assertAlmostEqual(rank, 1.0 / 3.0, places=6)

    def test_train_mask_excluded_from_denominator(self):
        # u0 训练集已交互 item4 -> 置 -inf，不参与分母
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ])
        train_pos = {0: {4}}
        test_pos = {0: {1}}
        rank = expected_percentile_rank(scores, train_pos, test_pos)
        # 候选 4 个：item1(4.0) 高于它的只有 item0 -> 1/4 = 0.25
        self.assertAlmostEqual(rank, 0.25, places=6)

    def test_ties_not_counted_above(self):
        scores = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        test_pos = {0: {2}}
        rank = expected_percentile_rank(scores, {}, test_pos)
        self.assertEqual(rank, 0.0)  # 无严格高于的物品

    def test_weights_respect_test_intensity(self):
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        test_pos = {0: {1}, 1: {4}}
        weights = {0: {1: 3.0}, 1: {4: 1.0}}
        rank = expected_percentile_rank(scores, {}, test_pos, weights)
        # (3*0.2 + 1*0.0) / 4 = 0.15
        self.assertAlmostEqual(rank, 0.15, places=6)

    def test_empty_test_returns_nan(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        rank = expected_percentile_rank(scores, {}, {})
        self.assertTrue(math.isnan(rank))

    def test_masked_test_item_skipped(self):
        # 测试正样本同时出现在训练集 -> 被过滤，跳过该观测
        scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        train_pos = {0: {1}}
        test_pos = {0: {1}}
        rank = expected_percentile_rank(scores, train_pos, test_pos)
        self.assertTrue(math.isnan(rank))

    def test_rank_values_feed_cdf(self):
        """逐观测秩是 Rank CDF 的数据源（论文 Fig.2）。"""
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        test_pos = {0: {1}, 1: {0, 4}}
        ranks, weights = rank_values(scores, {}, test_pos)
        self.assertEqual(ranks.shape, (3,))
        self.assertTrue(np.allclose(np.sort(ranks), [0.0, 0.2, 0.8]))
        self.assertTrue(np.allclose(weights, [1.0, 1.0, 1.0]))


class RecallNdcgSmokeTest(unittest.TestCase):
    """recall@K / ndcg@K 与现有共享实现一致（回归保护）。"""

    def test_compute_metrics_small(self):
        scores = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        train_pos = {0: set(), 1: set()}
        test_pos = {0: {1}, 1: {4}}
        out = compute_metrics(scores, train_pos, test_pos, k=2)
        self.assertEqual(out["recall@2"], 1.0)
        # u0 命中 rank2 -> dcg=1/log2(3)；u1 命中 rank1 -> 1.0
        self.assertAlmostEqual(
            out["ndcg@2"], (1.0 / math.log2(3) + 1.0) / 2.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
