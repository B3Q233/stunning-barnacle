"""物品交互频次统计、排名曲线与流行度直方图的单元测试（纯 stdlib）。"""
from collections import Counter
import unittest

from visualization.item_freq.plot_item_freq import (
    build_popularity_histogram,
    build_series,
    count_interactions,
)


class CountInteractionsTest(unittest.TestCase):

    def test_counts_each_item_occurrence(self):
        lines = [
            "1 3 4 5",
            "2 3 5 7",
            "3 9",
        ]
        counts = count_interactions(lines)
        self.assertEqual(counts[3], 2)
        self.assertEqual(counts[4], 1)
        self.assertEqual(counts[5], 2)
        self.assertEqual(counts[7], 1)
        self.assertEqual(counts[9], 1)
        self.assertNotIn(1, counts)  # 行首是用户 id，不参与统计

    def test_build_series_fills_zero_interaction_items(self):
        counts = count_interactions(["0 0 2", "1 2"])
        x, y, original_ids = build_series(counts)
        self.assertEqual(x.tolist(), [1, 2, 3])   # x 轴 = 物品 id + 1
        self.assertEqual(y.tolist(), [1, 0, 2])   # 物品 id=1 无交互，补 0
        self.assertEqual(original_ids.tolist(), [0, 1, 2])

    def test_build_series_rank_sorted_by_count(self):
        counts = count_interactions(
            ["0 0", "1 1", "2 2", "3 1", "4 1", "5 1", "6 1", "7 2", "8 2"]
        )  # 交互数：物品 0 -> 1, 1 -> 5, 2 -> 3
        x, y, original_ids = build_series(counts, sort_by_count=True)
        self.assertEqual(x.tolist(), [1, 2, 3])         # 重映射后的新 id
        self.assertEqual(y.tolist(), [5, 3, 1])         # 按交互数降序
        self.assertEqual(original_ids.tolist(), [1, 2, 0])

    def test_build_series_rank_ties_and_zero_tail(self):
        counts = count_interactions(
            ["1 0 2", "2 0", "3 2", "4 1", "5 4"]
        )
        # 交互数：物品 0 -> 2, 1 -> 1, 2 -> 2, 3 -> 0, 4 -> 1
        x, y, original_ids = build_series(counts, sort_by_count=True)
        self.assertEqual(x.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(y.tolist(), [2, 2, 1, 1, 0])   # 同数按原 id 升序，0 交互在尾部
        self.assertEqual(original_ids.tolist(), [0, 2, 1, 4, 3])

    def test_build_series_empty_raises(self):
        with self.assertRaises(ValueError):
            build_series(count_interactions([]), sort_by_count=True)

    def test_build_popularity_histogram_bins_and_heights(self):
        counts = count_interactions(["0 0", "1 1", "2 2", "3 1", "4 1"])
        # 物品交互数：item0 -> 1, item1 -> 3, item2 -> 1（n_total=3）
        centers, widths, ratios, edges = build_popularity_histogram(counts)
        # 分箱：[1] / [2] / [3-4]
        self.assertEqual(edges.tolist(), [1.0, 2.0, 3.0, 5.0])
        self.assertEqual(ratios.shape, (3, 3))  # 列 = Tail / Medium-hot / Hot
        # 排名：item1（交互 3）为 Top 1 -> Hot；item0（交互 1）-> Medium-hot；
        # item2（交互 1，rank 2 >= 40% 边界）-> Tail
        self.assertAlmostEqual(float(ratios[0, 0]), 1 / 3 * 100)  # [1] 内 1 个 Tail
        self.assertAlmostEqual(float(ratios[0, 1]), 1 / 3 * 100)  # [1] 内 1 个 Medium
        self.assertAlmostEqual(float(ratios[2, 2]), 1 / 3 * 100)  # [3-4] 内 1 个 Hot
        self.assertAlmostEqual(float(ratios.sum()), 100.0)
        self.assertTrue((widths > 0).all() and (centers > 0).all())

    def test_build_popularity_histogram_empty_raises(self):
        with self.assertRaises(ValueError):
            build_popularity_histogram(count_interactions([]))

    def test_popularity_histogram_tier_totals_are_5_35_60(self):
        # 20 个物品，交互数 20, 19, ..., 1：Top 5% = 1 个，5-40% = 7 个，40-100% = 12 个
        counts = Counter({i: 20 - i for i in range(20)})
        _, _, ratios, _ = build_popularity_histogram(counts)
        tail = float(ratios[:, 0].sum())
        medium = float(ratios[:, 1].sum())
        hot = float(ratios[:, 2].sum())
        self.assertAlmostEqual(tail, 60.0)
        self.assertAlmostEqual(medium, 35.0)
        self.assertAlmostEqual(hot, 5.0)
