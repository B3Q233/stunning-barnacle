"""攻击分类公共逻辑单测：按训练集交互数划分 popular/ordinary/cold。"""
import unittest
from collections import Counter

from attacks.classify_common import (
    classify_by_interaction_counts,
    interaction_counts,
)


class InteractionCountsTest(unittest.TestCase):

    def test_counts_each_item(self):
        pairs = [(0, 1), (1, 1), (2, 5), (3, 5), (4, 5)]
        counts = interaction_counts(pairs)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[5], 3)
        self.assertEqual(len(counts), 2)


class ClassifyByInteractionCountsTest(unittest.TestCase):

    def test_tier_sizes_5_35_60(self):
        counts = Counter({i: 100 - i for i in range(100)})  # 交互数 100..1
        categories, summary = classify_by_interaction_counts(counts, 100)
        self.assertEqual(len(categories["popular"]), 5)
        self.assertEqual(len(categories["ordinary"]), 35)
        self.assertEqual(len(categories["cold"]), 60)
        self.assertEqual(summary["popular_count"], 5)
        self.assertEqual(summary["ordinary_count"], 35)
        self.assertEqual(summary["cold_count"], 60)
        self.assertEqual(summary["basis"], "interaction_count")

    def test_tiers_follow_interaction_order(self):
        # 20 个物品，交互数 20..1：popular=1 个，ordinary=7 个，cold=12 个
        counts = Counter({i: 20 - i for i in range(20)})
        categories, summary = classify_by_interaction_counts(counts, 20)
        self.assertEqual(categories["popular"], [0])
        self.assertEqual(categories["ordinary"], list(range(1, 8)))
        self.assertEqual(categories["cold"], list(range(8, 20)))
        self.assertEqual(summary["min_popular_count"], 20)
        self.assertEqual(summary["max_cold_count"], 12)

    def test_ties_ordered_by_item_id(self):
        counts = Counter({0: 5, 1: 5, 2: 5, 3: 1, 4: 1})
        categories, _ = classify_by_interaction_counts(counts, 5, 0.4, 0.8)
        self.assertEqual(categories["popular"], [0, 1])
        self.assertEqual(categories["ordinary"], [2, 3])
        self.assertEqual(categories["cold"], [4])

    def test_zero_interaction_items_go_to_cold(self):
        counts = Counter({0: 10, 1: 5})  # 物品 2 无交互
        categories, summary = classify_by_interaction_counts(counts, 3, 0.2, 0.6)
        self.assertEqual(categories["popular"], [0])
        self.assertEqual(categories["ordinary"], [1])
        self.assertEqual(categories["cold"], [2])
        self.assertEqual(summary["interacting_items"], 2)

    def test_invalid_ratios_raise(self):
        with self.assertRaises(ValueError):
            classify_by_interaction_counts(Counter({0: 1}), 1, 0.5, 0.4)
