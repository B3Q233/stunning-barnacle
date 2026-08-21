"""评估快速路径单测：掩码索引预计算 + 分块 topk 与默认路径逐位一致。"""
import unittest

import torch

from evaluation.metrics import build_train_mask_indices, compute_metrics


class BuildTrainMaskIndicesTest(unittest.TestCase):

    def test_mask_indices_cover_expected_entries(self):
        train_user_items = {0: {1, 2}, 2: {5}, 4: {0, 7}}
        rows, cols = build_train_mask_indices(train_user_items, list(range(6)))
        got = set(zip(rows.tolist(), cols.tolist()))
        expected = {(0, 1), (0, 2), (2, 5), (4, 0), (4, 7)}
        self.assertEqual(got, expected)


class ComputeMetricsFastPathTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.scores = torch.randn(6, 10)
        self.train_user_items = {0: {1, 2}, 2: {5}, 4: {0, 7}}
        self.test_user_items = {0: {3}, 2: {4}, 5: {8}}
        self.rows, self.cols = build_train_mask_indices(
            self.train_user_items, list(range(6)))

    def _assert_same(self, a, b):
        self.assertEqual(sorted(a), sorted(b))
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=10)

    def test_mask_indices_equals_default(self):
        default = compute_metrics(self.scores.clone(), self.train_user_items,
                                  self.test_user_items, k=3)
        fast = compute_metrics(self.scores.clone(), self.train_user_items,
                               self.test_user_items, k=3,
                               mask_indices=(self.rows, self.cols))
        self._assert_same(default, fast)

    def test_chunked_topk_equals_default(self):
        default = compute_metrics(self.scores.clone(), self.train_user_items,
                                  self.test_user_items, k=3)
        fast = compute_metrics(self.scores.clone(), self.train_user_items,
                               self.test_user_items, k=3,
                               mask_indices=(self.rows, self.cols),
                               topk_device="cpu", chunk_size=2)
        self._assert_same(default, fast)
