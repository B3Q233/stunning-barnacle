"""ranking_scores / compute_target_metrics 显存安全改造单测（CPU）。"""
import unittest

import torch

from evaluation.attack_eval import compute_target_metrics, ranking_scores


class _FakeModel:
    def __init__(self, user_emb, item_emb):
        self._u = user_emb
        self._i = item_emb
        self.eval_called = False

    def set_eval(self):
        self.eval_called = True

    def get_user_embeddings(self):
        return self._u

    def get_item_embeddings(self):
        return self._i


class RankingScoresTest(unittest.TestCase):

    def test_chunked_matches_reference_and_returns_cpu(self):
        torch.manual_seed(0)
        user_emb = torch.randn(6, 4)
        item_emb = torch.randn(5, 4)
        model = _FakeModel(user_emb, item_emb)
        test_pairs = [(0, 1), (0, 3), (2, 0), (5, 4), (5, 2)]
        scores, users, test_pos = ranking_scores(
            model, test_pairs, batch_size=2)
        self.assertTrue(model.eval_called)
        self.assertFalse(scores.is_cuda, "分数应已回 CPU")
        self.assertEqual(scores.shape, (3, 5))
        self.assertEqual(users, [0, 2, 5])
        self.assertEqual(test_pos, {0: {1, 3}, 2: {0}, 5: {2, 4}})
        ref = user_emb[torch.LongTensor(users)] @ item_emb.T
        self.assertTrue(torch.allclose(scores, ref, atol=1e-6))


class ComputeTargetMetricsChunkTest(unittest.TestCase):

    def test_device_kwarg_accepts_cpu(self):
        torch.manual_seed(3)
        scores = torch.randn(5, 7)
        user_ids = list(range(5))
        clean = {1: {2}}
        a = compute_target_metrics(
            scores, user_ids, clean, [2, 5], k=3, device="cpu")
        b = compute_target_metrics(
            scores, user_ids, clean, [2, 5], k=3)
        self.assertEqual(a, b)

    def test_chunked_equals_unchunked(self):
        torch.manual_seed(1)
        scores = torch.randn(6, 8)
        user_ids = [0, 1, 2, 3, 4, 5]
        clean = {0: {1}, 2: {5}}
        targets = [3, 6]
        a = compute_target_metrics(
            scores, user_ids, clean, targets, k=3, chunk_size=2)
        b = compute_target_metrics(
            scores, user_ids, clean, targets, k=3, chunk_size=1024)
        self.assertEqual(a, b)
        self.assertIn(3, a)
        self.assertEqual(a[3]["n_elig"], 6)
        self.assertTrue(1.0 <= a[3]["mean_rank_all"] <= 8.0)
        self.assertTrue(0.0 <= a[3]["hr@k"] <= 1.0)
