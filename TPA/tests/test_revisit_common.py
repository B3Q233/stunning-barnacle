import unittest

import numpy as np

from models.revisit_common import (
    build_user_items,
    build_train_mask,
    compute_ranking_metrics,
    validate_pairs,
)


class RevisitCommonTest(unittest.TestCase):
    def setUp(self):
        self.pairs = [(0, 0), (0, 1), (1, 1)]

    def test_validate_pairs_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            validate_pairs([(2, 0)], num_users=2, num_items=2)

    def test_user_items_and_mask(self):
        user_items = build_user_items(self.pairs, num_users=2)
        self.assertEqual(user_items, [{0, 1}, {1}])
        mask = build_train_mask(user_items, [0, 1], num_items=3)
        self.assertEqual(mask.tolist(), [[True, True, False], [False, True, False]])

    def test_ranking_metrics(self):
        scores = np.array([[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]])
        test_items = [{2}, {0}]
        result = compute_ranking_metrics(scores, test_items, k=2)
        self.assertAlmostEqual(result["recall@2"], 0.5)
        self.assertGreater(result["ndcg@2"], 0.0)


if __name__ == "__main__":
    unittest.main()