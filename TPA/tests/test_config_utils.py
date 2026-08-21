"""配置 k 统一解析与指标模板展开单测。"""
import unittest

from training.config_utils import apply_k, expand_metrics, resolve_k


class ResolveKTest(unittest.TestCase):

    def test_top_level_k_wins(self):
        cfg = {"k": 5, "evaluation": {"k": 10}, "classification": {"k": 20},
               "training": {"k": 30}}
        self.assertEqual(resolve_k(cfg), 5)

    def test_fallback_priority(self):
        self.assertEqual(resolve_k({"evaluation": {"k": 10}}), 10)
        self.assertEqual(resolve_k({"classification": {"k": 20}}), 20)
        self.assertEqual(resolve_k({"training": {"k": 30}}), 30)
        self.assertEqual(resolve_k({}), 20)


class ExpandMetricsTest(unittest.TestCase):

    def test_dict_and_string_templates(self):
        k = 10
        metrics = [
            {"target_ndcg@{k}": "upper"},
            "target_hr@{k} upper",
            {"recall@{k}": "upper"},
            "ndcg@{k}",
            "rank lower",  # 无 {k} 保持原样
        ]
        out = expand_metrics(metrics, k)
        self.assertEqual(out[0], {"target_ndcg@10": "upper"})
        self.assertEqual(out[1], "target_hr@10 upper")
        self.assertEqual(out[2], {"recall@10": "upper"})
        self.assertEqual(out[3], "ndcg@10")
        self.assertEqual(out[4], "rank lower")


class ApplyKTest(unittest.TestCase):

    def test_fills_sections_and_expands_metrics(self):
        cfg = {
            "k": 10,
            "classification": {"popular_ratio": 0.2},
            "training": {"epochs": 5},
            "evaluation": {"metrics": [{"target_ndcg@{k}": "upper"},
                                       {"recall@{k}": "upper"}]},
        }
        out = apply_k(cfg)
        self.assertEqual(out["classification"]["k"], 10)
        self.assertEqual(out["training"]["k"], 10)
        self.assertEqual(out["evaluation"]["k"], 10)
        self.assertEqual(out["evaluation"]["metrics"],
                         [{"target_ndcg@10": "upper"}, {"recall@10": "upper"}])

    def test_flat_shape_metrics_expanded(self):
        cfg = {"metrics": [{"recall@{k}": "upper"}], "evaluation": {"k": 20}}
        out = apply_k(cfg)
        self.assertEqual(out["metrics"], [{"recall@20": "upper"}])
        self.assertNotIn("training", out)  # 无 training 段则不注入

    def test_does_not_mutate_input(self):
        cfg = {"k": 10, "evaluation": {"metrics": [{"recall@{k}": "upper"}]}}
        apply_k(cfg)
        self.assertEqual(cfg["evaluation"]["metrics"], [{"recall@{k}": "upper"}])
