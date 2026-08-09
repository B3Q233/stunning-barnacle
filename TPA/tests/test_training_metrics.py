"""training.metrics 单元测试（unittest，无第三方依赖）"""
import unittest

from training.metrics import (
    BestTracker,
    default_direction,
    eval_ks_from_metrics,
    match_metric_values,
    metric_k,
    parse_metrics,
    safe_checkpoint_name,
)


class ParseMetricsTest(unittest.TestCase):
    def test_dict_and_string_annotations(self):
        cfg = [{"recall@20": "upper"}, "ndcg@20 lower", "hit@10"]
        d = parse_metrics(cfg)
        self.assertEqual(d["recall@20"], "upper")
        self.assertEqual(d["ndcg@20"], "lower")
        self.assertEqual(d["hit@10"], "upper")

    def test_default_table_and_fallback(self):
        self.assertEqual(default_direction("loss"), "lower")
        self.assertEqual(default_direction("rmse@10"), "lower")
        self.assertEqual(default_direction("ndcg@20"), "upper")
        self.assertEqual(default_direction("custom@5"), "upper")

    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            parse_metrics(["recall@20 middle"])


class BestTrackerTest(unittest.TestCase):
    def test_upper_and_lower_best(self):
        t = BestTracker([{"recall@20": "upper"}, "loss lower"])
        self.assertEqual(
            t.update({"recall@20": 0.1, "loss": 1.0}, 1),
            ["recall@20", "loss"],
        )
        self.assertEqual(
            t.update({"recall@20": 0.2, "loss": 0.8}, 2),
            ["recall@20", "loss"],
        )
        self.assertEqual(t.update({"recall@20": 0.15, "loss": 0.9}, 3), [])
        r = t.best_results()
        self.assertEqual(r["recall@20"]["epoch"], 2)
        self.assertEqual(r["loss"]["epoch"], 2)

    def test_per_metric_keeps_one_file_per_metric(self):
        t = BestTracker([{"recall@20": "upper"}, "ndcg@20 upper"])
        t.update({"recall@20": 0.1, "ndcg@20": 0.2}, 1)
        t.update({"recall@20": 0.2, "ndcg@20": 0.3}, 1)  # 同 epoch 双指标刷新
        files = dict(t.best_checkpoints())
        self.assertEqual(files["recall@20"], "recall@20-best-model.pt")
        self.assertEqual(files["ndcg@20"], "ndcg@20-best-model.pt")
        self.assertEqual(len(files), 2)

    def test_single_mode_only_primary(self):
        t = BestTracker(
            [{"recall@20": "upper"}, "ndcg@20 upper"],
            checkpoint_mode="single",
        )
        t.update({"recall@20": 0.1, "ndcg@20": 0.2}, 1)
        self.assertEqual([m for m, _ in t.best_checkpoints()], ["recall@20"])
        self.assertEqual(t.primary_metric, "recall@20")

    def test_best_results_full_snapshot(self):
        t = BestTracker(["recall@20"])
        t.update({"recall@20": 0.3, "ndcg@20": 0.5}, 7)
        r = t.best_results()["recall@20"]
        self.assertEqual(r["value"], 0.3)
        self.assertEqual(r["metrics"], {"recall@20": 0.3, "ndcg@20": 0.5})
        self.assertEqual(r["checkpoint"], "recall@20-best-model.pt")

    def test_safe_name(self):
        self.assertEqual(safe_checkpoint_name("recall@20"), "recall@20")
        self.assertNotIn("/", safe_checkpoint_name("a/b"))

    def test_update_raises_on_total_mismatch(self):
        """配置指标名与传入指标键完全不匹配时必须显式报错（防静默失效）。"""
        t = BestTracker([{"recall@20": "upper"}])
        with self.assertRaises(ValueError):
            t.update({"recall@10": 0.1}, 1)


class MetricKTest(unittest.TestCase):
    def test_metric_k_parsing(self):
        self.assertEqual(metric_k("recall@20"), 20)
        self.assertEqual(metric_k("ndcg@10"), 10)
        self.assertIsNone(metric_k("loss"))

    def test_eval_ks_from_metrics(self):
        ks = eval_ks_from_metrics(
            [{"recall@20": "upper"}, "ndcg@10 lower"], fallback_k=5)
        self.assertEqual(ks, [10, 20])
        # 无 @K 时回退 fallback_k
        self.assertEqual(eval_ks_from_metrics(["recall"], fallback_k=5), [5])

    def test_match_metric_values(self):
        res_by_k = {
            10: {"recall@10": 0.1, "ndcg@10": 0.2},
            20: {"recall@20": 0.3, "ndcg@20": 0.4},
        }
        self.assertEqual(
            match_metric_values(["recall@20", "ndcg@20"], res_by_k),
            {"recall@20": 0.3, "ndcg@20": 0.4},
        )
        # 裸名按前缀匹配
        self.assertEqual(
            match_metric_values(["recall"], res_by_k),
            {"recall": 0.1},
        )


if __name__ == "__main__":
    unittest.main()
