"""指标注册表测试：新增指标自动可计算/输出。"""
from __future__ import annotations

import unittest

from evaluation.metrics_registry import (
    METRICS,
    compute_metrics,
    register_metric,
)


class MetricRegistryTest(unittest.TestCase):

    def tearDown(self):
        METRICS.pop("fake_metric@5", None)

    def test_register_and_compute(self):
        def fake(scores, **kwargs):
            return float(sum(value for row in scores for value in row))

        register_metric("fake_metric@5", fake)
        self.assertIn("fake_metric@5", METRICS)
        out = compute_metrics(["fake_metric@5"], [[1.0, 2.0]])
        self.assertEqual(out["fake_metric@5"], 3.0)

    def test_duplicate_register_raises(self):
        register_metric("dup@1", lambda scores, **kw: 0.0)
        with self.assertRaises(ValueError):
            register_metric("dup@1", lambda scores, **kw: 0.0)
        METRICS.pop("dup@1", None)


if __name__ == "__main__":
    unittest.main()
