# -*- coding: utf-8 -*-
"""pre 编排层的单元测试（stdlib unittest，不新增第三方依赖）。

只测纯逻辑（不训练模型、不调用攻击）：
    1) 目标物品选择：按交互数排序、门槛过滤、排名从 1 开始
    2) 交互退化：删除条数 = floor(r × n)、同种子稳定、缓存命中不重算
    3) 距离与恢复率：定义与边界（d_deg = 0）
    4) Procrustes：正交矩阵能把"旋转过"的矩阵对齐回参考帧
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.analysis.align import procrustes  # noqa: E402
from pre.analysis.distances import (cosine_distance, distances,  # noqa: E402
                                    recovery_rate)
from pre.analysis.exposure import select_targets_by_exposure  # noqa: E402
from pre.runners.degrade import build_degraded_meta  # noqa: E402
from pre.runners.targets import select_targets_popularity  # noqa: E402


def _fake_meta():
    """构造一个小型 meta：3 个物品，交互数分别 10 / 5 / 2。"""
    pairs = []
    for u in range(10):
        pairs.append((u, 0))
    for u in range(5):
        pairs.append((u, 1))
    for u in range(2):
        pairs.append((u, 2))
    user_items = {}
    for u, i in pairs:
        user_items.setdefault(u, set()).add(i)
    return {"num_users": 10, "num_items": 3, "train_pairs": pairs,
            "test_pairs": list(pairs), "user_items": user_items}


class TestTargets(unittest.TestCase):
    def test_top_popularity_order_and_rank(self):
        picked = select_targets_popularity(_fake_meta(), num_items=2,
                                           min_interactions=1, seed=42)
        self.assertEqual([p[0] for p in picked], [0, 1])
        self.assertEqual(picked[0][1], 10)

    def test_min_interactions_filter(self):
        picked = select_targets_popularity(_fake_meta(), num_items=2,
                                           min_interactions=5, seed=42)
        self.assertEqual([p[0] for p in picked], [0, 1])
        with self.assertRaises(ValueError):
            select_targets_popularity(_fake_meta(), num_items=3,
                                      min_interactions=5, seed=42)

    def test_top_exposure_selection(self):
        """按原生曝光排序（而不是交互数）：曝光高的优先，同分看交互数。"""
        exposure = {0: {"hr@k": 0.1}, 1: {"hr@k": 0.9}, 2: {"hr@k": 0.5}}
        counts = {0: 10, 1: 5, 2: 2}
        picked = select_targets_by_exposure(exposure, counts, num_items=2,
                                            min_interactions=1)
        self.assertEqual([p[0] for p in picked], [1, 2])
        self.assertAlmostEqual(picked[0][2], 0.9)


class TestDegrade(unittest.TestCase):
    def test_delete_count_is_floor(self):
        meta = _fake_meta()
        for ratio, expected in ((0.1, 1), (0.2, 2), (0.3, 3), (0.5, 5),
                                (0.8, 8), (0.9, 9)):
            _, info = build_degraded_meta(meta, 0, ratio, 42)
            self.assertEqual(info["deleted_interaction_count"], expected)
            self.assertEqual(info["remaining_interaction_count"], 10 - expected)

    def test_delete_is_deterministic_and_meta_consistent(self):
        meta = _fake_meta()
        m1, i1 = build_degraded_meta(meta, 0, 0.5, 42)
        m2, i2 = build_degraded_meta(meta, 0, 0.5, 42)
        self.assertEqual(i1["deleted_interactions"], i2["deleted_interactions"])
        self.assertEqual(len(m1["train_pairs"]), len(meta["train_pairs"]) - 5)
        # 未涉及的物品交互数不变
        self.assertEqual(sum(1 for _, i in m1["train_pairs"] if i == 1), 5)
        # user_items 与 train_pairs 保持一致（攻击模块依赖 user_items 做负采样）
        total = sum(len(v) for v in m1["user_items"].values())
        self.assertEqual(total, len(m1["train_pairs"]))

    def test_deletion_is_nested_across_ratios(self):
        """v2 修正的核心不变量：D(r1) ⊆ D(r2) 当 r1 < r2。

        早期实现按比例独立抽样，导致 0.9 删掉的集合不包含 0.1 删掉的集合，
        比例轴失去阶梯含义。这个用例锁死嵌套语义。
        """
        meta = _fake_meta()
        ratios = [0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
        prev: set = set()
        for r in ratios:
            _, info = build_degraded_meta(meta, 0, r, 42)
            cur = {tuple(p) for p in info["deleted_interactions"]}
            self.assertTrue(prev.issubset(cur),
                            msg=f"ratio={r} 的删除集合未包含前一级")
            self.assertEqual(info["deletion_scheme"], "nested")
            prev = cur
        # 最大比例删除 floor(0.9 × 10) = 9 条（保留 1 条），删除集合最大
        self.assertEqual(len(prev), 9)


class TestDistances(unittest.TestCase):
    def test_cosine_and_l2(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(cosine_distance(a, b), 1.0, places=6)
        self.assertAlmostEqual(distances(a, a)["cosine"], 0.0, places=6)
        self.assertAlmostEqual(distances(a, b)["l2"], 2 ** 0.5, places=6)

    def test_recovery_rate(self):
        self.assertAlmostEqual(recovery_rate(0.2, 0.1), 0.5, places=6)
        self.assertAlmostEqual(recovery_rate(0.2, 0.2), 0.0, places=6)
        self.assertAlmostEqual(recovery_rate(0.2, 0.3), -0.5, places=6)
        self.assertEqual(recovery_rate(0.0, 0.5), 0.0)


class TestProcrustes(unittest.TestCase):
    def test_recovers_rotation(self):
        torch.manual_seed(0)
        ref = torch.randn(50, 8)
        q, _ = torch.linalg.qr(torch.randn(8, 8))
        cond = ref @ q.T                      # 条件模型 = 参考帧旋转 q
        mask = torch.ones(50, dtype=torch.bool)
        r = procrustes(ref, cond, mask)
        err = (ref - cond @ r).norm() / ref.norm()
        self.assertLess(float(err), 1e-5)


if __name__ == "__main__":
    unittest.main()
