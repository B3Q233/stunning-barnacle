"""UBA uplift 纯算法层单测（CPU，无模型依赖）。

覆盖三类高风险点：
1. 分组背包 DP —— 与暴力枚举、与官方 DPA.py::pack5 口径逐例对齐；
2. A'³ 三跳路径的稀疏化简 —— 与"朴素构造 A' 再算 A'³" 的稠密结果逐元素比对；
3. 目标用户选择与三种分配策略 —— 约束、预算、确定性。
"""
import itertools
import random
import unittest

import numpy as np

from attacks.uba.uplift import (
    accessible_template_users,
    allocate,
    cooccurrence_neighbors,
    dp_allocate,
    interaction_sets,
    item_interaction_sets,
    select_target_users,
    three_hop_path_effect,
)


def official_pack5(weight, value, lengths, capacity):
    """官方代码 DPA.py::pack5 的等价移植（0/1 分组背包，返回 dp[capacity]）。

    官方实现：dp 初始化为 0（允许预算用不满），外层遍历"组"（目标用户），
    内层倒序遍历容量（保证每组最多选一个档位）。此处只保留数值部分，去掉
    官方用于回溯的 dict 记录（回溯正确性由 dp_allocate 的 T* 可行性断言覆盖）。
    """
    dp = [0 for _ in range(capacity + 1)]
    for i in range(1, len(weight) + 1):
        for j in reversed(range(1, capacity + 1)):
            for k in range(lengths[i - 1]):
                if j - weight[i - 1][k] >= 0:
                    dp[j] = max(dp[j], dp[j - weight[i - 1][k]] + value[i - 1][k])
    return dp[capacity]


def brute_force_best(values, budget, max_per_user):
    """暴力枚举所有可行分配，返回最优目标值。"""
    n = values.shape[0]
    best = 0.0
    for t in itertools.product(range(max_per_user + 1), repeat=n):
        if sum(t) <= budget:
            best = max(best, sum(values[i, t[i]] for i in range(n)))
    return best


def dense_three_hop(user_items, num_users, num_items, target_item,
                    fake_profiles):
    """朴素实现：显式构造 A' 再做三次矩阵乘法，返回 (A'³)[u, i] 全列。

    只用于单测对照（ml100k 规模下会占满内存，生产路径走稀疏化简）。
    """
    n_fake = len(fake_profiles)
    n_rows = num_users + n_fake
    d = np.zeros((n_rows, num_items), dtype=np.float64)
    for u, items in user_items.items():
        for i in items:
            d[u, i] = 1.0
    for r, items in enumerate(fake_profiles):
        for i in items:
            d[num_users + r, i] = 1.0
    a = np.zeros((n_rows + num_items, n_rows + num_items), dtype=np.float64)
    a[:n_rows, n_rows:] = d
    a[n_rows:, :n_rows] = d.T
    a3 = a.dot(a).dot(a)
    return a3[:num_users, n_rows + target_item]


class DpAllocateTest(unittest.TestCase):

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        values = rng.random((4, 4))  # H=3
        # Y[:, 0] 是"不投毒"的基线价值；论文 Eq.2 的求和含 t_u=0 项，它是常数，
        # 不影响 argmax。这里显式置 0，与官方代码（x[:,0] 恒为 0）口径一致。
        values[:, 0] = 0.0
        t_star, best = dp_allocate(values, budget=5, max_per_user=3)
        self.assertAlmostEqual(best, brute_force_best(values, 5, 3))
        self.assertLessEqual(int(t_star.sum()), 5)
        self.assertTrue(all(0 <= t <= 3 for t in t_star))
        # 返回的 T* 必须自己就能达到报告的最优值（回溯正确性）
        self.assertAlmostEqual(
            sum(values[i, t_star[i]] for i in range(len(t_star))), best)

    def test_matches_official_pack5(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            n_users = int(rng.integers(1, 6))
            max_per_user = int(rng.integers(1, 5))
            budget = int(rng.integers(0, 12))
            values = rng.random((n_users, max_per_user + 1))
            values[:, 0] = 0.0
            # 官方 pack5 只把"非零价值档位"放进候选（跳过的用户等价于 t=0 价值 0）
            weight, value, lengths = [], [], []
            for u in range(n_users):
                w_row, v_row = [], []
                for t in range(1, max_per_user + 1):
                    if values[u, t] != 0:
                        w_row.append(t)
                        v_row.append(float(values[u, t]))
                weight.append(w_row)
                value.append(v_row)
                lengths.append(len(w_row))
            ours = dp_allocate(values, budget, max_per_user)[1]
            theirs = official_pack5(weight, value, lengths, budget)
            self.assertAlmostEqual(ours, theirs, places=9)
            # 同时与暴力枚举对齐，确认"与官方一致"不是因为两者都错
            self.assertAlmostEqual(
                ours, brute_force_best(values, budget, max_per_user), places=9)

    def test_zero_column_is_skip_option(self):
        """Y[:, 0] 是"不给该用户投预算"的基线选项，必须参与比较。

        论文 Eq.2 的求和对每个用户取 t_u ∈ {0..H} 中的**一个**值；官方代码把
        x[:,0] 恒置 0（模拟实验只填 t=1..H），本实现允许 Y[:,0] 有值（三跳路径在
        干净图上的计数 / 代理模型在无假用户时的命中率）。因此：
        - 基线与被攻击价值相等 → 并列时选 t=0（省预算）；
        - 基线已高于被攻击价值 → 该用户不应再分配预算（否则总命中数下降）。
        """
        # 用户 1 的基线命中（0.9）已高于被攻击后的命中（0.2）→ 不应再投放预算
        values = np.array([[0.0, 0.3], [0.9, 0.2]])
        t_star, best = dp_allocate(values, budget=2, max_per_user=1)
        self.assertEqual(list(t_star), [1, 0])
        self.assertAlmostEqual(best, 0.3 + 0.9)
        # 基线置 0（官方口径 x[:,0]≡0）时，同一份数据会给出不同分配 —— 说明该列
        # 不是常数平移，而是"跳过该用户"的显式选项
        plain = np.array([[0.0, 0.3], [0.0, 0.2]])
        t_plain, best_plain = dp_allocate(plain, budget=2, max_per_user=1)
        self.assertEqual(list(t_plain), [1, 1])
        self.assertAlmostEqual(best_plain, 0.5)
        # 并列（基线 == 被攻击价值）时取 t=0，省下的预算留给其他用户
        tie = np.array([[0.0, 0.5], [0.5, 0.5]])
        t_tie, best_tie = dp_allocate(tie, budget=2, max_per_user=1)
        self.assertEqual(list(t_tie), [1, 0])
        self.assertAlmostEqual(best_tie, 1.0)

    def test_zero_budget_and_degenerate(self):
        values = np.zeros((3, 4))
        t_star, best = dp_allocate(values, budget=0, max_per_user=3)
        self.assertEqual(list(t_star), [0, 0, 0])
        self.assertEqual(best, 0.0)
        # 价值全 0 → 并列时取更小的 t（省预算）
        t_star2, best2 = dp_allocate(values, budget=10, max_per_user=3)
        self.assertEqual(list(t_star2), [0, 0, 0])
        self.assertEqual(best2, 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dp_allocate(np.zeros((3, 3)), budget=2, max_per_user=4)
        with self.assertRaises(ValueError):
            dp_allocate(np.zeros(3), budget=2, max_per_user=1)


class ThreeHopPathTest(unittest.TestCase):

    NUM_USERS = 6
    NUM_ITEMS = 8
    TARGET = 3
    USER_ITEMS = {
        0: {0, 1, 3, 5},
        1: {1, 2, 4},
        2: {0, 3, 4, 6},
        3: {2, 5, 7},
        4: {0, 1, 2, 7},
        5: {3, 4, 6},
    }
    USERS = [0, 1, 4]
    FIXED_PROFILE = [1, 5]  # 固定假画像，便于与稠密对照逐元素比对

    _META = {
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "train_pairs": [(u, i) for u, items in USER_ITEMS.items()
                        for i in items],
        "test_pairs": [],
    }

    def _fixed_profile_fn(self, _uid):
        return list(self.FIXED_PROFILE)

    def test_zero_fake_matches_dense(self):
        effect = three_hop_path_effect(
            self.USER_ITEMS, self.USERS, self.TARGET, max_per_user=0,
            num_items=self.NUM_ITEMS, num_users=self.NUM_USERS)
        dense = dense_three_hop(self.USER_ITEMS, self.NUM_USERS,
                                self.NUM_ITEMS, self.TARGET, [])
        expected = np.array([dense[u] for u in self.USERS])
        np.testing.assert_allclose(effect[:, 0], expected, rtol=1e-9)

    def test_with_fake_users_matches_dense(self):
        effect = three_hop_path_effect(
            self.USER_ITEMS, self.USERS, self.TARGET, max_per_user=2,
            num_items=self.NUM_ITEMS, num_users=self.NUM_USERS,
            profile_fn=self._fixed_profile_fn)
        for t in (1, 2):
            fake = [[*self.FIXED_PROFILE, self.TARGET]] * (len(self.USERS) * t)
            dense = dense_three_hop(self.USER_ITEMS, self.NUM_USERS,
                                    self.NUM_ITEMS, self.TARGET, fake)
            expected = np.array([dense[u] for u in self.USERS])
            np.testing.assert_allclose(effect[:, t], expected, rtol=1e-9)

    def test_alpha_beta_transform(self):
        base = three_hop_path_effect(
            self.USER_ITEMS, self.USERS, self.TARGET, max_per_user=0,
            num_items=self.NUM_ITEMS, num_users=self.NUM_USERS)
        scaled = three_hop_path_effect(
            self.USER_ITEMS, self.USERS, self.TARGET, max_per_user=0,
            num_items=self.NUM_ITEMS, num_users=self.NUM_USERS,
            alpha=0.5, beta=2.0)
        np.testing.assert_allclose(scaled, 0.5 * base ** 2, rtol=1e-9)

    def test_default_profile_uses_template_user_items(self):
        meta = self._META
        effect = three_hop_path_effect(
            self.USER_ITEMS, self.USERS, self.TARGET, max_per_user=1,
            num_items=self.NUM_ITEMS, num_users=self.NUM_USERS)
        fake = [sorted(self.USER_ITEMS[u]) + [self.TARGET] for u in self.USERS]
        dense = dense_three_hop(self.USER_ITEMS, self.NUM_USERS,
                                self.NUM_ITEMS, self.TARGET, fake)
        expected = np.array([dense[u] for u in self.USERS])
        np.testing.assert_allclose(effect[:, 1], expected, rtol=1e-9)
        self.assertEqual(meta["num_users"], self.NUM_USERS)


class TargetUserSelectionTest(unittest.TestCase):

    NUM_USERS = 12
    NUM_ITEMS = 7
    TARGET = 2
    USER_ITEMS = {
        0: {0, 2, 5},          # 已交互目标 → 必须被排除
        1: {0, 5},
        2: {0, 1, 5, 6},
        3: {1},
        4: {0, 6, 3},          # 类别交互数偏多，用于验证上界过滤
        5: {5, 6},
        6: {3, 4},
        7: {0, 1, 4, 5, 6},    # 类别交互数 5（> max=3）→ 排除
        8: {1, 6},
        9: {4},
        10: {3},
        11: {0, 5},
    }

    def _meta(self):
        return {
            "num_users": self.NUM_USERS,
            "num_items": self.NUM_ITEMS,
            "train_pairs": [(u, i) for u, items in self.USER_ITEMS.items()
                            for i in items],
            "test_pairs": [],
        }

    def test_cooccurrence_respects_constraints(self):
        cfg = {"strategy": "cooccurrence", "count": 3, "category_size": 4,
               "max_category_interactions": 3}
        out = select_target_users(self._meta(), self.TARGET, cfg, seed=42)
        users = out["users"]
        self.assertLessEqual(len(users), 3)
        self.assertTrue(users)
        category = set(out["category_items"])
        self.assertIn(self.TARGET, category)
        for u in users:
            self.assertNotIn(self.TARGET, self.USER_ITEMS[u])
            cnt = len(self.USER_ITEMS[u] & category)
            self.assertGreater(cnt, 0)
            self.assertLess(cnt, 3)

    def test_cooccurrence_is_deterministic(self):
        cfg = {"strategy": "cooccurrence", "count": 4, "category_size": 4,
               "max_category_interactions": 3}
        a = select_target_users(self._meta(), self.TARGET, cfg, seed=7)
        b = select_target_users(self._meta(), self.TARGET, cfg, seed=7)
        c = select_target_users(self._meta(), self.TARGET, cfg, seed=8)
        self.assertEqual(a["users"], b["users"])
        self.assertEqual(a["n_candidates"], b["n_candidates"])
        self.assertEqual(c["n_candidates"], a["n_candidates"])

    def test_random_strategy_excludes_target_interactors(self):
        cfg = {"strategy": "random", "count": 5}
        out = select_target_users(self._meta(), self.TARGET, cfg, seed=1)
        self.assertEqual(len(out["users"]), 5)
        for u in out["users"]:
            self.assertNotIn(self.TARGET, self.USER_ITEMS[u])

    def test_specified_strategy(self):
        cfg = {"strategy": "specified", "count": 2, "ids": [3, 9]}
        out = select_target_users(self._meta(), self.TARGET, cfg, seed=0)
        self.assertEqual(out["users"], [3, 9])
        with self.assertRaises(ValueError):
            select_target_users(self._meta(), self.TARGET,
                                {"strategy": "specified", "ids": []}, seed=0)
        with self.assertRaises(ValueError):
            select_target_users(self._meta(), self.TARGET,
                                {"strategy": "specified", "ids": [99]}, seed=0)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            select_target_users(self._meta(), self.TARGET,
                                {"strategy": "nope"}, seed=0)

    def test_cooccurrence_neighbors_ranking(self):
        user_items = interaction_sets(self._meta()["train_pairs"])
        neighbors = cooccurrence_neighbors(user_items, self.TARGET, size=3)
        self.assertEqual(neighbors[0], self.TARGET)
        self.assertEqual(len(neighbors), 3)
        item_users = item_interaction_sets(self._meta()["train_pairs"])
        # 共现次数单调不增（排序口径）
        counts = [len(item_users[self.TARGET] & item_users[j])
                  for j in neighbors[1:]]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_accessible_template_users_ratio(self):
        pool = accessible_template_users(self._meta(), ratio=1.0)
        self.assertEqual(len(pool), self.NUM_USERS)
        sub = accessible_template_users(self._meta(), ratio=0.5, seed=3)
        self.assertEqual(len(sub), 6)
        self.assertTrue(set(sub) <= set(pool))
        with self.assertRaises(ValueError):
            accessible_template_users(self._meta(), ratio=0.0)


class AllocationStrategyTest(unittest.TestCase):

    VALUES = np.array([
        [0.0, 0.1, 0.5, 0.5],
        [0.0, 0.4, 0.4, 0.4],
        [0.0, 0.0, 0.9, 0.9],
    ])
    USERS = [10, 20, 30]

    def test_uba_matches_dp_and_uses_dp_allocation(self):
        out = allocate(self.VALUES, self.USERS, "uba", budget=4,
                       max_per_user=3, seed=42)
        t_star, best = dp_allocate(self.VALUES, 4, 3)
        self.assertEqual([out["allocation"][u] for u in self.USERS],
                         [int(t) for t in t_star])
        self.assertAlmostEqual(out["estimated_value"], best)
        self.assertEqual(out["num_fake_users"], int(t_star.sum()))
        self.assertEqual(out["template_users"],
                         [u for u, t in zip(self.USERS, t_star)
                          for _ in range(int(t))])
        self.assertLessEqual(out["num_fake_users"], 4)

    def test_uniform_target_respects_budget_and_cap(self):
        out = allocate(self.VALUES, self.USERS, "uniform_target", budget=7,
                       max_per_user=2, seed=42)
        self.assertLessEqual(sum(out["allocation"].values()), 7)
        self.assertTrue(all(t <= 2 for t in out["allocation"].values()))
        self.assertEqual(out["num_fake_users"], sum(out["allocation"].values()))

    def test_uniform_target_spreads_evenly(self):
        out = allocate(self.VALUES, self.USERS, "uniform_target", budget=6,
                       max_per_user=5, seed=42)
        self.assertEqual(sorted(out["allocation"].values()), [2, 2, 2])

    def test_random_all_uses_template_pool_with_replacement(self):
        pool = [5, 6, 7]
        out = allocate(self.VALUES, self.USERS, "random_all", budget=8,
                       max_per_user=3, seed=42, template_pool=pool)
        self.assertEqual(len(out["template_users"]), 8)
        self.assertTrue(set(out["template_users"]) <= set(pool))
        self.assertEqual(out["allocation"], {u: 0 for u in self.USERS})
        with self.assertRaises(ValueError):
            allocate(self.VALUES, self.USERS, "random_all", budget=3,
                     max_per_user=3, seed=1, template_pool=[])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            allocate(self.VALUES, self.USERS, "nope", budget=1,
                     max_per_user=3, seed=1)
        with self.assertRaises(ValueError):
            allocate(self.VALUES, self.USERS, "uba", budget=-1,
                     max_per_user=3, seed=1)
        with self.assertRaises(ValueError):
            allocate(self.VALUES, [], "uba", budget=1, max_per_user=3, seed=1)

    def test_seed_reproducibility(self):
        a = allocate(self.VALUES, self.USERS, "random_all", budget=6,
                     max_per_user=3, seed=11, template_pool=[1, 2, 3, 4])
        b = allocate(self.VALUES, self.USERS, "random_all", budget=6,
                     max_per_user=3, seed=11, template_pool=[1, 2, 3, 4])
        self.assertEqual(a["template_users"], b["template_users"])
        c = allocate(self.VALUES, self.USERS, "random_all", budget=6,
                     max_per_user=3, seed=12, template_pool=[1, 2, 3, 4])
        self.assertNotEqual(a["template_users"], c["template_users"])

    def test_rng_argument_is_used(self):
        rng = random.Random(99)
        out = allocate(self.VALUES, self.USERS, "uniform_target", budget=7,
                       max_per_user=2, seed=0, rng=rng)
        self.assertEqual(out["strategy"], "uniform_target")


if __name__ == "__main__":
    unittest.main()
