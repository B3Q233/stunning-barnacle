"""UBA 数据阶段（generate/data）单测：合成 meta 上跑完整注入流程（CPU）。

覆盖：
- 注入数量硬断言（after − before == Σ|profile|）
- profiles.json / stats.json schema 与假用户 id 起始位置
- 预算与分配策略语义（uba / uniform_target / random_all）
- 处理效应缓存落盘与复用
"""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from attacks.uba import generate as uba_generate


def _synthetic_meta():
    """12 用户 × 8 物品的合成数据：目标物品 0 被一部分用户交互，便于选人。"""
    user_items = {
        0: {0, 1, 2},
        1: {1, 2, 3},
        2: {1, 3},
        3: {2, 4},
        4: {3, 4, 5},
        5: {4, 5, 6},
        6: {1, 5},
        7: {2, 6, 7},
        8: {5, 7},
        9: {6, 7},
        10: {1, 2, 4},
        11: {3, 6},
    }
    train_pairs = [(u, i) for u, items in user_items.items() for i in sorted(items)]
    test_pairs = [(u, 0) for u in (1, 2, 3, 4, 5)]
    return {
        "num_users": 12,
        "num_items": 8,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "user_items": {u: set(items) for u, items in user_items.items()},
    }


def _config(strategy: str = "uba", filler_source: str = "template_user",
            budget: int = 5) -> dict:
    return {
        "dataset": "uba-unittest",
        "seed": 7,
        "k": 2,
        "run_tag": "uba-unit",
        "model": {"name": "lightgcn", "overrides": {}},
        "classification": {"popular_ratio": 0.05, "medium_ratio": 0.4},
        "attack": {
            "name": "uba",
            "num_fake_users": budget,
            "ratio": None,
            "filler_size": 3,
            "target_items": {"strategy": "specified", "category": "cold",
                             "count": 1, "ids": [0]},
            "uba": {
                "treatment": {"method": "path", "max_per_user": 2,
                              "repeats": 1, "hit_k": 2, "alpha": 1.0, "beta": 1.0},
                "allocation": {"strategy": strategy},
                "target_users": {"strategy": "cooccurrence", "count": 4,
                                 "category_size": 3,
                                 "max_category_interactions": 4,
                                 "ids": [], "accessible_ratio": 1.0},
                "profile": {"filler_source": filler_source},
            },
        },
        "output": {"dir": "attacks/uba/outputs"},
    }


class GenerateTestCase(unittest.TestCase):
    """公共夹具：meta 写临时目录，处理效应缓存重定向到临时目录（不污染仓库）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.meta_path = self.tmp / "meta.pkl"
        self.meta = _synthetic_meta()
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)
        self.estimate_dir = self.tmp / "estimate"
        patcher = mock.patch.object(
            uba_generate, "estimate_dir",
            lambda config, model_name: self.estimate_dir / model_name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, cfg=None, tag="uba-unit"):
        cfg = cfg or _config()
        cfg["run_tag"] = tag
        out = self.tmp / "poisoned" / tag
        stats = uba_generate.main(cfg, raw_meta=self.meta_path, out_dir=out)
        return cfg, stats, out


class InjectionTest(GenerateTestCase):

    def test_injection_counts_and_fake_uid_range(self):
        _, stats, out = self._run()
        poisoned = uba_generate.load_meta(out / "meta.pkl")
        profiles = json.loads((out / "profiles.json").read_text(encoding="utf-8"))

        before = len(self.meta["train_pairs"])
        after = len(poisoned["train_pairs"])
        injected = sum(len(p["items"]) for p in profiles)
        self.assertEqual(after, before + injected)
        self.assertEqual(poisoned["num_users"], self.meta["num_users"] + len(profiles))
        self.assertEqual(stats["injected_pairs"], injected)
        self.assertEqual(
            stats["train_pairs_after"] - stats["train_pairs_before"], injected)

        fakes = sorted(p["fake_user"] for p in profiles)
        self.assertEqual(fakes, list(range(len(profiles))))
        for p in profiles:
            fake_uid = self.meta["num_users"] + p["fake_user"]
            self.assertLess(fake_uid, poisoned["num_users"])
            self.assertEqual(poisoned["user_items"][fake_uid], set(p["items"]))
            self.assertTrue(set(p["items"]) <= set(range(self.meta["num_items"])))

    def test_every_profile_contains_target_once_and_no_duplicates(self):
        _, stats, out = self._run()
        profiles = json.loads((out / "profiles.json").read_text(encoding="utf-8"))
        target = stats["targets"][0]["item_id"]
        self.assertEqual(target, 0)
        for p in profiles:
            self.assertEqual(p["items"].count(target), 1)
            self.assertEqual(len(p["items"]), len(set(p["items"])))
            self.assertEqual(p["target"], target)

    def test_products_written(self):
        _, stats, out = self._run()
        for name in ("meta.pkl", "profiles.json", "stats.json", "config.yaml"):
            self.assertTrue((out / name).exists(), name)
        self.assertTrue((out.parent / "latest.json").exists())
        latest = json.loads((out.parent / "latest.json").read_text(encoding="utf-8"))
        self.assertEqual(latest["run_tag"], "uba-unit")
        self.assertEqual(stats["attack"], "uba")
        self.assertEqual(stats["allocation"]["strategy"], "uba")
        for key in ("target_users", "targets", "treatment", "allocation",
                    "num_fake_users", "budget", "max_per_user"):
            self.assertIn(key, stats)

    def test_target_users_do_not_interact_target(self):
        _, stats, _ = self._run()
        for u in stats["target_users"]["ids"]:
            self.assertNotIn(0, self.meta["user_items"][u])
        self.assertLessEqual(stats["target_users"]["count"], 4)

    def test_fillers_do_not_exceed_filler_size(self):
        _, stats, out = self._run()
        profiles = json.loads((out / "profiles.json").read_text(encoding="utf-8"))
        for p in profiles:
            fillers = [i for i in p["items"] if i != p["target"]]
            self.assertLessEqual(len(fillers), stats["filler_size"])
            # filler 必须来自模板用户交互 ∪ 全量物品（模板不足时用流行池补齐）
            self.assertTrue(set(fillers) <= set(range(self.meta["num_items"])))


class AllocationStrategyTest(GenerateTestCase):

    def test_uba_respects_budget_and_cap(self):
        _, stats, out = self._run(_config(strategy="uba", budget=5))
        self.assertLessEqual(stats["num_fake_users"], 5)
        self.assertLessEqual(
            max(int(t) for t in stats["allocation"]["histogram"]),
            stats["max_per_user"])
        profiles = json.loads((out / "profiles.json").read_text(encoding="utf-8"))
        self.assertEqual(len(profiles), stats["num_fake_users"])

    def test_uniform_target_distribution(self):
        _, stats, _ = self._run(_config(strategy="uniform_target", budget=8))
        histogram = stats["allocation"]["histogram"]
        total = sum(int(t) * c for t, c in histogram.items())
        self.assertEqual(total, stats["num_fake_users"])
        # 8 预算 / 4 个目标用户、H=2 → 每个目标用户 2 个假用户
        self.assertEqual(histogram, {"2": 4})

    def test_random_all_uses_accessible_pool(self):
        _, stats, out = self._run(_config(strategy="random_all", budget=6))
        profiles = json.loads((out / "profiles.json").read_text(encoding="utf-8"))
        self.assertEqual(len(profiles), 6)
        for p in profiles:
            self.assertIn(p["template_user"], range(self.meta["num_users"]))
        # random_all 不针对目标用户：所有目标用户的分配数都是 0
        self.assertEqual(stats["allocation"]["histogram"], {"0": 4})

    def test_invalid_budget_raises(self):
        with self.assertRaises(ValueError):
            self._run(_config(budget=0))

    def test_multi_target_raises(self):
        cfg = _config()
        cfg["attack"]["target_items"]["count"] = 2
        with self.assertRaises(ValueError):
            self._run(cfg)


class SurrogateEstimateTest(GenerateTestCase):
    """surrogate 支路：真实训练一个小 MF 代理，验证 estimate → data 的闭环（CPU）。"""

    def _surrogate_cfg(self, **treatment_overrides) -> dict:
        cfg = _config(budget=3)
        treatment = {"method": "surrogate", "max_per_user": 1, "repeats": 2,
                     "hit_k": 2, "alpha": 1.0, "beta": 1.0}
        treatment.update(treatment_overrides)
        cfg["attack"]["uba"]["treatment"] = treatment
        cfg["training"] = {"device": "cpu"}
        cfg["surrogate"] = {
            "enabled": True,
            "name": "mf",
            "checkpoint": None,
            "training": {"epochs": 1, "lr": 0.001, "weight_decay": 0.0001,
                         "batch_size": 8, "eval_every": 1},
        }
        return cfg

    def test_estimate_writes_cache_and_generate_reads_it(self):
        from attacks.uba.estimate import main as estimate_main

        cfg = self._surrogate_cfg()
        payload = estimate_main(cfg, meta=self.meta)
        self.assertEqual(payload["method"], "surrogate")
        self.assertEqual(payload["effect"].shape[0], 4)   # |U_t| = 4
        self.assertEqual(payload["effect"].shape[1], 2)   # H + 1 = 2
        self.assertEqual(len(payload["per_repeat_hits"]), 2)
        cached = list(self.estimate_dir.rglob("item0_*surrogate*.json"))
        self.assertEqual(len(cached), 1, cached)

        # data 阶段应命中 estimate 写下的缓存（不再训练代理）
        cfg2 = self._surrogate_cfg()
        cfg2["run_tag"] = "uba-surrogate"
        stats = uba_generate.main(cfg2, raw_meta=self.meta_path,
                                  out_dir=self.tmp / "poisoned" / "uba-surrogate")
        self.assertEqual(stats["treatment"]["method"], "surrogate")
        self.assertLessEqual(stats["num_fake_users"], 3)

    def test_surrogate_requires_enabled_flag(self):
        from attacks.uba.estimate import main as estimate_main

        cfg = self._surrogate_cfg()
        cfg["surrogate"]["enabled"] = False
        with self.assertRaises(ValueError):
            estimate_main(cfg, meta=self.meta)


class EffectCacheTest(GenerateTestCase):

    def test_cache_written_and_reused(self):
        cfg, _, out = self._run(tag="first")
        cached = list(self.estimate_dir.rglob("item0_*.json"))
        self.assertEqual(len(cached), 1, cached)
        payload = json.loads(cached[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["method"], "path")
        self.assertEqual(len(payload["effect"]), len(payload["target_users"]))
        self.assertEqual(
            len(payload["effect"][0]),
            cfg["attack"]["uba"]["treatment"]["max_per_user"] + 1)

        # 第二次运行命中缓存：同一目标物品的结果保持一致
        _, stats2, out2 = self._run(tag="second")
        first = json.loads((out / "stats.json").read_text(encoding="utf-8"))
        self.assertEqual(first["num_fake_users"], stats2["num_fake_users"])
        self.assertEqual(first["injected_pairs"], stats2["injected_pairs"])
        self.assertTrue((out2 / "stats.json").exists())

    def test_unknown_method_raises(self):
        cfg = _config()
        cfg["attack"]["uba"]["treatment"]["method"] = "magic"
        with self.assertRaises(ValueError):
            self._run(cfg)


if __name__ == "__main__":
    unittest.main()
