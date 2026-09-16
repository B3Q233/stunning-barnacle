# -*- coding: utf-8 -*-
"""pre 扩展约定测试：注册表、自检、适配器默认超参、攻击登记校验。

这些用例锁定"接入新模型/新攻击"这条路径的契约，避免以后有人为了加一个模型
去改编排核心（那正是这套机制要避免的事）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

TPA_ROOT = Path(__file__).resolve().parents[2]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from pre.runners.common import build_training_config  # noqa: E402
from pre.runners.model_adapters import (ModelAdapter, available_models,  # noqa: E402
                                        check_scaffold, get_adapter,
                                        register_model)
from pre.runners.run_attack import (ATTACK_SPECS, available_attacks,  # noqa: E402
                                    register_attack)


class TestModelAdapterRegistry(unittest.TestCase):
    def test_builtin_adapters_cover_repo_models(self):
        for name in ("lightgcn", "mf", "wmf"):
            self.assertIn(name, available_models())

    def test_unknown_model_error_is_actionable(self):
        with self.assertRaises(KeyError) as ctx:
            get_adapter("definitely_not_a_model")
        msg = str(ctx.exception)
        self.assertIn("register_model", msg)       # 报错里带可执行修复代码
        self.assertIn("已支持", msg)

    def test_register_and_overwrite(self):
        ad = ModelAdapter(name="tmp_model", dataset="x:Y", loop="pairwise")
        register_model(ad)
        self.assertEqual(get_adapter("tmp_model").loop, "pairwise")
        with self.assertRaises(ValueError):
            register_model(ad)                      # 重名默认报错
        register_model(ad, overwrite=True)          # 显式覆盖允许

    def test_bad_loop_rejected(self):
        with self.assertRaises(ValueError):
            register_model(ModelAdapter(name="x", dataset="a:B", loop="nope"))

    def test_adapter_defaults_flow_into_training_config(self):
        cfg = {"pre": {"training": {"device": "cpu", "lr": 1e-3}},
               "model": {"overrides": {}}}
        wmf = build_training_config(cfg, "wmf")
        self.assertEqual(wmf.get("optimizer"), "als")     # 来自适配器
        self.assertEqual(wmf.get("factors"), 100)
        self.assertAlmostEqual(wmf.get("alpha"), 40.0)
        # 用户显式 overrides 优先级最高
        cfg2 = {"pre": {"training": {"device": "cpu"}},
                "model": {"overrides": {"factors": 32}}}
        self.assertEqual(build_training_config(cfg2, "wmf").get("factors"), 32)

    def test_check_scaffold_reports_ok(self):
        report = {r["model"]: r for r in check_scaffold(["lightgcn", "mf", "wmf"])}
        for name in ("lightgcn", "mf", "wmf"):
            self.assertTrue(report[name]["ok"], msg=report[name]["issues"])
            self.assertTrue(report[name]["has_adapter"])


class TestAttackRegistry(unittest.TestCase):
    def test_builtin_attacks_present(self):
        for name in ("random", "bandwagon", "pgd", "tpa", "advinject", "uba"):
            self.assertIn(name, available_attacks())

    def test_register_attack_and_validation(self):
        register_attack("tmp_attack", "attacks.random.generate", entry="main",
                        meta_kwarg="raw_meta")
        self.assertIn("tmp_attack", ATTACK_SPECS)
        with self.assertRaises(ValueError):
            register_attack("tmp_attack", "x.y")
        with self.assertRaises(ValueError):
            register_attack("bad_kwarg", "x.y", meta_kwarg="nope")
        ATTACK_SPECS.pop("tmp_attack", None)


if __name__ == "__main__":
    unittest.main()
