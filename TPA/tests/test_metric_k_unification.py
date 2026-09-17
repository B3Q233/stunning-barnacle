# -*- coding: utf-8 -*-
"""模型指标 K 统一装配的回归测试（spec 2026-09-17 §5 的 L1–L4）。

事故背景：canonical 化把 `evaluation.k` 提到顶层 `k` 后，
`models/{lightgcn,mf,wmf}/train.py` 仍只展平 data/model/training/evaluation 四个分片，
顶层 `k`/`dataset` 被静默丢弃 → `apply_k` 回退 default=20，产物变成 recall@20/ndcg@20，
`dataset` 还会静默回退 gowalla。本文件按四层守住"K 唯一权威"：

- L1 `flatten_model_config` 单元行为；
- L2 全模型配置契约（canonicalize → resolve_k → apply_k → 检查最终 metrics）；
- L3 攻击侧（registry / fit / 兜底指标）一致性；
- L4 模型入口与报告的守卫（禁止内联展平与硬编码指标名回流）。

运行：``G:\\Idea\\.venv\\Scripts\\python.exe -m unittest tests.test_metric_k_unification -v``
"""
from __future__ import annotations

import copy
import importlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml

TPA_ROOT = Path(__file__).resolve().parents[1]
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from training.config_utils import (  # noqa: E402
    apply_k,
    canonicalize_config,
    resolve_k,
)
from training.metrics import eval_ks_from_metrics, metric_k, parse_metrics  # noqa: E402

MODELS_DIR = TPA_ROOT / "models"

# 全部注册了 build_training_config / resolve_metrics_cfg 的攻击入口
ATTACK_FIT_MODULES = (
    "attacks.random.fit",
    "attacks.bandwagon.fit",
    "attacks.pgd.fit",
    "attacks.tpa.fit",
    "attacks.uba.fit",
)


def _canonical(k: int = 10) -> dict:
    """canonical 模型配置样板（顶层 dataset/k + 分片），供 L1 使用。"""
    return {
        "dataset": "toy",
        "k": k,
        "run_tag": None,
        "model": {"emb_dim": 8},
        "training": {"epochs": 1},
        "evaluation": {
            "metrics": [{"recall@{k}": "upper"}, {"ndcg@{k}": "upper"}],
            "checkpoint_mode": "per_metric",
        },
    }


def _metric_k_errors(raw: dict) -> list:
    """canonicalize → resolve_k → apply_k → 检查最终 metrics（spec I1/I4）。

    为什么按解析结果而不是 YAML 文本判断：要守的是运行时契约。`k: 10` 配
    `recall@20` 在运行时确实会产出 @20 的指标名（第二权威），必须在解析后的
    metrics 上判红；而 `recall@{k}` 模板合法，展开后比较即可。
    """
    canonical = canonicalize_config(raw)
    k = resolve_k(canonical)
    resolved = apply_k(canonical)
    metrics = (resolved.get("evaluation") or {}).get(
        "metrics", resolved.get("metrics"))
    errors: list = []
    for name in (parse_metrics(metrics) if metrics else []):
        if "{k}" in name:
            errors.append(f"模板未展开: {name}")                 # 绕过统一装配
            continue
        mk = metric_k(name)
        if mk is not None and mk != k:
            errors.append(f"{name} 的 K={mk} 与顶层 k={k} 不一致")  # 第二权威
    return errors


def _attack_cfg(k: int = 5) -> dict:
    """canonical 攻击配置样板（顶层 k + 已展开的攻击选优指标）。

    注意：真实链路里 run.py 已做 canonicalize + apply_k，所以 fit.py 收到的是
    展开后的 `target_ndcg@5`；夹具必须保持同一形态，否则测的不是运行契约。
    """
    return {
        "dataset": "ml100k",
        "k": k,
        "model": {"name": "lightgcn", "overrides": {}},
        "training": {},
        "evaluation": {"metrics": [{f"target_ndcg@{k}": "upper"}]},
    }


class FlattenModelConfigTest(unittest.TestCase):
    """L1：统一装配函数的单元行为。"""

    def test_flatten_keeps_top_level_k(self):
        from training.config_utils import flatten_model_config

        self.assertEqual(flatten_model_config(_canonical(k=10))["k"], 10)

    def test_flatten_keeps_top_level_dataset(self):
        from training.config_utils import flatten_model_config

        self.assertEqual(flatten_model_config(_canonical())["dataset"], "toy")

    def test_flatten_expands_metric_templates(self):
        from training.config_utils import flatten_model_config

        flat = flatten_model_config(_canonical(k=10))
        self.assertEqual(flat["metrics"],
                         [{"recall@10": "upper"}, {"ndcg@10": "upper"}])

    def test_top_level_k_beats_section_k(self):
        from training.config_utils import flatten_model_config

        raw = _canonical(k=10)
        raw["evaluation"]["k"] = 20          # legacy 分片键不得覆盖 canonical 顶层
        self.assertEqual(flatten_model_config(raw)["k"], 10)

    def test_legacy_section_k_mapped(self):
        from training.config_utils import flatten_model_config

        raw = _canonical(k=10)
        raw.pop("k")
        raw["evaluation"]["k"] = 10
        flat = flatten_model_config(raw)
        self.assertEqual(flat["k"], 10)
        self.assertEqual(flat["metrics"],
                         [{"recall@10": "upper"}, {"ndcg@10": "upper"}])

    def test_missing_k_warns(self):
        from training.config_utils import flatten_model_config

        raw = _canonical(k=10)
        raw.pop("k")
        with self.assertWarns(UserWarning):
            flat = flatten_model_config(raw)
        self.assertEqual(flat["k"], 20)      # 兼容默认可见，但不是静默

    def test_missing_dataset_raises(self):
        from training.config_utils import flatten_model_config

        raw = _canonical()
        raw.pop("dataset")
        with self.assertRaises(ValueError):
            flatten_model_config(raw)

    def test_does_not_mutate_input(self):
        from training.config_utils import flatten_model_config

        raw = _canonical()
        before = copy.deepcopy(raw)
        flatten_model_config(raw)
        self.assertEqual(raw, before)


class ModelConfigContractTest(unittest.TestCase):
    """L2：全模型配置契约（A3 核心交付）。"""

    def test_every_model_config_metric_k_matches_resolved_k(self):
        """任意模型声明 k=X，其解析后 metrics 只能出现 @X（spec I4）。"""
        cfg_paths = sorted(MODELS_DIR.glob("*/config.yaml"))
        self.assertTrue(cfg_paths, "未找到任何 models/*/config.yaml")
        for cfg_path in cfg_paths:
            with self.subTest(model=cfg_path.parent.name):
                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                self.assertEqual(_metric_k_errors(raw), [],
                                 f"{cfg_path} 出现 K 的第二权威")

    def test_contract_catches_hardcoded_metric_k(self):
        """契约测试必须能抓住本次事故（否则是恒真测试）。"""
        bad = {"dataset": "toy", "k": 10,
               "evaluation": {"metrics": [{"recall@20": "upper"}]}}
        self.assertTrue(_metric_k_errors(bad),
                        "k:10 + recall@20 必须判红")

    def test_lightgcn_entry_yields_k10_yelp2018(self):
        """本次事故的复现用例：k:10 必须落成 @10，dataset 不得被丢弃。"""
        from training.config_utils import build_training_config_from_yaml

        cfg = build_training_config_from_yaml(
            MODELS_DIR / "lightgcn" / "config.yaml")
        self.assertEqual(cfg.get("k"), 10)
        self.assertEqual(cfg.get("dataset"), "yelp2018")
        self.assertEqual(list(parse_metrics(cfg.get("metrics"))),
                         ["recall@10", "ndcg@10"])

    def test_mf_wmf_entry_preserve_declared_k(self):
        """统一的语义是"听顶层 k"，不是"统一改成 10"。"""
        from training.config_utils import build_training_config_from_yaml

        for model, expected_k in (("mf", 20), ("wmf", 10)):
            with self.subTest(model=model):
                cfg = build_training_config_from_yaml(
                    MODELS_DIR / model / "config.yaml")
                self.assertEqual(cfg.get("k"), expected_k)
                for name in parse_metrics(cfg.get("metrics") or []):
                    mk = metric_k(name)
                    if mk is not None:
                        self.assertEqual(mk, expected_k, name)


class AttackSideTest(unittest.TestCase):
    """L3：攻击侧不得拿到未展开模板，也不得自带第二权威。"""

    def test_load_model_config_expands_k(self):
        from models.registry import load_model_config

        cfg = load_model_config("lightgcn")
        self.assertEqual(cfg["evaluation"]["k"], 10)
        self.assertNotIn("{k}", str(cfg["evaluation"]["metrics"]))

    def test_attack_fit_k_follows_config(self):
        cfg = _attack_cfg(k=5)
        for name in ATTACK_FIT_MODULES:
            with self.subTest(attack=name):
                mod = importlib.import_module(name)
                tcfg = mod.build_training_config(cfg, "ml100k", "lightgcn")
                self.assertEqual(tcfg.get("k"), 5)
                metrics = mod.resolve_metrics_cfg(cfg, "lightgcn")
                self.assertEqual(eval_ks_from_metrics(metrics, 5), [5])

    def test_metrics_fallback_uses_resolved_k(self):
        """旧模型没有 evaluation.metrics 时，兜底必须按 resolved K 派生。"""
        cfg = _attack_cfg(k=5)
        cfg["evaluation"].pop("metrics")     # 触发派生兜底分支
        fake_model_cfg = {"model": {}, "training": {}, "evaluation": {}}
        for name in ATTACK_FIT_MODULES:
            with self.subTest(attack=name):
                mod = importlib.import_module(name)
                with mock.patch.object(mod, "load_model_config",
                                       return_value=fake_model_cfg):
                    metrics = mod.resolve_metrics_cfg(cfg, "lightgcn")
                names = list(parse_metrics(metrics))
                self.assertEqual(names, ["recall@5", "ndcg@5"])


class EntryPointGuardTest(unittest.TestCase):
    """L4：入口与报告的守卫（防止旧实现回流）。"""

    def test_model_entrypoints_use_shared_assembler(self):
        from training.config_utils import build_training_config_from_yaml

        for name in ("models.lightgcn.train", "models.mf.train",
                     "models.wmf.train"):
            with self.subTest(module=name):
                mod = importlib.import_module(name)
                self.assertIs(
                    getattr(mod, "build_training_config_from_yaml", None),
                    build_training_config_from_yaml,
                    f"{name} 必须使用统一装配入口")

    def test_wmf_report_metric_names_follow_history(self):
        from models.wmf.report import _metric_names_from_history

        history = [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0,
                    "rank": 0.3, "recall@10": 0.1, "ndcg@10": 0.2}]
        names = _metric_names_from_history(history)
        self.assertEqual(names, ["rank", "recall@10", "ndcg@10"])
        self.assertNotIn("@20", " ".join(names))


if __name__ == "__main__":
    unittest.main()
