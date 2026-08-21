# 批量投毒攻击系统（v1 最终版）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 本会话按 multi_agent_mode 不启用子代理，采用 inline 执行（executing-plans），每个任务结束跑测试并提交。

**Goal:** 按 v1.0 Design Freeze 实现 `TPA/attacks/batch/` 批跑系统：插件化 Attack Registry + 四层 Deep Merge 配置继承 + 分层采样生成原子配置（分层目录）→ 公共分类缓存 → 逐个训练 → results.csv/summary.md/meta.json/logs 整合，先以 ml100k + bandwagon mini 实验跑通。

**Architecture:** 纯新增，不改任何现有攻击代码。`utils.py`（deep_merge/路径/JSON）+ `registry.py`（AttackSpec 插件注册，内置注册四个攻击）+ `generator.py`（四层合并 + 采样 + 原子配置）+ `runner.py`（registry 调度 + 公共缓存 + 目录整理 + 日志）+ `aggregate.py`（整合）+ `run.py`（CLI）。

**Tech Stack:** Python 3.12 / PyTorch 2.5 / stdlib unittest / PyYAML

## Global Constraints

- 测试只用 stdlib unittest，放 `TPA/tests/test_*.py`；运行命令（在 `G:\Idea\TPA` 下）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`。
- 仓库文档与提交信息使用中文；提交格式 Conventional Commits `type(scope): 中文描述`。
- 只用 `git add` 加明确路径，禁止 `git add -A` / `git add -f`。
- **不修改**任何 `attacks/{bandwagon,pgd,random,tpa}/*` 与 `models/*` 代码；仅新增 `attacks/batch/`、`TPA/tests/` 与 `.gitignore` 一行。
- 单测全部 CPU 可跑、不训练真实模型（用 fixture）；实验产物（output/、cache/、data/poisoned/、*.pt）不入库。
- 配置合并：Deep Merge 嵌套 dict 递归、list/标量覆盖、不改入参；优先级 P1 运行时 > P2 override > P3 Batch > P4 攻击默认。
- 分层命名固定：`popular | normal | cold`；攻击侧分类缓存 `ordinary` 在归一化时映射为 `normal`。

---

### Task 1: 骨架 + utils（deep_merge）+ Registry + 配置校验

**Files:**
- Create: `TPA/attacks/batch/__init__.py`、`TPA/attacks/batch/utils.py`、`TPA/attacks/batch/registry.py`、`TPA/attacks/batch/generator.py`
- Modify: `G:\Idea\.gitignore`（追加一行）
- Test: `TPA/tests/test_batch_config.py`、`TPA/tests/test_batch_registry.py`

**Interfaces:**
- Produces: `utils.deep_merge(base, overlay) -> dict`；`utils.flatten_experiment(cfg)`；`utils.group_name(cfg)`；`utils.resolution_k(cfg)`；`utils.read_json/write_json`；`utils.public_cache_dir/public_rec_freq_path`；`registry.AttackSpec`；`registry.register/get/registered_names`；`generator.validate_batch_config/load_batch_config`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_config.py`：

```python
"""Batch 配置校验与公共工具单测（CPU）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.generator import load_batch_config, validate_batch_config
from attacks.batch.utils import deep_merge, flatten_experiment, group_name


def _base_cfg():
    return {
        "attack": {"name": "bandwagon"},
        "experiment": {"dataset": "ml100k", "seed": 42},
        "model": {"name": "lightgcn", "overrides": {}},
        "classification": {"k": 10, "popular_ratio": 0.2,
                           "checkpoint": "models/lightgcn/checkpoints/best.pt"},
        "warm_start": {"enabled": True,
                       "checkpoint": "models/lightgcn/checkpoints/best.pt"},
        "training": {"epochs": 5, "device": "cpu"},
        "batch": {"tiers": ["popular", "normal", "cold"], "per_tier": 3,
                  "strategy": "random", "seed": 42},
        "override": {},
    }


class ValidateBatchConfigTest(unittest.TestCase):

    def test_valid_passes(self):
        validate_batch_config(_base_cfg())

    def test_missing_experiment_raises(self):
        cfg = _base_cfg()
        del cfg["experiment"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_missing_batch_raises(self):
        cfg = _base_cfg()
        del cfg["batch"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_zero_per_tier_raises(self):
        cfg = _base_cfg()
        cfg["batch"]["per_tier"] = 0
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_unknown_tier_raises(self):
        cfg = _base_cfg()
        cfg["batch"]["tiers"] = ["hot"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)


class BatchConfigIOTest(unittest.TestCase):

    def test_load_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "batch.yaml"
            p.write_text(json.dumps(_base_cfg()), encoding="utf-8")
            cfg = load_batch_config(p)
        self.assertEqual(cfg["experiment"]["dataset"], "ml100k")


class UtilsTest(unittest.TestCase):

    def test_group_name(self):
        self.assertEqual(group_name(_base_cfg()),
                         "bandwagon_ml100k_lightgcn_top10")

    def test_flatten_experiment(self):
        flat = flatten_experiment(_base_cfg())
        self.assertEqual(flat["dataset"], "ml100k")
        self.assertEqual(flat["seed"], 42)
        self.assertNotIn("experiment", flat)
        self.assertNotIn("batch", flat)

    def test_deep_merge_nested_and_priority(self):
        attack_default = {"attack": {"ratio": 0.03, "filler_size": 20},
                          "training": {"epochs": 30}}
        batch = {"training": {"epochs": 10}}
        override = {"attack": {"filler_size": 40}}
        merged = deep_merge(deep_merge(attack_default, batch), override)
        self.assertEqual(merged, {
            "attack": {"ratio": 0.03, "filler_size": 40},
            "training": {"epochs": 10},
        })
        self.assertEqual(attack_default["training"]["epochs"], 30)
```

`TPA/tests/test_batch_registry.py`：

```python
"""Attack Registry 插件注册单测（CPU）。"""
import unittest

from attacks.batch import registry


class RegistryTest(unittest.TestCase):

    def test_builtin_attacks_registered(self):
        names = registry.registered_names()
        for name in ("bandwagon", "random", "pgd", "tpa"):
            self.assertIn(name, names)

    def test_get_returns_spec(self):
        spec = registry.get("bandwagon")
        self.assertEqual(spec.name, "bandwagon")
        self.assertTrue(spec.config_path.endswith(
            "attacks/bandwagon/config.yaml"))
        self.assertTrue(callable(spec.classify))
        self.assertTrue(callable(spec.generate))
        self.assertTrue(callable(spec.fit))

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            registry.get("not_exist")

    def test_duplicate_register_raises(self):
        with self.assertRaises(ValueError):
            registry.register("bandwagon", "x.yaml",
                              lambda: None, lambda: None, lambda: None)
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_config tests.test_batch_registry -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.batch.utils'`。

- [ ] **Step 3: 最小实现**

`TPA/attacks/batch/__init__.py`（空）。`TPA/attacks/batch/utils.py`：

```python
"""Batch 公共工具：deep_merge / 路径 / 命名 / JSON。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：嵌套 dict 深合并，list/标量整体覆盖；不修改入参。"""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolution_k(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("classification", {}).get("k")
               or cfg.get("training", {}).get("k") or 20)


def group_name(cfg: Dict[str, Any]) -> str:
    return (f"{cfg['attack']['name']}_{cfg['experiment']['dataset']}_"
            f"{cfg['model']['name']}_top{resolution_k(cfg)}")


def flatten_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把 experiment.* 展开到顶层，删除 batch 段（override 由生成器另行处理）。"""
    out = dict(cfg)
    out.update(out.pop("experiment"))
    out.pop("batch", None)
    return out


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def public_cache_dir(cfg: Dict[str, Any]) -> Path:
    return (PROJECT_ROOT / "attacks" / "batch" / "cache" / "classification"
            / cfg["experiment"]["dataset"] / cfg["model"]["name"]
            / f"top{resolution_k(cfg)}")


def public_rec_freq_path(cfg: Dict[str, Any]) -> Path:
    return public_cache_dir(cfg) / "rec_freq.json"
```

`TPA/attacks/batch/registry.py`：

```python
"""Batch 攻击插件注册器（Plugin Registry）。"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_path: str
    classify: Callable
    generate: Callable
    fit: Callable


_REGISTRY: Dict[str, AttackSpec] = {}


def register(name: str, config_path: str, classify: Callable,
             generate: Callable, fit: Callable) -> None:
    if name in _REGISTRY:
        raise ValueError(f"攻击 {name} 已注册")
    _REGISTRY[name] = AttackSpec(
        name=name, config_path=config_path,
        classify=classify, generate=generate, fit=fit)


def get(name: str) -> AttackSpec:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的攻击 {name!r}，可用：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_names() -> List[str]:
    return sorted(_REGISTRY)


def _register_builtin() -> None:
    for name in ("bandwagon", "random", "pgd", "tpa"):
        register(
            name,
            f"attacks/{name}/config.yaml",
            classify=importlib.import_module(f"attacks.{name}.classify").main,
            generate=importlib.import_module(f"attacks.{name}.generate").main,
            fit=importlib.import_module(f"attacks.{name}.fit").main,
        )


_register_builtin()
```

`TPA/attacks/batch/generator.py`（本任务只写校验与加载）：

```python
"""批量投毒攻击：配置生成（四层 Deep Merge + 分层采样）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIER_NAMES = ("popular", "normal", "cold")


def validate_batch_config(cfg: Dict[str, Any]) -> None:
    exp = cfg.get("experiment")
    if not isinstance(exp, dict) or not exp.get("dataset"):
        raise ValueError("缺少 experiment.dataset")
    batch = cfg.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("缺少 batch 段")
    per_tier = batch.get("per_tier")
    if not isinstance(per_tier, int) or per_tier <= 0:
        raise ValueError("batch.per_tier 必须为正整数")
    tiers = batch.get("tiers")
    if not tiers:
        raise ValueError("batch.tiers 不能为空")
    for tier in tiers:
        if tier not in TIER_NAMES:
            raise ValueError(f"未知分层 {tier!r}，可选 {TIER_NAMES}")
    if batch.get("strategy", "random") not in ("random", "first"):
        raise ValueError("batch.strategy 仅支持 random|first")
    if not cfg.get("model", {}).get("name"):
        raise ValueError("缺少 model.name")
    if not cfg.get("attack", {}).get("name"):
        raise ValueError("缺少 attack.name")


def load_batch_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_batch_config(cfg)
    return cfg
```

`G:\Idea\.gitignore` 末尾追加：`attacks/batch/cache/`。

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_config tests.test_batch_registry -v`
Expected: `OK`（9 + 4 个用例）。

- [ ] **Step 5: 提交**

```bash
git add .gitignore TPA/attacks/batch/__init__.py TPA/attacks/batch/utils.py TPA/attacks/batch/registry.py TPA/attacks/batch/generator.py TPA/tests/test_batch_config.py TPA/tests/test_batch_registry.py
git commit -m "feat(attacks): Batch 公共工具、插件 Registry 与配置校验"
```

---

### Task 2: Generator（四层合并 + 分层采样 + 原子配置）

**Files:**
- Modify: `TPA/attacks/batch/generator.py`
- Test: `TPA/tests/test_batch_merge.py`、`TPA/tests/test_batch_generator.py`

**Interfaces:**
- Consumes: Task 1 `utils.deep_merge/flatten_experiment/group_name/resolution_k`、`registry.get`。
- Produces: `load_attack_default(cfg) -> dict`；`build_atomic_base(cfg) -> dict`；`sample_targets(categories, tiers, per_tier, strategy, seed) -> Dict[str, List[int]]`；`atomic_run_tag(cfg, tier, item_id, batch_tag)`；`build_atomic_config(cfg, item_id, tier, batch_tag) -> dict`；`config_rel_path(cfg, tier, item_id) -> str`；`generate_configs(cfg, categories, batch_tag) -> List[Tuple[str, dict]]`；`write_configs(entries, configs_dir) -> List[Path]`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_merge.py`：

```python
"""Deep Merge 配置继承单测（CPU）。"""
import unittest

from attacks.batch.utils import deep_merge


class DeepMergeTest(unittest.TestCase):

    def test_nested_dict_merge(self):
        out = deep_merge(
            {"attack": {"ratio": 0.03, "filler_size": 20},
             "training": {"epochs": 30}},
            {"training": {"epochs": 10}})
        self.assertEqual(out["training"]["epochs"], 10)
        self.assertEqual(out["attack"]["ratio"], 0.03)

    def test_scalar_and_list_overwrite(self):
        out = deep_merge({"a": 1, "b": [1, 2], "c": {"x": 1}},
                         {"b": [3], "c": 5})
        self.assertEqual(out, {"a": 1, "b": [3], "c": 5})

    def test_inputs_not_mutated(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"y": 2}})
        self.assertEqual(base, {"a": {"x": 1}})
```

`TPA/tests/test_batch_generator.py`：

```python
"""Generator：分层采样 + 四层合并原子配置单测（CPU）。"""
import tempfile
import unittest
from pathlib import Path

from attacks.batch.generator import (
    atomic_run_tag,
    build_atomic_config,
    config_rel_path,
    generate_configs,
    sample_targets,
    write_configs,
)

from tests.test_batch_config import _base_cfg


def _categories():
    return {"popular": [1, 2], "normal": [5, 6], "cold": [9]}


class SampleTargetsTest(unittest.TestCase):

    def test_random_fixed_seed_deterministic(self):
        categories = {"popular": [0, 1, 2, 3, 4],
                      "normal": [10, 11, 12], "cold": [20, 21, 22, 23]}
        a = sample_targets(categories, ["popular", "normal", "cold"], 2,
                           "random", 42)
        b = sample_targets(categories, ["popular", "normal", "cold"], 2,
                           "random", 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a["popular"]), 2)
        self.assertTrue(all(i in categories["popular"] for i in a["popular"]))
        self.assertEqual(len(a["cold"]), 2)

    def test_first_takes_head(self):
        out = sample_targets({"cold": [20, 21, 22]}, ["cold"], 2,
                             "first", 42)
        self.assertEqual(out["cold"], [20, 21])

    def test_empty_tier_skipped(self):
        out = sample_targets({"cold": []}, ["cold"], 2, "random", 42)
        self.assertEqual(out["cold"], [])


class AtomicConfigTest(unittest.TestCase):

    def test_merged_atomic_config(self):
        cfg = _base_cfg()
        cfg["override"] = {"attack": {"filler_size": 40}}
        atomic = build_atomic_config(cfg, 251, "cold", "2026-08-21-15-30")
        self.assertEqual(atomic["attack"]["ratio"], 0.03)       # P4 默认
        self.assertEqual(atomic["attack"]["filler_size"], 40)   # P2 override
        self.assertEqual(atomic["training"]["epochs"], 5)       # P3 batch
        self.assertEqual(atomic["attack"]["target_items"],
                         {"strategy": "specified", "ids": [251]})
        self.assertEqual(atomic["run_tag"], "2026-08-21-15-30-cold-item251")
        self.assertEqual(atomic["output"]["dir"],
                         "attacks/batch/output/2026-08-21-15-30/runs")
        self.assertEqual(atomic["dataset"], "ml100k")
        self.assertNotIn("experiment", atomic)
        self.assertNotIn("batch", atomic)
        self.assertNotIn("override", atomic)

    def test_run_tag_and_rel_path(self):
        cfg = _base_cfg()
        self.assertEqual(atomic_run_tag(cfg, "cold", 251, "2026-08-21-15-30"),
                         "2026-08-21-15-30-cold-item251")
        self.assertEqual(config_rel_path(cfg, "cold", 251),
                         "bandwagon_ml100k_lightgcn_top10/cold/item251.yaml")


class GenerateAndWriteTest(unittest.TestCase):

    def test_generate_configs_count_and_names(self):
        cfg = _base_cfg()
        entries = generate_configs(cfg, _categories(), "2026-08-21-15-30")
        self.assertEqual(len(entries), 3)
        rels = [rel for rel, _ in entries]
        self.assertEqual(len(set(rels)), 3)
        self.assertIn("bandwagon_ml100k_lightgcn_top10/cold/item9.yaml", rels)

    def test_write_configs_hierarchical(self):
        cfg = _base_cfg()
        entries = generate_configs(cfg, _categories(), "t")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_configs(entries, Path(tmp))
            self.assertEqual(len(paths), 3)
            rel = "bandwagon_ml100k_lightgcn_top10/cold/item9.yaml"
            self.assertTrue((Path(tmp) / rel).exists())
            import yaml
            content = yaml.safe_load((Path(tmp) / rel).read_text(encoding="utf-8"))
            self.assertEqual(content["attack"]["target_items"]["ids"], [9])
            self.assertNotIn("batch", content)
            self.assertNotIn("override", content)
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_merge tests.test_batch_generator -v`
Expected: FAIL with `ImportError: cannot import name 'sample_targets'`。

- [ ] **Step 3: 最小实现（追加到 generator.py）**

```python
import random
from typing import List, Tuple

from attacks.batch.registry import get as get_attack
from attacks.batch.utils import (
    deep_merge, flatten_experiment, group_name)
from training.run_tag import sanitize_run_tag


def load_attack_default(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import yaml
    spec = get_attack(cfg["attack"]["name"])
    with open(PROJECT_ROOT / spec.config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_atomic_base(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """四层合并（P4 攻击默认 ← P3 Batch ← P2 override），剔除扩展段。"""
    merged = deep_merge(load_attack_default(cfg), flatten_experiment(cfg))
    if isinstance(cfg.get("override"), dict):
        merged = deep_merge(merged, cfg["override"])
    merged.pop("override", None)
    return merged


def sample_targets(categories, tiers, per_tier, strategy="random",
                   seed=42):
    rng = random.Random(seed)
    out = {}
    for tier in tiers:
        pool = list(categories.get(tier, []))
        if not pool:
            print(f"[batch] 层 {tier} 为空，跳过")
            out[tier] = []
            continue
        out[tier] = (pool[:per_tier] if strategy == "first"
                     else rng.sample(pool, min(per_tier, len(pool))))
    return out


def atomic_run_tag(cfg, tier, item_id, batch_tag) -> str:
    return sanitize_run_tag(f"{batch_tag}-{tier}-item{item_id}")


def build_atomic_config(cfg, item_id, tier, batch_tag) -> dict:
    atomic = build_atomic_base(cfg)
    atomic["attack"]["target_items"] = {
        "strategy": "specified", "ids": [int(item_id)]}
    atomic["run_tag"] = atomic_run_tag(cfg, tier, item_id, batch_tag)
    atomic["output"] = {"dir": f"attacks/batch/output/{batch_tag}/runs"}
    return atomic


def config_rel_path(cfg, tier, item_id) -> str:
    return f"{group_name(cfg)}/{tier}/item{item_id}.yaml"


def generate_configs(cfg, categories, batch_tag) -> List[Tuple[str, dict]]:
    sampling = cfg["batch"]
    targets = sample_targets(
        categories, sampling["tiers"], sampling["per_tier"],
        sampling.get("strategy", "random"), sampling.get("seed", 42))
    entries = []
    for tier, items in targets.items():
        for item in items:
            entries.append((config_rel_path(cfg, tier, item),
                            build_atomic_config(cfg, item, tier, batch_tag)))
    return entries


def write_configs(entries, configs_dir) -> List[Path]:
    import yaml
    paths = []
    for rel, atomic in entries:
        p = configs_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(atomic, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        paths.append(p)
    return paths
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_merge tests.test_batch_generator -v`
Expected: `OK`（3 + 7 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/generator.py TPA/tests/test_batch_merge.py TPA/tests/test_batch_generator.py
git commit -m "feat(attacks): 四层配置合并与原子配置生成"
```

---

### Task 3: 公共分类缓存（registry 驱动）

**Files:**
- Create: `TPA/attacks/batch/runner.py`
- Test: `TPA/tests/test_batch_cache.py`

**Interfaces:**
- Consumes: Task 1/2 的 `utils` 与 `generator.build_atomic_base`。
- Produces: `attack_cache_path(cfg) -> Path`；`normalize_cache(cfg, attack_cache, cache_dir)`；`ensure_classify_cache(cfg, cache_dir=None) -> dict`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_cache.py`：

```python
"""公共分类缓存归一化单测（fixture，CPU）。"""
import tempfile
import unittest
from pathlib import Path

from attacks.batch.runner import (
    attack_cache_path, ensure_classify_cache, normalize_cache)
from attacks.batch.utils import read_json

from tests.test_batch_config import _base_cfg


class NormalizeCacheTest(unittest.TestCase):

    def test_ordinary_mapped_to_normal_and_meta(self):
        cfg = _base_cfg()
        attack_cache = {
            "categories": {"popular": [1, 5], "ordinary": [31, 42],
                           "cold": [251, 987]},
            "summary": {"num_items": 1000},
        }
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            normalize_cache(cfg, attack_cache, cache_dir)
            rec = read_json(cache_dir / "rec_freq.json")
            self.assertEqual(rec["popular"], [1, 5])
            self.assertEqual(rec["normal"], [31, 42])
            self.assertEqual(rec["cold"], [251, 987])
            meta = read_json(cache_dir / "meta.json")
            for key in ("dataset", "model", "topk", "checkpoint", "generated_at"):
                self.assertIn(key, meta)
            self.assertEqual(meta["dataset"], "ml100k")
            self.assertEqual(meta["topk"], 10)

    def test_ensure_reads_existing_cache(self):
        cfg = _base_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            rec_path = cache_dir / "rec_freq.json"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            rec_path.write_text(
                '{"popular": [1], "normal": [2], "cold": [3]}',
                encoding="utf-8")
            got = ensure_classify_cache(cfg, cache_dir=cache_dir)
            self.assertEqual(got["cold"], [3])

    def test_attack_cache_path(self):
        p = attack_cache_path(_base_cfg())
        self.assertTrue(p.as_posix().endswith(
            "attacks/bandwagon/data/rec_freq/ml100k/lightgcn_top10.json"))
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_cache -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.batch.runner'`。

- [ ] **Step 3: 最小实现（runner.py 本任务部分）**

```python
"""批量投毒：调度器（公共分类缓存 + 原子实验执行）。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.batch.generator import build_atomic_base
from attacks.batch.registry import get as get_attack
from attacks.batch.utils import (
    public_cache_dir, read_json, resolution_k, write_json)


def attack_cache_path(cfg: Dict[str, Any]) -> Path:
    return (PROJECT_ROOT / "attacks" / cfg["attack"]["name"]
            / "data" / "rec_freq" / cfg["experiment"]["dataset"]
            / f"{cfg['model']['name']}_top{resolution_k(cfg)}.json")


def normalize_cache(cfg: Dict[str, Any], attack_cache: Dict[str, Any],
                    cache_dir: Path) -> None:
    categories = attack_cache["categories"]
    write_json({
        "popular": categories["popular"],
        "normal": categories["ordinary"],
        "cold": categories["cold"],
    }, cache_dir / "rec_freq.json")
    write_json({
        "dataset": cfg["experiment"]["dataset"],
        "model": cfg["model"]["name"],
        "topk": resolution_k(cfg),
        "checkpoint": cfg.get("classification", {}).get("checkpoint"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, cache_dir / "meta.json")


def ensure_classify_cache(cfg: Dict[str, Any],
                          cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    cache_dir = cache_dir or public_cache_dir(cfg)
    target = cache_dir / "rec_freq.json"
    if target.exists():
        return read_json(target)
    spec = get_attack(cfg["attack"]["name"])
    base = build_atomic_base(cfg)
    base["mode"] = "classify"
    spec.classify(base)
    normalize_cache(cfg, read_json(attack_cache_path(cfg)), cache_dir)
    return read_json(target)
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_cache -v`
Expected: `OK`（3 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/runner.py TPA/tests/test_batch_cache.py
git commit -m "feat(attacks): 公共分类缓存（registry 驱动 + ordinary→normal）"
```

---

### Task 4: Runner 调度（dry-run + 目录整理 + logs）

**Files:**
- Modify: `TPA/attacks/batch/runner.py`
- Test: `TPA/tests/test_batch_runner.py`

**Interfaces:**
- Consumes: Task 2/3。
- Produces: `plan_runs(cfg, categories, batch_tag)`；`write_meta(cfg, batch_tag, entries, meta_path)`；`staging_dir(runs_root, atomic_cfg)`；`run_atomic(atomic_cfg, stage)`；`run_batch(cfg, batch_tag, out_root, cache, max_targets=None, dry_run=False)`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_runner.py`：

```python
"""Runner 调度单测：dry-run、目录整理、logs（stub run_atomic，不训练模型）。"""
import tempfile
import unittest
from pathlib import Path

import attacks.batch.runner as runner
from attacks.batch.utils import read_json

from tests.test_batch_config import _base_cfg
from tests.test_batch_generator import _categories


class RunnerTest(unittest.TestCase):

    def test_write_meta_fields(self):
        cfg = _base_cfg()
        entries = [("a", {"run_tag": "x"}), ("b", {"run_tag": "y"})]
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "meta.json"
            runner.write_meta(cfg, "2026-08-21-15-30", entries, meta_path)
            meta = read_json(meta_path)
        self.assertEqual(meta["batch_tag"], "2026-08-21-15-30")
        self.assertEqual(meta["attack"], "bandwagon")
        self.assertEqual(meta["dataset"], "ml100k")
        self.assertEqual(meta["model"], "lightgcn")
        self.assertEqual(meta["topk"], 10)
        self.assertEqual(meta["tiers"], ["popular", "normal", "cold"])
        self.assertEqual(meta["per_tier"], 3)
        self.assertEqual(meta["total_runs"], 2)
        self.assertEqual(meta["seed"], 42)

    def test_dry_run_writes_configs_and_meta_without_execution(self):
        cfg = _base_cfg()
        calls = []
        original = runner.run_atomic
        runner.run_atomic = lambda atomic, stage: calls.append(stage)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_root = Path(tmp)
                runner.run_batch(cfg, "t", out_root, _categories(),
                                 dry_run=True)
                self.assertEqual(calls, [])
                self.assertTrue((out_root / "configs").exists())
                self.assertTrue((out_root / "meta.json").exists())
                config_files = list((out_root / "configs").rglob("*.yaml"))
                self.assertEqual(len(config_files), 3)
        finally:
            runner.run_atomic = original

    def test_run_moves_staging_and_writes_logs(self):
        cfg = _base_cfg()

        def fake_run(atomic, stage):
            if stage == "model":
                src = runner.staging_dir(runs_root, atomic)
                src.mkdir(parents=True, exist_ok=True)
                (src / "history.json").write_text(
                    '{"history": [], "best": {}}', encoding="utf-8")

        original = runner.run_atomic
        runner.run_atomic = fake_run
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out_root = Path(tmp)
                runs_root = out_root / "runs"
                runner.run_batch(cfg, "t", out_root, _categories())
                moved = (runs_root / "bandwagon_ml100k_lightgcn_top10"
                         / "cold" / "item9")
                self.assertTrue(moved.exists())
                self.assertTrue((moved / "history.json").exists())
                self.assertTrue((out_root / "logs" / "runner.log").exists())
        finally:
            runner.run_atomic = original
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_runner -v`
Expected: FAIL——`runner.run_batch` 不存在（AttributeError）。

- [ ] **Step 3: 最小实现（追加到 runner.py）**

```python
import logging
import shutil
from typing import List, Tuple

from attacks.batch.generator import generate_configs, write_configs
from attacks.batch.registry import get as get_attack


def plan_runs(cfg, categories, batch_tag):
    return generate_configs(cfg, categories, batch_tag)


def write_meta(cfg, batch_tag, entries, meta_path) -> None:
    write_json({
        "batch_tag": batch_tag,
        "attack": cfg["attack"]["name"],
        "dataset": cfg["experiment"]["dataset"],
        "model": cfg["model"]["name"],
        "topk": resolution_k(cfg),
        "tiers": list(cfg["batch"]["tiers"]),
        "per_tier": cfg["batch"]["per_tier"],
        "total_runs": len(entries),
        "seed": cfg["batch"].get("seed", 42),
    }, meta_path)


def staging_dir(runs_root: Path, atomic_cfg: Dict[str, Any]) -> Path:
    return (runs_root / atomic_cfg["dataset"]
            / atomic_cfg["model"]["name"] / atomic_cfg["run_tag"])


def run_atomic(atomic_cfg: Dict[str, Any], stage: str) -> None:
    spec = get_attack(atomic_cfg["attack"]["name"])
    fn = {"data": spec.generate, "model": spec.fit}[stage]
    fn(atomic_cfg)


def _logger(out_root: Path) -> logging.Logger:
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"batch.{out_root.name}")
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / "runner.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def run_batch(cfg, batch_tag, out_root, cache,
              max_targets=None, dry_run=False) -> None:
    configs_dir = out_root / "configs"
    runs_root = out_root / "runs"
    entries = plan_runs(cfg, cache, batch_tag)
    if max_targets is not None:
        entries = entries[:max_targets]
    write_configs(entries, configs_dir)
    write_meta(cfg, batch_tag, entries, out_root / "meta.json")
    logger = _logger(out_root)
    logger.info("batch_tag=%s total_runs=%d dry_run=%s",
                batch_tag, len(entries), dry_run)
    print(f"[batch] 原子配置 {len(entries)} 个 -> {configs_dir}")
    if dry_run:
        return
    for rel, atomic in entries:
        print(f"[batch] {atomic['run_tag']}")
        logger.info("run %s", atomic["run_tag"])
        run_atomic(atomic, "data")
        run_atomic(atomic, "model")
        src = staging_dir(runs_root, atomic)
        dst = runs_root / rel[:-len(".yaml")]
        if src != dst and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        logger.info("done %s", atomic["run_tag"])
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_runner -v`
Expected: `OK`（3 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/runner.py TPA/tests/test_batch_runner.py
git commit -m "feat(attacks): Batch 调度器（dry-run、分层整理、runner.log）"
```

---

### Task 5: 结果整合（aggregate.py）

**Files:**
- Create: `TPA/attacks/batch/aggregate.py`
- Test: `TPA/tests/test_batch_aggregate.py`

**Interfaces:**
- Produces: `scan_runs(runs_root, group)`；`build_results_rows(runs_root, group, cfg, k)`；`write_results_csv(rows, k, path)`；`tier_summary(rows, k)`；`write_summary_md(batch_tag, summary, clean_baseline, k, path)`；`compute_clean_baseline(cfg, k)`（Task 6 补齐）。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_aggregate.py`：

```python
"""结果整合单测（fixture 分层 runs 目录，CPU）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.aggregate import (
    build_results_rows, scan_runs, tier_summary,
    write_results_csv, write_summary_md)

from tests.test_batch_config import _base_cfg


def _make_run(root, group, tier, item, best):
    d = root / group / tier / f"item{item}"
    d.mkdir(parents=True)
    (d / "history.json").write_text(
        json.dumps({"history": [], "best": best}, ensure_ascii=False),
        encoding="utf-8")


class AggregateTest(unittest.TestCase):

    def test_scan_and_rows(self):
        cfg = _base_cfg()
        group = "bandwagon_ml100k_lightgcn_top10"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_run(root, group, "cold", 251,
                      {"target_hr@10": 0.5, "target_ndcg@10": 0.4,
                       "recall@10": 0.3, "ndcg@10": 0.2})
            _make_run(root, group, "cold", 987,
                      {"target_hr@10": 0.7, "target_ndcg@10": 0.6,
                       "recall@10": 0.31, "ndcg@10": 0.21})
            _make_run(root, group, "popular", 32,
                      {"target_hr@10": 0.2, "target_ndcg@10": 0.18,
                       "recall@10": 0.32, "ndcg@10": 0.22})
            self.assertEqual(len(scan_runs(root, group)), 3)
            rows = build_results_rows(root, group, cfg, 10)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["attack"], "bandwagon")
            self.assertEqual(rows[0]["dataset"], "ml100k")
            self.assertEqual(rows[0]["model"], "lightgcn")
            self.assertEqual(rows[0]["tier"], "cold")
            summary = tier_summary(rows, 10)
            self.assertAlmostEqual(summary["cold"]["target_hr@10"]["mean"], 0.6)
            self.assertAlmostEqual(summary["cold"]["target_ndcg@10"]["mean"], 0.5)
            self.assertEqual(summary["cold"]["target_hr@10"]["n"], 2)
            write_results_csv(rows, 10, root / "results.csv")
            write_summary_md(
                "2026-08-21-15-30", summary,
                {"recall@10": 0.35, "ndcg@10": 0.25}, 10,
                root / "summary.md")
            csv_text = (root / "results.csv").read_text(encoding="utf-8")
            self.assertIn("target_ndcg@10", csv_text)
            md_text = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("0.6000", md_text)
            self.assertIn("Clean Model Utility", md_text)

    def test_missing_history_skipped(self):
        cfg = _base_cfg()
        group = "bandwagon_ml100k_lightgcn_top10"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / group / "cold").mkdir(parents=True)
            self.assertEqual(build_results_rows(root, group, cfg, 10), [])
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_aggregate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.batch.aggregate'`。

- [ ] **Step 3: 最小实现（aggregate.py）**

```python
"""批量投毒：结果整合（results.csv + 按层 mean±std + clean 基线）。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_best_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("best")


def scan_runs(runs_root: Path, group: str) -> List[Tuple[str, int, Dict[str, Any]]]:
    base = runs_root / group
    if not base.exists():
        return []
    out = []
    for tier_dir in sorted(base.iterdir()):
        if not tier_dir.is_dir():
            continue
        for item_dir in sorted(tier_dir.iterdir()):
            if not item_dir.is_dir() or not item_dir.name.startswith("item"):
                continue
            best = load_best_metrics(item_dir)
            if best is None:
                continue
            out.append((tier_dir.name, int(item_dir.name[len("item"):]), best))
    return out


def build_results_rows(runs_root: Path, group: str, cfg: Dict[str, Any],
                       k: int) -> List[Dict[str, Any]]:
    rows = []
    for tier, item, best in scan_runs(runs_root, group):
        row = {
            "attack": cfg["attack"]["name"],
            "dataset": cfg["experiment"]["dataset"],
            "model": cfg["model"]["name"],
            "tier": tier,
            "item": item,
        }
        for key in (f"target_hr@{k}", f"target_ndcg@{k}",
                    f"recall@{k}", f"ndcg@{k}"):
            row[key] = best.get(key)
        rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, Any]], k: int, path: Path) -> None:
    fieldnames = ["attack", "dataset", "model", "tier", "item",
                  f"target_hr@{k}", f"target_ndcg@{k}",
                  f"recall@{k}", f"ndcg@{k}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})


def tier_summary(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tier"], []).append(row)
    out = {}
    for tier, group in sorted(grouped.items()):
        out[tier] = {}
        for metric in (f"target_hr@{k}", f"target_ndcg@{k}"):
            vals = [r[metric] for r in group
                    if isinstance(r.get(metric), (int, float))]
            if not vals:
                out[tier][metric] = {"mean": None, "std": None, "n": 0}
                continue
            out[tier][metric] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.4f}"


def write_summary_md(batch_tag, summary, clean_baseline, k, path) -> None:
    lines = [
        f"# 批量投毒攻击结果汇总（batch_tag={batch_tag}）",
        "",
        f"| Tier | HR@{k} | NDCG@{k} |",
        "|---|---|---|",
    ]
    for tier in sorted(summary):
        hr = summary[tier][f"target_hr@{k}"]
        nd = summary[tier][f"target_ndcg@{k}"]
        lines.append(f"| {tier} | {_fmt(hr['mean'])} ± {_fmt(hr['std'])} "
                     f"| {_fmt(nd['mean'])} ± {_fmt(nd['std'])} |")
    if clean_baseline:
        r = clean_baseline.get(f"recall@{k}", float("nan"))
        n = clean_baseline.get(f"ndcg@{k}", float("nan"))
        lines += ["", "## Clean Model Utility",
                  "", f"Recall@{k} : {r:.4f}", f"NDCG@{k}   : {n:.4f}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_aggregate -v`
Expected: `OK`（2 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/aggregate.py TPA/tests/test_batch_aggregate.py
git commit -m "feat(attacks): 批量投毒结果整合（results.csv + 按层汇总）"
```

---

### Task 6: CLI（run.py）+ clean 基线 + 配置与文档

**Files:**
- Create: `TPA/attacks/batch/run.py`、`TPA/attacks/batch/config.yaml`、`TPA/attacks/batch/docs/USAGE.md`、`TPA/attacks/batch/docs/DESIGN.md`

**Interfaces:**
- Consumes: Task 1-5 全部函数。
- Produces: CLI `python attacks/batch/run.py --mode all|generate|run|aggregate [--batch-tag] [--dry-run] [--skip-classify] [--max-targets]`。

- [ ] **Step 1: 实现 `compute_clean_baseline`（追加到 aggregate.py）**

```python
def compute_clean_baseline(cfg: Dict[str, Any], k: int) -> Dict[str, float]:
    """用 w_clean 在 clean 数据上算 recall@k / ndcg@k。"""
    import torch

    from attacks.batch.generator import build_atomic_base
    from attacks.batch.registry import get as get_attack
    from evaluation.attack_eval import ranking_scores
    from evaluation.metrics import build_train_mask_indices, compute_metrics
    from training.paths import resolve_from_root

    attack = cfg["attack"]["name"]
    model_name = cfg["model"]["name"]
    dataset = cfg["experiment"]["dataset"]
    spec = get_attack(attack)
    gen_mod = __import__(f"attacks.{attack}.generate", fromlist=["load_meta"])
    fit_mod = __import__(f"attacks.{attack}.fit",
                         fromlist=["build_training_config"])

    meta = gen_mod.load_meta(
        Path(str(gen_mod.DEFAULT_RAW_META).format(dataset=dataset)))
    base = build_atomic_base(cfg)
    train_cfg = fit_mod.build_training_config(base, dataset, model_name)
    reg_mod = __import__(f"attacks.{attack}.registry",
                         fromlist=["get_model_cls"])
    model = reg_mod.get_model_cls(model_name)(
        train_cfg, meta["num_users"], meta["num_items"], None)
    ckpt = resolve_from_root(cfg["classification"]["checkpoint"], PROJECT_ROOT)
    model.load_state_dict(torch.load(
        ckpt, map_location=model._device, weights_only=True)["model_state_dict"])
    scores, users, test_pos = ranking_scores(model, meta["test_pairs"])
    rows, cols = build_train_mask_indices(meta["user_items"], users)
    return compute_metrics(
        scores, meta["user_items"], test_pos, k=k,
        mask_indices=(rows, cols),
        topk_device=getattr(model._device, "type", "cpu"))
```

- [ ] **Step 2: 实现 run.py**

```python
"""批量投毒攻击编排 CLI。"""
if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    import yaml

    from attacks.batch.aggregate import (
        build_results_rows, compute_clean_baseline, tier_summary,
        write_results_csv, write_summary_md)
    from attacks.batch.generator import load_batch_config
    from attacks.batch.runner import (
        ensure_classify_cache, run_atomic, run_batch, staging_dir)
    from attacks.batch.utils import group_name, public_rec_freq_path
    from training.run_tag import resolve_run_tag

    parser = argparse.ArgumentParser(description="批量投毒攻击")
    parser.add_argument("--config", type=str,
                        default="attacks/batch/config.yaml")
    parser.add_argument("--mode", choices=["generate", "run", "aggregate", "all"],
                        default="all")
    parser.add_argument("--batch-tag", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-classify", action="store_true")
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()

    cfg = load_batch_config(Path(args.config))
    batch_tag = args.batch_tag or resolve_run_tag(cfg)
    out_root = Path(cfg.get("output", {}).get(
        "dir", "attacks/batch/output")) / batch_tag
    configs_dir, runs_root = out_root / "configs", out_root / "runs"
    k = cfg.get("evaluation", {}).get("k", 10)
    group = group_name(cfg)

    if args.mode in ("generate", "all"):
        if args.skip_classify and not public_rec_freq_path(cfg).exists():
            raise FileNotFoundError(
                f"缓存不存在：{public_rec_freq_path(cfg)}"
                "（--skip-classify 需要已有缓存）")
        cache = ensure_classify_cache(cfg)
        run_batch(cfg, batch_tag, out_root, cache,
                  max_targets=args.max_targets,
                  dry_run=(args.mode == "generate") or args.dry_run)
    if args.mode == "run":
        for p in sorted(configs_dir.rglob("*.yaml")):
            atomic = yaml.safe_load(p.read_text(encoding="utf-8"))
            print(f"[batch] run {atomic['run_tag']}")
            run_atomic(atomic, "data")
            run_atomic(atomic, "model")
            src = staging_dir(runs_root, atomic)
            dst = runs_root / p.relative_to(configs_dir).with_suffix("")
            if src != dst and src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    if args.mode in ("aggregate", "all") and not args.dry_run:
        rows = build_results_rows(runs_root, group, cfg, k)
        write_results_csv(rows, k, out_root / "results.csv")
        clean = compute_clean_baseline(cfg, k)
        write_summary_md(batch_tag, tier_summary(rows, k), clean, k,
                         out_root / "summary.md")
        print(f"[batch] 整合完成：{len(rows)} 个原子实验 -> {out_root}")
```

- [ ] **Step 3: 写 `config.yaml`（默认 mini 规模，四层继承只需写差异项）**

```yaml
# 批量投毒攻击批跑配置（v1 最终版）
# 其余参数继承 attacks/bandwagon/config.yaml 默认值（P4）
attack:
  name: bandwagon

experiment:
  dataset: ml100k
  seed: 42

model:
  name: lightgcn
  overrides: {}

classification:
  k: 10
  popular_ratio: 0.2
  checkpoint: models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt

warm_start:
  enabled: true
  checkpoint: models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt

training:
  epochs: 5
  device: cuda

batch:
  tiers: [popular, normal, cold]
  per_tier: 2
  strategy: random
  seed: 42

override: {}
```

- [ ] **Step 4: 验证配置可加载且合并正确**

Run（在 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe -c "from pathlib import Path; from attacks.batch.generator import load_batch_config, build_atomic_base; c=load_batch_config(Path('attacks/batch/config.yaml')); b=build_atomic_base(c); print(c['experiment']['dataset'], b['training']['epochs'], b['attack']['ratio'])"`
Expected: `ml100k 5 0.03`（epochs 来自 Batch P3，ratio 继承攻击默认 P4）。

- [ ] **Step 5: 写 USAGE.md / DESIGN.md 并提交**

USAGE.md：结构、前置条件（w_clean checkpoint / 数据集 meta）、运行方式（`--mode all|generate|run|aggregate`、`--dry-run`、`--max-targets`、`--batch-tag`）、输出目录、四层配置说明。DESIGN.md：引用对应 spec，说明插件化/配置继承/SRP 与 v2 迭代方向（多数据集×多划分、tpa 缓存适配）。

```bash
git add TPA/attacks/batch/run.py TPA/attacks/batch/config.yaml TPA/attacks/batch/aggregate.py TPA/attacks/batch/docs
git commit -m "feat(attacks): Batch CLI、clean 基线与使用文档"
```

---

### Task 7: mini E2E 与全量回归

**Files:** 无代码改动（产出不入库）。

- [ ] **Step 1: 训练 ml100k clean lightgcn checkpoint（w_clean）**

Run（在 `G:\Idea`，内联脚本）：训练 50 epoch（batch=256，emb_dim=64，device=cuda），保存
`models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt`（含 `model_state_dict`）。
Expected：loss 下降，无 NaN，文件生成。

- [ ] **Step 2: mini 冒烟（E2E）**

复制 `attacks/batch/config.yaml` 到 `tmp/batch_smoke.yaml`，改
`training.epochs: 1`、`training.device: cpu`、`batch.per_tier: 1`、
`batch.tiers: [cold]`，Run（在 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --config ../tmp/batch_smoke.yaml --mode all`
Expected：classify 一次 → 1 个原子实验 data+model 跑通；
`attacks/batch/output/{batch_tag}/` 下 `configs/.../cold/item{id}.yaml` 1 个、
`runs/.../cold/item{id}/` 1 个、results.csv 1 行、summary.md 含 cold 与
Clean Model Utility、meta.json 字段齐全、logs/runner.log 存在。

- [ ] **Step 3: 正式 mini 批量**

用仓库默认 `attacks/batch/config.yaml`（epochs=5、per_tier=2、三层、GPU）运行
`python attacks/batch/run.py --mode all`。Expected：6 个原子实验全部跑完，
configs/ 与 runs/ 各 6 项，results.csv 6 行，summary.md 三层 mean±std。

- [ ] **Step 4: 全量回归**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`
Expected：全部通过（既有 105 个 + 新增 batch 测试）。

- [ ] **Step 5: 自查并汇报**

Run（在 `G:\Idea`）：`git status --short` 与 `git log --oneline -10`。
Expected：仅 batch 模块与测试提交；产物（output/、cache/、data/poisoned/、*.pt）未入库。
把 results.csv 与 summary.md 关键数字（各层 target_hr@10 / target_ndcg@10 mean±std、
clean 基线）汇报给用户。
