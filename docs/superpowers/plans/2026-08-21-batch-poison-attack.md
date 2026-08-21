# 批量投毒攻击（配置生成器 + 分层采样 + 结果整合）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 本会话按 multi_agent_mode 不启用子代理，采用 inline 执行（executing-plans），每个任务结束跑测试并提交。

**Goal:** 新增 `TPA/attacks/batch/` 批跑工具：用一份批跑配置生成多个原子投毒配置（按 clean 模型 top@k 频次分 popular/normal/cold 三层、每层采样 K 个目标），逐个训练并整合按层平均结果，先以 ml100k + bandwagon 的 mini 实验跑通全链路。

**Architecture:** 纯新增、不改现有攻击代码。`generator.py` 负责分层采样与原子配置生成；`runner.py` 复用各攻击 classify/data/model 模块 in-process 调度；`aggregate.py` 读各 run 的 history.json 聚合 results.csv 与 summary.md；`run.py` 提供 CLI。

**Tech Stack:** Python 3.12 / PyTorch 2.5 / stdlib unittest / PyYAML / scipy.sparse

## Global Constraints

- 测试只用 stdlib unittest，放 `TPA/tests/test_*.py`；运行命令（在 `G:\Idea\TPA` 下）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`。
- 仓库文档与提交信息使用中文；提交格式 Conventional Commits `type(scope): 中文描述`。
- 只用 `git add` 加明确路径，禁止 `git add -A` / `git add -f`。
- v1 **不修改**任何 `attacks/*/`（除 batch 外）与 `models/*/` 代码；攻击实验产物（`data/poisoned/`、`outputs/`、`*.pt`）不入库。
- 单测全部 CPU 可跑、不训练真实模型（用 fixture 数据）。
- 批跑配置缺省时 `device: cpu`（测试友好）；真实 mini 运行可用 GPU。

---

### Task 1: 批跑配置加载与校验（generator 基础）

**Files:**
- Create: `TPA/attacks/batch/__init__.py`、`TPA/attacks/batch/generator.py`
- Test: `TPA/tests/test_batch_generator.py`

**Interfaces:**
- Produces: `load_batch_config(path: Path) -> dict`；`validate_batch_config(cfg: dict) -> None`；`classify_cache_path(cfg: dict) -> Path`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_generator.py` 前半部分：

```python
"""批量投毒：批跑配置校验与分层采样/原子配置生成单测（CPU、不训练模型）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.generator import (
    atomic_run_tag,
    build_atomic_config,
    classify_cache_path,
    generate_configs,
    load_batch_config,
    sample_targets,
    validate_batch_config,
    write_configs,
)


def _base_cfg():
    return {
        "dataset": "ml100k", "mode": "all", "seed": 42,
        "model": {"name": "lightgcn", "overrides": {}},
        "classification": {"k": 10, "popular_ratio": 0.2,
                           "checkpoint": "models/lightgcn/outputs/checkpoints/latest.pt"},
        "attack": {"name": "bandwagon", "ratio": 0.03, "filler_size": 20,
                   "target_items": {"strategy": "specified", "ids": []}},
        "warm_start": {"enabled": True,
                       "checkpoint": "models/lightgcn/outputs/checkpoints/latest.pt"},
        "training": {"epochs": 5, "batch_size": 256, "lr": 0.001,
                     "weight_decay": 0.0001, "neg_ratio": 1, "device": "cpu"},
        "evaluation": {"k": 10,
                       "metrics": [{"target_ndcg@10": "upper"},
                                   {"target_hr@10": "upper"},
                                   {"recall@10": "upper"},
                                   {"ndcg@10": "upper"}],
                       "report_model_utility": True},
        "output": {"dir": "attacks/batch/output"},
        "sampling": {"tiers": ["popular", "normal", "cold"], "per_tier": 3,
                     "strategy": "random", "seed": 42},
    }


class ValidateBatchConfigTest(unittest.TestCase):

    def test_valid_config_passes(self):
        validate_batch_config(_base_cfg())

    def test_missing_sampling_raises(self):
        cfg = _base_cfg()
        del cfg["sampling"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_zero_per_tier_raises(self):
        cfg = _base_cfg()
        cfg["sampling"]["per_tier"] = 0
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)

    def test_unknown_tier_raises(self):
        cfg = _base_cfg()
        cfg["sampling"]["tiers"] = ["hot"]
        with self.assertRaises(ValueError):
            validate_batch_config(cfg)


class LoadBatchConfigTest(unittest.TestCase):

    def test_load_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "batch.yaml"
            p.write_text(json.dumps(_base_cfg()), encoding="utf-8")
            cfg = load_batch_config(p)
        self.assertEqual(cfg["dataset"], "ml100k")

    def test_classify_cache_path(self):
        cfg = _base_cfg()
        p = classify_cache_path(cfg)
        self.assertTrue(p.as_posix().endswith(
            "attacks/bandwagon/data/rec_freq/ml100k/lightgcn_top10.json"))
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_generator -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.batch.generator'`。

- [ ] **Step 3: 最小实现**

`TPA/attacks/batch/__init__.py`（空文件），`TPA/attacks/batch/generator.py`：

```python
"""批量投毒攻击：分层采样 + 原子配置生成。"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIER_NAMES = ("popular", "normal", "cold")


def _resolution_k(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("classification", {}).get("k")
               or cfg.get("training", {}).get("k") or 20)


def validate_batch_config(cfg: Dict[str, Any]) -> None:
    sampling = cfg.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("批跑配置缺少 sampling 段")
    per_tier = sampling.get("per_tier")
    if not isinstance(per_tier, int) or per_tier <= 0:
        raise ValueError("sampling.per_tier 必须为正整数")
    tiers = sampling.get("tiers")
    if not tiers:
        raise ValueError("sampling.tiers 不能为空")
    for tier in tiers:
        if tier not in TIER_NAMES:
            raise ValueError(f"未知分层 {tier!r}，可选 {TIER_NAMES}")
    if sampling.get("strategy", "random") not in ("random", "first"):
        raise ValueError("sampling.strategy 仅支持 random|first")
    if not cfg.get("dataset"):
        raise ValueError("缺少必填字段 dataset")
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


def classify_cache_path(cfg: Dict[str, Any]) -> Path:
    """分类缓存路径：复用攻击自带缓存（与独立运行共享）。"""
    attack = cfg["attack"]["name"]
    dataset = cfg["dataset"]
    model = cfg["model"]["name"]
    k = _resolution_k(cfg)
    return (PROJECT_ROOT / "attacks" / attack / "data" / "rec_freq"
            / dataset / f"{model}_top{k}.json")
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_generator -v`
Expected: `OK`（6 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/__init__.py TPA/attacks/batch/generator.py TPA/tests/test_batch_generator.py
git commit -m "feat(attacks): 批跑配置加载与校验（批量投毒 v1）"
```

---

### Task 2: 分层采样与原子配置生成

**Files:**
- Modify: `TPA/attacks/batch/generator.py`
- Test: `TPA/tests/test_batch_generator.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `validate_batch_config` / `classify_cache_path`。
- Produces: `sample_targets(categories, tiers, per_tier, strategy, seed) -> Dict[str, List[int]]`；`atomic_run_tag(cfg, tier, item_id) -> str`；`build_atomic_config(cfg, item_id, tier, batch_tag) -> Dict`；`generate_configs(cfg, categories, batch_tag) -> List[Dict]`；`write_configs(configs, out_dir) -> List[Path]`。

- [ ] **Step 1: 写失败测试（追加到 test_batch_generator.py）**

```python
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
        categories = {"cold": [20, 21, 22]}
        out = sample_targets(categories, ["cold"], 2, "first", 42)
        self.assertEqual(out["cold"], [20, 21])

    def test_empty_tier_skipped(self):
        out = sample_targets({"cold": []}, ["cold"], 2, "random", 42)
        self.assertEqual(out["cold"], [])


class AtomicConfigTest(unittest.TestCase):

    def test_run_tag_and_target_override(self):
        cfg = _base_cfg()
        atomic = build_atomic_config(cfg, 21, "cold", "2026-08-21-10-00")
        self.assertEqual(
            atomic["run_tag"],
            "bandwagon_ml100k_lightgcn_top10_cold_item21")
        self.assertEqual(atomic["attack"]["target_items"],
                         {"strategy": "specified", "ids": [21]})
        self.assertNotIn("sampling", atomic)
        self.assertEqual(atomic["output"]["dir"],
                         "attacks/batch/output/2026-08-21-10-00/runs")


class GenerateConfigsTest(unittest.TestCase):

    def test_one_config_per_sampled_item(self):
        cfg = _base_cfg()
        categories = {"popular": [1, 2], "normal": [5, 6], "cold": [9]}
        configs = generate_configs(cfg, categories, "2026-08-21-10-00")
        self.assertEqual(len(configs), 3)
        tags = [c["run_tag"] for c in configs]
        self.assertEqual(len(set(tags)), 3)
        self.assertTrue(any(t.endswith("_item9") for t in tags))

    def test_write_configs(self):
        cfg = _base_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_configs(generate_configs(
                cfg, {"cold": [9]}, "t"), Path(tmp))
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            self.assertEqual(paths[0].suffix, ".yaml")
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_generator -v`
Expected: FAIL with `ImportError: cannot import name 'sample_targets'`。

- [ ] **Step 3: 最小实现（追加到 generator.py）**

```python
def sample_targets(categories: Dict[str, List[int]], tiers: List[str],
                   per_tier: int, strategy: str = "random",
                   seed: int = 42) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    out: Dict[str, List[int]] = {}
    for tier in tiers:
        pool = list(categories.get(tier, []))
        if not pool:
            print(f"[batch] 层 {tier} 为空，跳过")
            out[tier] = []
            continue
        out[tier] = (pool[:per_tier] if strategy == "first"
                     else rng.sample(pool, min(per_tier, len(pool))))
    return out


def atomic_run_tag(cfg: Dict[str, Any], tier: str, item_id: int) -> str:
    from training.run_tag import sanitize_run_tag
    name = (f"{cfg['attack']['name']}_{cfg['dataset']}_{cfg['model']['name']}"
            f"_top{_resolution_k(cfg)}_{tier}_item{item_id}")
    return sanitize_run_tag(name)


def build_atomic_config(cfg: Dict[str, Any], item_id: int, tier: str,
                        batch_tag: str) -> Dict[str, Any]:
    atomic = copy.deepcopy(cfg)
    atomic.pop("sampling", None)
    atomic["attack"]["target_items"] = {
        "strategy": "specified", "ids": [int(item_id)]}
    atomic["run_tag"] = atomic_run_tag(cfg, tier, item_id)
    atomic["output"] = {"dir": f"attacks/batch/output/{batch_tag}/runs"}
    return atomic


def generate_configs(cfg: Dict[str, Any], categories: Dict[str, List[int]],
                     batch_tag: str) -> List[Dict[str, Any]]:
    sampling = cfg["sampling"]
    targets = sample_targets(
        categories, sampling["tiers"], sampling["per_tier"],
        sampling.get("strategy", "random"), sampling.get("seed", 42))
    configs: List[Dict[str, Any]] = []
    for tier, items in targets.items():
        for item in items:
            configs.append(build_atomic_config(cfg, item, tier, batch_tag))
    return configs


def write_configs(configs: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    import yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for cfg in configs:
        p = out_dir / f"{cfg['run_tag']}.yaml"
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        paths.append(p)
    return paths
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_generator -v`
Expected: `OK`（12 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/generator.py TPA/tests/test_batch_generator.py
git commit -m "feat(attacks): 分层采样与原子投毒配置生成（批量投毒 v1）"
```

---

### Task 3: 结果整合（aggregate.py）

**Files:**
- Create: `TPA/attacks/batch/aggregate.py`
- Test: `TPA/tests/test_batch_aggregate.py`

**Interfaces:**
- Consumes: 无（运行目录结构来自 Task 2 的 run_tag 命名）。
- Produces: `parse_tier_item(run_tag) -> Tuple[str, int] | None`；`load_best_metrics(run_dir) -> Dict | None`；`build_results_rows(runs_root, k) -> List[Dict]`；`write_results_csv(rows, k, path)`；`tier_summary(rows, k) -> Dict`；`write_summary_md(batch_tag, summary, clean_baseline, k, path)`。

- [ ] **Step 1: 写失败测试**

`TPA/tests/test_batch_aggregate.py`：

```python
"""批量投毒：结果整合单测（fixture 目录，CPU）。"""
import json
import tempfile
import unittest
from pathlib import Path

from attacks.batch.aggregate import (
    build_results_rows,
    parse_tier_item,
    tier_summary,
    write_results_csv,
    write_summary_md,
)


class ParseTierItemTest(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(
            parse_tier_item("bandwagon_ml100k_lightgcn_top10_cold_item21"),
            ("cold", 21))
        self.assertIsNone(parse_tier_item("other"))


class AggregateTest(unittest.TestCase):

    def _make_run(self, root, run_tag, best):
        d = root / run_tag
        d.mkdir(parents=True)
        (d / "history.json").write_text(
            json.dumps({"history": [], "best": best}, ensure_ascii=False),
            encoding="utf-8")

    def test_build_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_run(
                root, "bandwagon_ml100k_lightgcn_top10_cold_item21",
                {"target_hr@10": 0.5, "target_ndcg@10": 0.4,
                 "recall@10": 0.3, "ndcg@10": 0.2})
            self._make_run(
                root, "bandwagon_ml100k_lightgcn_top10_cold_item22",
                {"target_hr@10": 0.7, "target_ndcg@10": 0.6,
                 "recall@10": 0.31, "ndcg@10": 0.21})
            self._make_run(
                root, "bandwagon_ml100k_lightgcn_top10_normal_item5",
                {"target_hr@10": 0.9, "target_ndcg@10": 0.8,
                 "recall@10": 0.32, "ndcg@10": 0.22})
            rows = build_results_rows(root, 10)
            self.assertEqual(len(rows), 3)
            summary = tier_summary(rows, 10)
            self.assertAlmostEqual(summary["cold"]["target_hr@10"]["mean"], 0.6)
            self.assertAlmostEqual(summary["cold"]["target_ndcg@10"]["mean"], 0.5)
            self.assertEqual(summary["cold"]["target_hr@10"]["n"], 2)
            write_results_csv(rows, 10, root / "results.csv")
            write_summary_md(
                "2026-08-21-10-00", summary,
                {"recall@10": 0.35, "ndcg@10": 0.25}, 10, root / "summary.md")
            csv_text = (root / "results.csv").read_text(encoding="utf-8")
            self.assertIn("target_ndcg@10", csv_text)
            md_text = (root / "summary.md").read_text(encoding="utf-8")
            self.assertIn("0.6000", md_text)
            self.assertIn("Clean 基线", md_text)

    def test_missing_history_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad_run").mkdir()
            self.assertEqual(build_results_rows(root, 10), [])
```

- [ ] **Step 2: 运行测试确认失败**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_aggregate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'attacks.batch.aggregate'`。

- [ ] **Step 3: 最小实现**

`TPA/attacks/batch/aggregate.py`：

```python
"""批量投毒：结果整合（results.csv + 按层 mean±std 的 summary.md）。"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple


RUN_TAG_RE = re.compile(r"_top\d+_(popular|normal|cold)_item(\d+)$")


def parse_tier_item(run_tag: str) -> Optional[Tuple[str, int]]:
    m = RUN_TAG_RE.search(run_tag)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_best_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("best")


def build_results_rows(runs_root: Path, k: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_tier_item(run_dir.name)
        best = load_best_metrics(run_dir)
        if parsed is None or best is None:
            continue
        tier, item = parsed
        row: Dict[str, Any] = {
            "run_tag": run_dir.name, "tier": tier, "target_item": item}
        for key in (f"target_hr@{k}", f"target_ndcg@{k}",
                    f"recall@{k}", f"ndcg@{k}"):
            row[key] = best.get(key)
        rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, Any]], k: int, path: Path) -> None:
    fieldnames = ["run_tag", "tier", "target_item",
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
    out: Dict[str, Any] = {}
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


def write_summary_md(batch_tag: str, summary: Dict[str, Any],
                     clean_baseline: Optional[Dict[str, float]], k: int,
                     path: Path) -> None:
    lines = [
        f"# 批量投毒攻击结果汇总（batch_tag={batch_tag}）",
        "",
        "| 分层 | 指标 | mean | std | n |",
        "|---|---|---|---|---|",
    ]
    for tier in sorted(summary):
        for metric in (f"target_hr@{k}", f"target_ndcg@{k}"):
            s = summary[tier][metric]
            lines.append(f"| {tier} | {metric} | {_fmt(s['mean'])} | "
                         f"{_fmt(s['std'])} | {s['n']} |")
    if clean_baseline:
        r = clean_baseline.get(f"recall@{k}", float("nan"))
        n = clean_baseline.get(f"ndcg@{k}", float("nan"))
        lines += [
            "",
            f"Clean 基线（w_clean）：recall@{k}={r:.4f}，ndcg@{k}={n:.4f}",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_batch_aggregate -v`
Expected: `OK`（3 个用例）。

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/aggregate.py TPA/tests/test_batch_aggregate.py
git commit -m "feat(attacks): 批量投毒结果整合（results.csv + 按层汇总）"
```

---

### Task 4: 调度器与 CLI（runner.py + run.py）

**Files:**
- Create: `TPA/attacks/batch/runner.py`、`TPA/attacks/batch/run.py`
- 验证：E2E mini（Task 6），不另写单测（涉及真实模型训练，超出单测范围）。

**Interfaces:**
- Consumes: Task 1 `classify_cache_path` / `load_batch_config`；Task 2 `generate_configs` / `write_configs`；Task 3 聚合函数。
- Produces: `ensure_classify_cache(cfg) -> Dict`；`run_atomic(atomic_cfg, stage) -> None`；`run_batch(cfg, batch_tag, configs_dir, runs_root, skip_classify, max_targets, dry_run) -> None`；`compute_clean_baseline(cfg, k) -> Dict[str, float]`；`main()` CLI。

- [ ] **Step 1: 实现 `ensure_classify_cache`**

```python
def ensure_classify_cache(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """确保分类缓存存在（缺失时调攻击 classify.main 生成一次），返回缓存 dict。"""
    cache_path = classify_cache_path(cfg)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    attack = cfg["attack"]["name"]
    try:
        classify_mod = importlib.import_module(f"attacks.{attack}.classify")
    except ImportError:
        raise RuntimeError(
            f"攻击 {attack} 无 classify 模块；v1 支持 bandwagon/pgd/random，"
            "tpa 待 v2 适配")
    base = copy.deepcopy(cfg)
    base.pop("sampling", None)
    base["mode"] = "classify"
    classify_mod.main(base)
    return json.loads(cache_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: 实现 `run_atomic` / `run_batch`**

```python
def run_atomic(atomic_cfg: Dict[str, Any], stage: str) -> None:
    """执行单个原子实验的 data 或 model 阶段（in-process 复用攻击模块）。"""
    attack = atomic_cfg["attack"]["name"]
    mod_name = {"data": "generate", "model": "fit"}[stage]
    mod = importlib.import_module(f"attacks.{attack}.{mod_name}")
    mod.main(atomic_cfg)


def run_batch(cfg, batch_tag, configs_dir, runs_root, cache,
              max_targets=None, dry_run=False) -> None:
    configs = generate_configs(cfg, cache, batch_tag)
    if max_targets is not None:
        configs = configs[:max_targets]
    write_configs(configs, configs_dir)
    print(f"[batch] 生成原子配置 {len(configs)} 个 -> {configs_dir}")
    if dry_run:
        return
    for i, atomic in enumerate(configs, 1):
        print(f"[batch] ({i}/{len(configs)}) {atomic['run_tag']}")
        run_atomic(atomic, "data")
        run_atomic(atomic, "model")
```

- [ ] **Step 3: 实现 `compute_clean_baseline`**

```python
def compute_clean_baseline(cfg: Dict[str, Any], k: int) -> Dict[str, float]:
    """用 w_clean 在 clean 数据上算 recall@k / ndcg@k（批量开始前算一次）。"""
    attack = cfg["attack"]["name"]
    model_name = cfg["model"]["name"]
    registry = importlib.import_module(f"attacks.{attack}.registry")
    import torch
    from attacks.bandwagon.fit import build_training_config
    from attacks.bandwagon.generate import DEFAULT_RAW_META, load_meta
    from evaluation.attack_eval import ranking_scores
    from evaluation.metrics import build_train_mask_indices, compute_metrics
    from training.paths import resolve_from_root

    meta = load_meta(Path(str(DEFAULT_RAW_META).format(dataset=cfg["dataset"])))
    train_cfg = build_training_config(cfg, cfg["dataset"], model_name)
    model = registry.get_model_cls(model_name)(
        train_cfg, meta["num_users"], meta["num_items"], None)
    ckpt = resolve_from_root(cfg["classification"]["checkpoint"], PROJECT_ROOT)
    model.load_state_dict(torch.load(ckpt, map_location=model._device,
                                     weights_only=True)["model_state_dict"])
    scores, users, test_pos = ranking_scores(model, meta["test_pairs"])
    rows, cols = build_train_mask_indices(meta["user_items"], users)
    res = compute_metrics(scores, meta["user_items"], test_pos, k=k,
                          mask_indices=(rows, cols),
                          topk_device=getattr(model._device, "type", "cpu"))
    return res
```

> 注：`get_model_cls` 需要 `edge_index` 参数；LightGCN/MF 构造签名兼容 `edge_index=None`，WMF 忽略该参数。v1 只跑 lightgcn。

- [ ] **Step 4: 实现 `run.py` CLI**

```python
"""批量投毒攻击编排：generate / run / aggregate / all。"""
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from training.run_tag import resolve_run_tag
    from attacks.batch.aggregate import (
        build_results_rows, tier_summary, write_results_csv, write_summary_md)
    from attacks.batch.generator import classify_cache_path, load_batch_config
    from attacks.batch.runner import (
        compute_clean_baseline, ensure_classify_cache, run_atomic, run_batch)
    import yaml

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

    if args.mode in ("generate", "all"):
        if args.skip_classify and not classify_cache_path(cfg).exists():
            raise FileNotFoundError(
                f"缓存不存在：{classify_cache_path(cfg)}（--skip-classify 需要已有缓存）")
        cache = ensure_classify_cache(cfg)
        run_batch(cfg, batch_tag, configs_dir, runs_root,
                  cache,
                  max_targets=args.max_targets,
                  dry_run=(args.mode == "generate") or args.dry_run)
    if args.mode == "run":
        if not configs_dir.exists():
            raise FileNotFoundError(
                f"未找到生成目录 {configs_dir}，请先 --mode generate")
        for p in sorted(configs_dir.glob("*.yaml")):
            atomic = yaml.safe_load(p.read_text(encoding="utf-8"))
            print(f"[batch] run {atomic['run_tag']}")
            run_atomic(atomic, "data")
            run_atomic(atomic, "model")
    if args.mode in ("aggregate", "all") and not args.dry_run:
        rows = build_results_rows(runs_root, k)
        write_results_csv(rows, k, out_root / "results.csv")
        clean = compute_clean_baseline(cfg, k)
        write_summary_md(batch_tag, tier_summary(rows, k), clean, k,
                         out_root / "summary.md")
        print(f"[batch] 整合完成：{len(rows)} 个原子实验 -> {out_root}")
```

- [ ] **Step 5: 提交**

```bash
git add TPA/attacks/batch/runner.py TPA/attacks/batch/run.py
git commit -m "feat(attacks): 批量投毒调度器与 CLI（批量投毒 v1）"
```

---

### Task 5: 批跑配置与文档

**Files:**
- Create: `TPA/attacks/batch/config.yaml`、`TPA/attacks/batch/docs/USAGE.md`、`TPA/attacks/batch/docs/DESIGN.md`

- [ ] **Step 1: 写 `config.yaml`（mini 默认值）**

```yaml
# 批量投毒攻击批跑配置（v1 mini）
# 说明：sampling 段为批量扩展；其余字段与 attacks/bandwagon/config.yaml 同构。
dataset: ml100k
mode: all
seed: 42
run_tag: null

model:
  name: lightgcn
  overrides: {}

classification:
  k: 10
  popular_ratio: 0.2
  checkpoint: models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt

attack:
  name: bandwagon
  ratio: 0.03
  filler_size: 20
  target_items: {strategy: specified, ids: []}

warm_start:
  enabled: true
  checkpoint: models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt

training:
  epochs: 5
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  neg_ratio: 1
  device: cuda

evaluation:
  k: 10
  metrics:
  - target_ndcg@10: upper
  - target_hr@10: upper
  - recall@10: upper
  - ndcg@10: upper
  report_model_utility: true
  checkpoint_mode: per_metric

output:
  dir: attacks/batch/output

sampling:
  tiers: [popular, normal, cold]
  per_tier: 2
  strategy: random
  seed: 42
```

- [ ] **Step 2: 验证配置可加载**

Run（在 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe -c "from pathlib import Path; from attacks.batch.generator import load_batch_config; c=load_batch_config(Path('attacks/batch/config.yaml')); print(len(c['sampling']['tiers']), c['sampling']['per_tier'])"`
Expected: `3 2`。

- [ ] **Step 3: 写 USAGE.md 与 DESIGN.md**

USAGE.md 包含：项目结构、前置条件（clean checkpoint / 数据集 meta）、
运行方式（`python attacks/batch/run.py --config attacks/batch/config.yaml`、
`--dry-run`、`--max-targets`、`--batch-tag`）、输出目录说明、config 字段表。
DESIGN.md 引用本计划对应 spec，说明原子实验语义与 v2 迭代方向
（多数据集 × split_seed 网格、pgd/random/tpa）。

- [ ] **Step 4: 提交**

```bash
git add TPA/attacks/batch/config.yaml TPA/attacks/batch/docs
git commit -m "docs(attacks): 批量投毒批跑配置与使用/设计文档"
```

---

### Task 6: mini 投毒实验准备与端到端验证

**Files:** 无代码改动（产出为 gitignored 实验产物，不入库）。

- [ ] **Step 1: 训练 ml100k clean lightgcn checkpoint（w_clean）**

Run（在 `G:\Idea`，用内联脚本）：
训练 50 epoch（batch=256，emb_dim=64，device=cuda），保存
`models/lightgcn/outputs/clean-ml100k/checkpoints/latest.pt`
（含 `model_state_dict`）。Expected：loss 下降，无 NaN，文件生成。

- [ ] **Step 2: 冒烟批量（最小规模）**

复制 `attacks/batch/config.yaml` 到 `tmp/batch_smoke.yaml`，改
`training.epochs: 1`、`sampling.per_tier: 1`、`sampling.tiers: [cold]`、
`training.device: cpu`，然后 Run（在 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --config ../tmp/batch_smoke.yaml`
Expected：classify 一次 → 1 个原子实验 data+model 跑通；
`attacks/batch/output/{batch_tag}/` 下 configs/ 1 个文件、runs/ 1 个目录、
results.csv 1 行、summary.md 含 cold 层汇总与 Clean 基线。

- [ ] **Step 3: 正式 mini 批量**

用仓库默认 `attacks/batch/config.yaml`（epochs=5、per_tier=2、三层、GPU）运行
`python attacks/batch/run.py`。Expected：6 个原子实验全部跑完，
`configs/` 与 `runs/` 各 6 项，results.csv 6 行，summary.md 三层均有 mean±std。

- [ ] **Step 4: 记录结果并汇报**

把 `results.csv` 与 `summary.md` 的关键数字（各层 target_hr@10 / target_ndcg@10
的 mean±std、clean 基线）整理给用户。

---

### Task 7: 全量回归与收尾

- [ ] **Step 1: 全量单测**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`
Expected：全部通过（含既有 105 个 + 新增 batch 测试）。

- [ ] **Step 2: 自查提交**

Run（在 `G:\Idea`）：`git status --short` 与 `git log --oneline -10`。
Expected：仅本计划相关文件提交；实验产物（outputs / data/poisoned / *.pt）未入库。
