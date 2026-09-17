# 模型指标 K 统一装配——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec（`docs/superpowers/specs/2026-09-17-model-metric-k-unification-design.md`）把"K 的解析"收敛到 `training/config_utils.py` 的唯一装配入口，使任何模型声明 `k=X` 后训练、攻击、报告、产物中只出现 `@X`。

**Architecture:** `load_config`（canonicalize）→ `flatten_model_config`（分片合并 + 顶层 canonical 键覆盖 + 缺省策略）→ `apply_k`（`{k}` 单次展开）→ `TrainingConfig`。三个模型入口、`models/registry.load_model_config`、攻击 `fit.py`、`wmf/report.py` 全部走这一条路径；`DEFAULT_K` 是全仓库唯一的兼容兜底常量。

**Tech Stack:** Python 3.12（仓库 `.venv`）、PyYAML、torch 2.5.1、stdlib `unittest`（不新增第三方依赖）。

## Global Constraints

- canonical 唯一事实源仍是 `TPA/docs/config-template.unified.yaml`；本计划不新增配置键。
- 不修改历史产物（`outputs/` 下已有 run 目录）、不改攻击/训练算法、不改指标计算语义、不改 `training/framework.py` 抽象接口。
- K 唯一权威：canonical 顶层 `k`；legacy `evaluation.k / training.k / classification.k` 只兼容读取。
- 缺 `k` → `UserWarning` + 兼容默认 `DEFAULT_K`；缺 `dataset` → `ValueError`（不再静默回退 `gowalla`）。
- `{k}` 只在统一装配入口展开一次；业务代码不得再出现裸 `20` 作为 K 来源。
- 无 `evaluation.metrics` 的旧模型允许按 resolved K 派生 `recall@K / ndcg@K` 并告警。
- 注释中文，来源标注沿用 `[paper]/[ai]/[unreported]/[官方代码]`。
- 测试命令（工作目录 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest <模块> -v`；
  全量：`G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`。
- 提交信息 Conventional Commits 中文；只 `git add` 明确路径，禁止 `git add -f` / `git add -A`。

## 文件结构（新增 / 修改）

新增：

```
TPA/tests/test_metric_k_unification.py                       L1–L4 测试（A3 契约）
docs/superpowers/specs/2026-09-17-model-metric-k-unification-design.md   （已写）
docs/superpowers/plans/2026-09-17-model-metric-k-unification.md          （本文件）
```

修改：

```
TPA/training/config_utils.py                 DEFAULT_K + flatten_model_config + build_training_config_from_yaml
TPA/models/lightgcn/train.py                 内联展平 → 统一装配；eval_k 兜底改 DEFAULT_K；main 增 config_path
TPA/models/mf/train.py                       同上
TPA/models/wmf/train.py                      内联展平 → 统一装配（保留 epochs_override）
TPA/models/lightgcn/main.py / mf/main.py     新增 --config（对齐 wmf 既有 CLI，供 smoke 使用）
TPA/models/wmf/report.py                     内联展平 → 统一装配；指标名不写死
TPA/models/registry.py                       load_model_config 返回 canonical + apply_k
TPA/attacks/{random,bandwagon,pgd,tpa,uba}/fit.py   K 去裸 20 + 兜底指标按 resolved K 派生
TPA/attacks/uba/generate.py                  全局 K 读取去裸 20（hit_k 语义不动）
TPA/models/{lightgcn,mf,wmf}/docs/USAGE.md   指标表改为 {k} 语义
```

---

## P1 `config_utils` 统一装配入口

**Files:**
- Modify: `TPA/training/config_utils.py`
- Test: `TPA/tests/test_metric_k_unification.py`（L1 部分，P6 汇总）

**Interfaces:**
- Produces: `DEFAULT_K: int`、`flatten_model_config(raw, *, top_level_keys, require_k, require_dataset) -> dict`、`build_training_config_from_yaml(path, *, extra_overrides=None) -> TrainingConfig`
- Consumes: 既有 `canonicalize_config` / `load_config` / `apply_k` / `_warn`

- [ ] **Step 1: 写 L1 失败测试**（见 P6 的 L1 代码块，先只建该文件）

```python
def test_flatten_keeps_top_level_k(self):
    flat = flatten_model_config(_canonical(k=10))
    self.assertEqual(flat["k"], 10)
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: FAIL / ERROR —— `ImportError: cannot import name 'flatten_model_config'`

- [ ] **Step 3: 加 `DEFAULT_K` 并让 `resolve_k` 引用它**

```python
# 全仓库唯一的 K 兼容兜底常量。
# 为什么：过去 `models/*/train.py`、`attacks/*/fit.py` 各写裸 20，canonical 化后
# 顶层 k 被丢弃也不会报错，直接变成"第二权威"（见 spec §1）。集中到这里后，
# 任何需要 K 兜底的位置只能引用该常量，审计时只需检查一处。
# 使用举例：resolve_k(cfg, default=DEFAULT_K) / ev_defaults.get("k", DEFAULT_K)
DEFAULT_K: int = 20


def resolve_k(cfg: Dict[str, Any], default: int = DEFAULT_K) -> int:
```

- [ ] **Step 4: 实现 `flatten_model_config` 与 `build_training_config_from_yaml`**

```python
# 模型 config.yaml 的分片与顶层 canonical 键：唯一展平规则。
_CONFIG_SECTIONS = ("data", "model", "training", "evaluation")
_TOP_LEVEL_KEYS = ("dataset", "k", "seed", "mode", "run_tag")


def flatten_model_config(
    raw: Dict[str, Any],
    *,
    top_level_keys: Sequence[str] = _TOP_LEVEL_KEYS,
    require_k: bool = True,
    require_dataset: bool = True,
) -> Dict[str, Any]:
    """canonical 模型配置 → TrainingConfig 可消费的扁平 dict（模型入口唯一装配）。

    为什么这样做（设计动机）：
        models/{lightgcn,mf,wmf}/train.py 曾各自内联展平 data/model/training/evaluation
        四个分片；2026-09-02 canonical 化把 `dataset`/`k` 提到顶层后，顶层键被静默
        丢弃 → apply_k 回退 default=20（评测/选优/checkpoint 全按 20），dataset 还会
        静默回退 gowalla。本函数是该类回归的唯一修复点。

    功能是什么（优先级）：
        1) 分片键先合并；2) 顶层 canonical 键后写入 → 顶层赢；
        3) 缺 dataset 直接 ValueError；4) 缺 k 发 UserWarning 并按 DEFAULT_K 继续；
        5) 最后 apply_k 展开 `{k}`（只展开一次）。

    参考出处：spec `2026-09-17-model-metric-k-unification-design.md` §4.2/§4.3/§4.4。

    使用举例：
        raw = load_config("models/lightgcn/config.yaml")
        flat = flatten_model_config(raw)
        # {"dataset": "yelp2018", "k": 10,
        #  "metrics": [{"recall@10": "upper"}, {"ndcg@10": "upper"}], ...}
    """
    flat: Dict[str, Any] = {}
    for section in _CONFIG_SECTIONS:
        sec = raw.get(section)
        if isinstance(sec, dict):
            flat.update(sec)          # 兼容读取旧分片键（含 legacy evaluation.k）
    for key in top_level_keys:
        if raw.get(key) is not None:
            flat[key] = raw[key]      # canonical 顶层键覆盖分片（顶层赢）

    if require_dataset and not flat.get("dataset"):
        raise ValueError(
            "[config] 缺少 canonical 顶层 dataset；禁止静默回退默认数据集"
            f"（已有顶层键：{sorted(raw)}）"
        )
    if require_k and flat.get("k") is None:
        _warn(f"[config] 缺少 canonical 顶层 k：按兼容默认 {DEFAULT_K} 继续；"
              "请在 config.yaml 显式声明顶层 k")
    return apply_k(flat)


def build_training_config_from_yaml(
    path,
    *,
    extra_overrides: Dict[str, Any] | None = None,
):
    """模型主入口唯一装配点：load_config → flatten_model_config → TrainingConfig。

    为什么要集中：把"哪些键必须从顶层带下来"变成一处可测的代码，而不是三个
    main() 里各写一遍的 for 循环（那正是本次事故的来源）。

    使用举例：
        config = build_training_config_from_yaml("models/lightgcn/config.yaml",
                                                 extra_overrides={"epochs": 1})
    """
    from training.framework import TrainingConfig  # 延迟 import，避免环依赖

    raw = load_config(path)
    flat = flatten_model_config(raw)
    if extra_overrides:
        flat.update(extra_overrides)
    return TrainingConfig(overrides=flat)
```

- [ ] **Step 5: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: L1 全 PASS（L2–L4 仍 FAIL，属后续任务）
Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_config_utils -v`
Expected: PASS（`apply_k` 语义未变）

---

## P2 lightgcn / mf / wmf 接入统一装配

**Files:**
- Modify: `TPA/models/lightgcn/train.py`、`TPA/models/mf/train.py`、`TPA/models/wmf/train.py`
- Modify: `TPA/models/lightgcn/main.py`、`TPA/models/mf/main.py`（新增 `--config`，对齐 wmf 既有 CLI）

**Interfaces:**
- Consumes: P1 的 `flatten_model_config` / `build_training_config_from_yaml` / `DEFAULT_K`
- Produces: 三个 `main(tag, resume, [config_path], [epochs_override])`，内部配置来源唯一

- [ ] **Step 1: 写 L2/L4 失败测试**（P6 的 `test_lightgcn_entry_yields_k10_yelp2018`、`test_mf_wmf_entry_preserve_declared_k`、`test_model_entrypoints_use_shared_assembler`）

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: FAIL —— lightgcn 入口 `k=20`、`dataset is None`

- [ ] **Step 3: lightgcn 入口替换内联展平**

```python
def main(tag: str | None = None, resume: bool = False,
         config_path: str | None = None):
    # 配置装配唯一走 config_utils：顶层 dataset/k 必须带下来（见 spec I1/I3）
    config_path = config_path or os.path.join(
        os.path.dirname(__file__), "config.yaml")
    config = build_training_config_from_yaml(config_path)
```

（删除原 `if not os.path.exists(...)` / `for section in [...]` 整段；配置缺失由 `load_config` 抛 `FileNotFoundError`，不再静默回退默认配置）
并改 `self.eval_k = config.get("k", DEFAULT_K)`，import：

```python
from training.config_utils import DEFAULT_K, build_training_config_from_yaml
```

- [ ] **Step 4: mf 入口同样替换；wmf 入口替换但保留 `epochs_override`**

```python
    config = build_training_config_from_yaml(
        config_path,
        extra_overrides=({"epochs": epochs_override}
                         if epochs_override is not None else None))
```

- [ ] **Step 5: 两个 CLI 补 `--config`**（`lightgcn/main.py`、`mf/main.py`）

```python
    parser.add_argument("--config", type=str, default=None,
                        help="config.yaml 路径（缺省=模型目录下 config.yaml，供冒烟/实验隔离用）")
    ...
    main(tag=args.tag, resume=args.resume, config_path=args.config)
```

- [ ] **Step 6: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: L1/L2 相关用例 PASS
Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_history_completeness -v`
Expected: PASS

---

## P3 `models/registry.load_model_config` 接入

**Files:**
- Modify: `TPA/models/registry.py`
- Test: `TPA/tests/test_metric_k_unification.py::test_load_model_config_expands_k`

**Interfaces:**
- Produces: `load_model_config(name, overrides=None) -> dict`，返回 canonical + `apply_k` 后的嵌套配置（结构不变，`model.* / training.* / evaluation.*` 读取路径不变）

- [ ] **Step 1: 写失败测试**

```python
def test_load_model_config_expands_k(self):
    cfg = load_model_config("lightgcn")
    self.assertEqual(cfg["evaluation"]["k"], 10)
    self.assertNotIn("{k}", str(cfg["evaluation"]["metrics"]))
```

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: FAIL —— `KeyError: 'k'`

- [ ] **Step 3: 实现**

```python
    import yaml
    from training.config_utils import apply_k, canonicalize_config

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = canonicalize_config(yaml.safe_load(f))

    if overrides:
        if "model" in cfg and isinstance(cfg["model"], dict):
            cfg["model"].update(overrides)
        else:
            cfg.update(overrides)
    # 返回前展开 {k}：调用方（attacks/*/fit.py、uba/estimate.py）读到的
    # evaluation.k / metrics 必须已是最终 K，不得再自行回退 20。
    return apply_k(cfg)
```

- [ ] **Step 4: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification tests.test_model_registry -v`
Expected: PASS

- [ ] **Step 5: 提交（P1–P3 作为一个逻辑变更）**

```bash
git add TPA/training/config_utils.py TPA/models/registry.py TPA/models/lightgcn/train.py TPA/models/mf/train.py TPA/models/wmf/train.py TPA/models/lightgcn/main.py TPA/models/mf/main.py
git commit -m "fix(config): 模型入口统一装配 K，顶层 dataset/k 不再丢失"
```

---

## P4 攻击 `fit.py` 去裸 20

**Files:**
- Modify: `TPA/attacks/{random,bandwagon,pgd,tpa,uba}/fit.py`
- Modify: `TPA/attacks/uba/generate.py`（仅全局 K 读取；`hit_k` 不动）
- Test: `TPA/tests/test_metric_k_unification.py::test_attack_fit_k_follows_config`、`::test_metrics_fallback_uses_resolved_k`

**Interfaces:**
- Consumes: P1 `DEFAULT_K` / `resolve_k`，P3 `load_model_config`
- Produces: `resolve_metrics_cfg(config, model_name) -> list`（兜底按 resolved K 派生并告警）

- [ ] **Step 1: 写失败测试**（P6 的 L3 两个用例）

- [ ] **Step 2: 运行确认失败**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: FAIL —— 兜底指标为 `recall@20`（与 `k:5` 不一致）

- [ ] **Step 3: `build_training_config` 的 K 来源唯一化**

```python
from training.config_utils import DEFAULT_K, resolve_k
...
        # K 优先级（spec §4.3）：攻击配置顶层 k > 模型配置已解析的 k > DEFAULT_K
        "k": resolve_k(config, default=ev_defaults.get("k", DEFAULT_K)),
```

- [ ] **Step 4: `resolve_metrics_cfg` 兜底按 resolved K 派生**

```python
def resolve_metrics_cfg(config: Dict[str, Any], model_name: str) -> list:
    """攻击配置 evaluation.metrics 优先；缺省取模型自身 resolved metrics。

    都没有时（itemcf/itemae/ncf/multvae/cml 等旧模型未声明 metrics）按**已解析的 K**
    派生 recall@K/ndcg@K 并打印告警——禁止再写死 recall@20/ndcg@20（spec I3）。
    """
    model_cfg = load_model_config(
        model_name, overrides=config.get("model", {}).get("overrides"))
    metrics = config.get("evaluation", {}).get("metrics")
    if metrics is None:
        metrics = model_cfg.get("evaluation", {}).get("metrics")
    if metrics is None:
        k = resolve_k(config, default=resolve_k(model_cfg, default=DEFAULT_K))
        metrics = [f"recall@{k}", f"ndcg@{k}"]
        print(f"[fit] [!] {model_name} 未声明 evaluation.metrics，"
              f"按 resolved K={k} 派生 {metrics}")
    return metrics
```

- [ ] **Step 5: 其余 K 读取统一引用 `DEFAULT_K`**

```python
    k = cfg.get("k", DEFAULT_K)
```

（`random/bandwagon/pgd/tpa/uba` 的 `fit.py` 各 2 处；`uba/generate.py` 的
`int(config.get("k", 20))` 3 处改 `int(config.get("k", DEFAULT_K))`；
`hit_k` 是 UBA 论文自有超参，保持 `20` 不动）

- [ ] **Step 6: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification tests.test_attack_eval tests.test_uba_generate -v`
Expected: PASS

---

## P5 `wmf/report.py` 接入 resolved metrics

**Files:**
- Modify: `TPA/models/wmf/report.py`
- Test: `TPA/tests/test_metric_k_unification.py::test_wmf_report_metric_names_follow_history`

**Interfaces:**
- Produces: `_metric_names_from_history(history: list[dict]) -> list[str]`（纯函数，无绘图副作用）、
  `plot_training_curve(history, curve_path, metric_names=None)`

- [ ] **Step 1: 写失败测试**

```python
def test_wmf_report_metric_names_follow_history(self):
    from models.wmf.report import _metric_names_from_history
    names = _metric_names_from_history(
        [{"epoch": 1, "train_loss": 1.0, "val_loss": 1.0,
          "rank": 0.3, "recall@10": 0.1, "ndcg@10": 0.2}])
    self.assertEqual(names, ["rank", "recall@10", "ndcg@10"])
    self.assertNotIn("@20", " ".join(names))
```

- [ ] **Step 2: 运行确认失败** → `ImportError: cannot import name '_metric_names_from_history'`

- [ ] **Step 3: 实现纯函数并替换硬编码**

```python
def _metric_names_from_history(history):
    """从 history 的实际键推导曲线指标（不写死 @K，spec I2/I4）。

    为什么不用固定元组：指标名由配置 resolved 决定；写死 recall@20/ndcg@20
    会在 k=10 的 run 上画出空曲线（本次事故的同类表现）。
    """
    skip = {"epoch", "train_loss", "val_loss", "epoch_seconds"}
    names = []
    for entry in history:
        for key in entry:
            if key not in skip and key not in names:
                names.append(key)
    return names


def plot_training_curve(history, curve_path, metric_names=None):
    """两联图：左=Eq.(3) 全量损失曲线；右=排序指标曲线（指标名来自 history）。"""
    ...
    for m in (metric_names or _metric_names_from_history(history)):
```

- [ ] **Step 4: `_load_latest_model` 改用统一装配**

```python
    from training.config_utils import build_training_config_from_yaml

    config = build_training_config_from_yaml(config_path)
```

- [ ] **Step 5: 运行确认通过**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: PASS

---

## P6 L1/L2/L3/L4 测试汇总（A3 核心交付）

**Files:**
- Create: `TPA/tests/test_metric_k_unification.py`

**Interfaces:**
- Consumes: P1–P5 全部产物
- Produces: L1 单元 / L2 全模型契约 / L3 攻击一致性 / L4 入口守卫

完整用例清单（每条对应 spec §5）：

| 层 | 用例 | 守住什么 |
|---|---|---|
| L1 | `test_flatten_keeps_top_level_k` | 顶层 k 必须带下来 |
| L1 | `test_flatten_keeps_top_level_dataset` | 顶层 dataset 必须带下来 |
| L1 | `test_flatten_expands_metric_templates` | `{k}` 单次展开 |
| L1 | `test_top_level_k_beats_section_k` | canonical 顶层赢 |
| L1 | `test_legacy_section_k_mapped` | legacy 只兼容读取 |
| L1 | `test_missing_k_warns` | 无静默回退 |
| L1 | `test_missing_dataset_raises` | 无静默数据集回退 |
| L1 | `test_does_not_mutate_input` | 纯函数 |
| L2 | `test_every_model_config_metric_k_matches_resolved_k` | A3 契约：canonicalize→resolve_k→apply_k→metrics 比对 |
| L2 | `test_contract_catches_hardcoded_metric_k` | 契约测试非恒真（`k:10 + recall@20` 必须红） |
| L2 | `test_lightgcn_entry_yields_k10_yelp2018` | 本次事故复现用例 |
| L2 | `test_mf_wmf_entry_preserve_declared_k` | 统一的是"听顶层 k"而非固定 10 |
| L3 | `test_load_model_config_expands_k` | 攻击侧不再拿到 `{k}` |
| L3 | `test_attack_fit_k_follows_config` | 5 个攻击 fit 跟随攻击配置 k |
| L3 | `test_metrics_fallback_uses_resolved_k` | 兜底不含 `@20` |
| L4 | `test_model_entrypoints_use_shared_assembler` | 内联展平不回流 |
| L4 | `test_wmf_report_metric_names_follow_history` | 报告不写死指标名 |

关键实现片段（契约测试不得扫描 YAML 文本，必须走真实解析链）：

```python
def _metric_k_errors(raw: dict) -> list:
    """canonicalize → resolve_k → apply_k → 检查最终 metrics（spec I1/I4）。"""
    canonical = canonicalize_config(raw)
    k = resolve_k(canonical)
    resolved = apply_k(canonical)
    metrics = (resolved.get("evaluation") or {}).get(
        "metrics", resolved.get("metrics"))
    errors = []
    for name in (parse_metrics(metrics) if metrics else []):
        if "{k}" in name:
            errors.append(f"模板未展开: {name}")                 # 绕过统一装配
            continue
        mk = metric_k(name)
        if mk is not None and mk != k:
            errors.append(f"{name} 的 K={mk} 与顶层 k={k} 不一致")  # 第二权威
    return errors


def test_contract_catches_hardcoded_metric_k(self):
    bad = {"dataset": "toy", "k": 10,
           "evaluation": {"metrics": [{"recall@20": "upper"}]}}
    self.assertTrue(_metric_k_errors(bad))    # 证明契约测试能抓本次事故
```

- [ ] **Step 1: 建测试文件（先 RED）**
- [ ] **Step 2: 逐条确认失败原因正确**（不是语法错误）

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected（修复前）：用户点名的 6 条必须失败且原因明确：

```
test_flatten_keeps_top_level_k                ERROR  ImportError: flatten_model_config
test_flatten_keeps_top_level_dataset          ERROR  同上
test_flatten_expands_metric_templates         ERROR  同上
test_top_level_k_beats_section_k              ERROR  同上
test_lightgcn_entry_yields_k10_yelp2018       FAIL   k=20 / dataset=None
test_metrics_fallback_uses_resolved_k         FAIL   兜底得到 recall@20
```

- [ ] **Step 3: GREEN 后运行**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_metric_k_unification -v`
Expected: 17 条全 PASS

---

## P7 USAGE 文档同步

**Files:**
- Modify: `TPA/models/lightgcn/docs/USAGE.md`、`TPA/models/mf/docs/USAGE.md`、`TPA/models/wmf/docs/USAGE.md`

- [ ] **Step 1: 指标表改为 `{k}` 语义**

```markdown
| evaluation | metrics | `[recall@{k}, ndcg@{k}]` | 指标名中的 `@K` 由顶层 `k` 单次展开；K 的唯一权威是顶层 `k` |
```

- [ ] **Step 2: 补一段"K 唯一来源"说明**

```markdown
K 只由 config.yaml 顶层 `k` 决定（legacy `evaluation.k` 仅兼容读取）。
改 `k` 后同步生效于：评测指标名、history/eval_log 列名、best checkpoint 文件名。
缺 `dataset` 会直接报错（不再回退 gowalla）；缺 `k` 会告警并按 20 继续。
```

- [ ] **Step 3: 提交（P4–P7 一个逻辑变更）**

```bash
git add TPA/attacks/random/fit.py TPA/attacks/bandwagon/fit.py TPA/attacks/pgd/fit.py TPA/attacks/tpa/fit.py TPA/attacks/uba/fit.py TPA/attacks/uba/generate.py TPA/models/wmf/report.py TPA/tests/test_metric_k_unification.py TPA/models/lightgcn/docs/USAGE.md TPA/models/mf/docs/USAGE.md TPA/models/wmf/docs/USAGE.md
git commit -m "fix(attacks): 攻击与报告 K 统一走 resolved 配置，去除 @20 硬编码"
```

---

## P8 ml100k epochs=1 smoke

**Files:** 无源码改动（只跑与核对；产物落 `tmp/`，不入库）

为什么必须先备份：`main()` 结束会把本次 tag 产物复制到 `outputs/` 根的稳定指针
（`checkpoints/latest.pt`、`history.json`、`eval_log.csv`、`latest.json`）。1 epoch 的
smoke 模型若留在稳定指针，会污染攻击流程使用的 `checkpoint.clean`。

- [ ] **Step 1: 备份稳定指针**

```powershell
$smokeBackup = "G:\Idea\tmp\smoke-k10-backup"
New-Item -ItemType Directory -Force -Path $smokeBackup | Out-Null
Copy-Item G:\Idea\TPA\models\lightgcn\outputs\checkpoints\latest.pt $smokeBackup -Force -ErrorAction SilentlyContinue
Copy-Item G:\Idea\TPA\models\lightgcn\outputs\history.json $smokeBackup -Force -ErrorAction SilentlyContinue
Copy-Item G:\Idea\TPA\models\lightgcn\outputs\eval_log.csv $smokeBackup -Force -ErrorAction SilentlyContinue
Copy-Item G:\Idea\TPA\models\lightgcn\outputs\latest.json $smokeBackup -Force -ErrorAction SilentlyContinue
```

- [ ] **Step 2: 写 smoke 配置**（`tmp/smoke-lightgcn-ml100k.yaml`，不改入库配置）

```yaml
# tmp/smoke-lightgcn-ml100k.yaml：仅用于 K 一致性冒烟
dataset: ml100k
k: 10
run_tag: smoke-k10-20260917
evaluation:
  metrics:
  - recall@{k}: upper
  - ndcg@{k}: upper
  checkpoint_mode: per_metric
model:
  emb_dim: 64
  n_layers: 3
  init_method: normal
training:
  batch_size: 256
  checkpoint_dir: models/lightgcn/outputs/checkpoints
  device: cpu
  epochs: 1
  eval_every: 1
  lr: 0.001
  save_every_n_epochs: 1
  neg_ratio: 1
  shuffle: true
  weight_decay: 0.0001
  num_workers: 0
  persistent_workers: false
```

- [ ] **Step 3: 运行 smoke**

Run（工作目录 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe models\lightgcn\main.py --config ..\tmp\smoke-lightgcn-ml100k.yaml --tag smoke-k10-20260917`

- [ ] **Step 4: 核对验收判据**

```powershell
Get-Content -TotalCount 1 G:\Idea\TPA\models\lightgcn\outputs\smoke-k10-20260917\eval_log.csv
Get-ChildItem G:\Idea\TPA\models\lightgcn\outputs\smoke-k10-20260917\checkpoints
Select-String -Path G:\Idea\TPA\models\lightgcn\outputs\smoke-k10-20260917\config.yaml -Pattern "^dataset:|^k:|recall@"
```

Expected：

- `eval_log.csv` 表头 = `epoch,recall@10,ndcg@10`
- checkpoints 目录含 `ndcg@10-best-model.pt`、`recall@10-best-model.pt`、`latest.pt`
- `config.yaml` 快照 = `dataset: ml100k`、`k: 10`、`metrics: [recall@10, ndcg@10]`

- [ ] **Step 5: 还原稳定指针**

```powershell
Copy-Item "$smokeBackup\latest.pt" G:\Idea\TPA\models\lightgcn\outputs\checkpoints -Force
Copy-Item "$smokeBackup\history.json" G:\Idea\TPA\models\lightgcn\outputs -Force
Copy-Item "$smokeBackup\eval_log.csv" G:\Idea\TPA\models\lightgcn\outputs -Force
Copy-Item "$smokeBackup\latest.json" G:\Idea\TPA\models\lightgcn\outputs -Force
```

---

## P9 全量回归

- [ ] **Step 1: 主测试全量**

Run（工作目录 `G:\Idea\TPA`）：
`G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`
Expected: 全绿（记录改动前后的用例数）

- [ ] **Step 2: pre 层测试**

Run: `G:\Idea\.venv\Scripts\python.exe -m unittest discover -s pre/tests -t . -v`
Expected: 全绿

- [ ] **Step 3: 索引卫生自查**

```bash
git status --porcelain
git diff --stat
```

Expected: 只含本计划列出的文件；`outputs/`、`tmp/` 无新增入库项

- [ ] **Step 4: 记录验收结论**（写进最终答复：smoke 三行输出 + 全量测试计数）

---

## 计划自查

**1. Spec 覆盖**

| spec 章节 | 落到任务 |
|---|---|
| §2 I1 唯一权威 | P1（DEFAULT_K + 优先级）+ P2 + P3 + P4 |
| §2 I2 单次展开 | P1 `apply_k` + P5 报告 + P6 契约测试 |
| §2 I3 无静默回退 | P1（warn/raise）+ P4 兜底派生 |
| §2 I4 产物自证 | P6 L2 契约 + P8 smoke 核对 |
| §3.1 在范围内 | P1–P7 |
| §5 测试 L1–L4 | P6 |
| §6 验收判据 1–5 | P6（1/2）+ P8（3）+ P9（4）+ 最终答复（5） |
| §7 风险与回滚 | P1 延迟 import、P3 复跑 test_model_registry、P8 备份还原 |

**2. 占位符扫描**：无 TBD/TODO；每个代码步骤都给出可直接粘贴的实现或明确的替换语句。

**3. 类型/命名一致性**：`DEFAULT_K`、`flatten_model_config`、
`build_training_config_from_yaml`、`_metric_names_from_history`、
`resolve_metrics_cfg` 在 P1–P6 中同名同签名。

**4. 已知边界（不阻塞）**：

- `attacks/uba/generate.py` 的 `hit_k`（UBA 论文自有超参）保持 20 不动；仅全局 K 读取改 `DEFAULT_K`。
- 缺 `evaluation.metrics` 的 5 个旧模型本次不补配置，继续走派生兜底 + 告警。
- `lightgcn/mf` 的 `--config` 参数是对齐 wmf 既有 CLI 的使能性改动，用于 P8 smoke 与后续实验隔离，不改变训练语义。
