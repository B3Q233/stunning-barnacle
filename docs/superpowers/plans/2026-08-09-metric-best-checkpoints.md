# 多指标最优 Checkpoint 保存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让所有实验（受害模型训练 + 攻击投毒训练 + skill 模板）按 `evaluation.metrics`
的方向标注（upper/lower）独立保存每个指标的最优 checkpoint（`{指标}-best-model.pt`），
并在 `history.json` 记录全部最优结果。

**Architecture:** 新增无依赖的 `training/metrics.py`（指标方向解析 + BestTracker 跟踪器），
模型训练回调与攻击 fit 的训练循环都只调用 BestTracker；配置层扩展
`evaluation.metrics`（支持方向标注）与 `evaluation.checkpoint_mode`（per_metric/single）。

**Tech Stack:** Python 3.12 + PyTorch 2.5，测试用 stdlib unittest（不新增依赖）。

## Global Constraints

- 最优 checkpoint 命名：`{指标}-best-model.pt`（如 `recall@20-best-model.pt`），
  文件名对 `\ / : * ? " < > |` 与空白做安全化，`@` 保留。
- 默认保存模式：`evaluation.checkpoint_mode: per_metric`（N 指标 → N 份），
  `single` 仅保存第一指标一份；同 epoch 多指标刷新时各存一份，不去重。
- 默认方向：内置表（recall/ndcg/precision/hit/hr/map/auc/acc/f1 → upper；
  loss/rmse/mae/mse/err → lower），未标注未命中默认 upper；显式标注优先。
- `history.json` 结构：`{"best": {指标: {epoch, value, metrics, checkpoint}}, "history": [...]}`。
- `--skip-train` 加载顺序：`{首指标}-best-model.pt` → `best.pt`（旧）→ `latest.pt`。
- 不引入新第三方依赖；不改 run_tag 隔离；`latest.pt` 语义不变。
- 所有改动同步到 skill：`assets/attack-imp-direct-poison/fit.py`、
  SKILL.md、`references/config_template.yaml`、`references/attack_imp_direct_poison.md`。

---

### Task 1: metrics 解析与 BestTracker（TDD 核心）

**Files:**
- Create: `TPA/training/metrics.py`
- Create: `TPA/tests/test_training_metrics.py`
- Test: `G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_training_metrics -v`

**Interfaces:**
- `parse_metrics(metrics_cfg) -> Dict[str, str]`（指标名 → upper/lower）
- `default_direction(name) -> str`
- `class BestTracker`:
  - `__init__(metrics_cfg, checkpoint_mode="per_metric")`
  - `primary_metric -> str | None`
  - `update(metrics: Dict[str, float], epoch: int) -> List[str]`（返回刷新指标名）
  - `best_checkpoints() -> List[Tuple[str, str]]`（(指标, 文件名)）
  - `best_results() -> Dict[str, Dict]`
- `safe_checkpoint_name(metric) -> str`

- [ ] **Step 1: 写失败测试**（覆盖三种写法、默认表、显式覆盖、方向比较、
      per_metric/single、同 epoch 双指标各存一份、命名、best_results 结构）

```python
import unittest
from training.metrics import BestTracker, parse_metrics, safe_checkpoint_name

class ParseMetricsTest(unittest.TestCase):
    def test_dict_and_string_annotations(self):
        cfg = [{"recall@20": "upper"}, "ndcg@20 lower", "hit@10"]
        d = parse_metrics(cfg)
        self.assertEqual(d["recall@20"], "upper")
        self.assertEqual(d["ndcg@20"], "lower")
        self.assertEqual(d["hit@10"], "upper")
    def test_default_table_and_fallback(self):
        self.assertEqual(default_direction("loss"), "lower")
        self.assertEqual(default_direction("ndcg@20"), "upper")
        self.assertEqual(default_direction("custom@5"), "upper")
    def test_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            parse_metrics(["recall@20 middle"])

class BestTrackerTest(unittest.TestCase):
    def test_upper_and_lower_best(self):
        t = BestTracker([{"recall@20": "upper"}, "loss lower"])
        self.assertEqual(t.update({"recall@20": 0.1, "loss": 1.0}, 1), ["recall@20", "loss"])
        self.assertEqual(t.update({"recall@20": 0.2, "loss": 0.8}, 2), ["recall@20", "loss"])
        self.assertEqual(t.update({"recall@20": 0.15, "loss": 0.9}, 3), [])
        r = t.best_results()
        self.assertEqual(r["recall@20"]["epoch"], 2)
        self.assertEqual(r["loss"]["epoch"], 2)
    def test_per_metric_keeps_one_file_per_metric(self):
        t = BestTracker([{"recall@20": "upper"}, "ndcg@20 upper"])
        t.update({"recall@20": 0.1, "ndcg@20": 0.2}, 1)
        t.update({"recall@20": 0.2, "ndcg@20": 0.3}, 1)   # 同 epoch 双指标刷新
        files = dict(t.best_checkpoints())
        self.assertEqual(files["recall@20"], "recall@20-best-model.pt")
        self.assertEqual(files["ndcg@20"], "ndcg@20-best-model.pt")
        self.assertEqual(len(files), 2)
    def test_single_mode_only_primary(self):
        t = BestTracker([{"recall@20": "upper"}, "ndcg@20 upper"], checkpoint_mode="single")
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
```

- [ ] **Step 2: 运行测试确认失败**（预期 FAIL：`training.metrics` 不存在）
- [ ] **Step 3: 最小实现 `training/metrics.py`**（parse_metrics / default_direction /
      safe_checkpoint_name / BestTracker，按上文接口）
- [ ] **Step 4: 运行测试确认通过**
- [ ] **Step 5: Commit**（`git add TPA/training/metrics.py TPA/tests/test_training_metrics.py`）

### Task 2: 受害模型训练接入（models/mf、models/lightgcn）

**Files:**
- Modify: `TPA/models/mf/config.yaml`、`TPA/models/lightgcn/config.yaml`
  （`evaluation.metrics` 加方向标注，新增 `checkpoint_mode: per_metric`）
- Modify: `TPA/models/mf/train.py`、`TPA/models/lightgcn/train.py`

**Interfaces:** 复用 Task 1 的 BestTracker。

- [ ] **Step 1: 更新两个模型 config.yaml**

```yaml
evaluation:
  metrics:
  - recall@20: upper
  - ndcg@20: upper
  checkpoint_mode: per_metric
```

- [ ] **Step 2: FullRankingCallback 接入 BestTracker**
  - `__init__` 中 `self.tracker = BestTracker(config.get("metrics"), config.get("checkpoint_mode", "per_metric"))`
  - `on_epoch_end` 计算 `result` 后：
    `improved = self.tracker.update(result, epoch)`；对每个 `name in improved` 保存
    `os.path.join(self.tag_checkpoint_dir, f"{name}-best-model.pt")`（内容含
    epoch/model_state_dict/recall/ndcg 与全量 result）。
- [ ] **Step 3: history.json 写入 best 段**：main() 中每次写
  `{"history": history, "best": full_rank.tracker.best_results()}`（含最终指针复制）。
- [ ] **Step 4: 语法/导入检查 + 单测回归**
- [ ] **Step 5: Commit**

### Task 3: 四个攻击 fit.py 接入

**Files:**
- Modify: `TPA/attacks/{pgd,bandwagon,tpa,random}/config.yaml`（evaluation 增加 metrics + checkpoint_mode）
- Modify: `TPA/attacks/{pgd,bandwagon,tpa,random}/fit.py`

**Interfaces:**
- 新增 `resolve_metrics_cfg(config, model_name) -> list`：
  优先 `config["evaluation"]["metrics"]`，缺省 `load_model_config(model_name)["evaluation"]["metrics"]`
- 新增 `primary_metric(metrics_cfg) -> str | None`（Task 1 的 BestTracker(metrics_cfg).primary_metric）

- [ ] **Step 1: 每个攻击 fit.py 的 `train_poisoned_model` 接入 BestTracker**
  （同上 Task 2 模式；`improved` 时保存到 `ckpt_dir / f"{name}-best-model.pt"`；
  history 写 `{"history": ..., "best": tracker.best_results()}`）
- [ ] **Step 2: `--skip-train` 加载链改为
  `{primary}-best-model.pt` → `best.pt` → `latest.pt`**
- [ ] **Step 3: 更新 4 个攻击 config.yaml 的 evaluation 段**
- [ ] **Step 4: 语法检查 + 端到端冒烟**（pgd 用 1-2 epoch 临时 tag，
      检查 `*-best-model.pt` 文件与 history.json 的 best 段）
- [ ] **Step 5: Commit**

### Task 4: skill 模板同步

**Files:**
- Modify: `.codex/skills/paper-code-implementation/assets/attack-imp-direct-poison/fit.py`
- Modify: `.codex/skills/paper-code-implementation/assets/attack-imp-direct-poison/config.yaml`
- Modify: `.codex/skills/paper-code-implementation/SKILL.md`
- Modify: `.codex/skills/paper-code-implementation/references/config_template.yaml`
- Modify: `.codex/skills/paper-code-implementation/references/attack_imp_direct_poison.md`

- [ ] **Step 1: 模板 fit.py 应用 Task 3 的同样改动**
- [ ] **Step 2: 模板 config.yaml evaluation 段增加 metrics + checkpoint_mode**
- [ ] **Step 3: SKILL.md 增加"多指标最优 checkpoint"约定段落**
- [ ] **Step 4: references 更新（config_template.yaml 的 metrics 示例、
      attack_imp_direct_poison.md 交付清单）**
- [ ] **Step 5: 语法检查 + Commit**

### Task 5: 总体验证

- [ ] **Step 1: 运行全部单测**（unittest，预期全绿）
- [ ] **Step 2: 全项目 py 语法检查**
- [ ] **Step 3: 冒烟复核**：pgd 临时 tag 跑 data+model(1-2 epoch)，
      核对 `{指标}-best-model.pt`、`history.json.best`、`--skip-train` 可加载
- [ ] **Step 4: 清理冒烟产物**
