# 多指标最优 Checkpoint 保存设计

> 日期：2026-08-09
> 状态：已获用户确认（含命名规则调整）

## 目标

修改模型参数更新/保存逻辑：按传入的 `evaluation.metrics` 独立跟踪每个指标的最优，
按指标方向（upper 越高越好 / lower 越低越好）保存多份最优 checkpoint，并在
`history.json` 中记录全部最优结果。同时支持 YAML 中对指标方向的上/下标注解析。

## 需求要点（来自用户）

1. 提供保存模式选择；默认按传入 metrics 保留各自最优。
2. 指标方向可配置：`upper`（越高越好）/ `lower`（越低越好）。
3. 多个指标时按"每个指标一份最优"保存（N 指标 → N 份），不做两两组合；
   即使两个指标的最优 epoch 相同也各保存一份（不去重）。
4. `history.json` 存入最优所有结果（每个指标最优时该 epoch 的全部指标快照）。
5. YAML 解析：`- recall@20` 后添加 `upper` / `lower` 进行标注。
6. 最优模型命名：`{指标}-best-model`（如 `recall@20-best-model.pt`），
   便于后续导入流程。
7. 该逻辑对所有实验统一采用（受害模型训练 + 四个攻击 + skill 模板）。

## 设计

### 1. 指标方向解析（`training/metrics.py`）

`parse_metrics(metrics_cfg) -> Dict[str, str]` 支持三种写法：

```yaml
evaluation:
  metrics:
  - recall@20: upper        # YAML 字典，显式标注
  - "ndcg@20 lower"         # 字符串后跟方向
  - hit@10                  # 裸字符串，按默认方向
  checkpoint_mode: per_metric   # per_metric（默认）| single
```

默认方向规则（`default_direction(name)`）：

- 内置表（按指标名前缀匹配）：`recall / ndcg / precision / hit / hr / map /
  auc / acc / f1 → upper`；`loss / rmse / mae / mse / err → lower`。
- 未命中且未标注：默认 `upper`。
- 显式标注（`upper`/`lower`）优先于内置表。

### 2. 最优跟踪器（`training/metrics.py` 的 `BestTracker`）

- `update(metrics: Dict[str, float], epoch: int) -> List[str]`：按方向比较，
  返回本 epoch 新刷新的指标名列表。
- `best_checkpoints() -> List[Tuple[metric, filename]]`：
  - `per_metric`：每个指标一份，文件名为 `{指标}-best-model.pt`；
  - `single`：仅第一个指标一份（同命名）。
- `best_results() -> Dict[str, Dict]`：每个指标最优的
  `{epoch, value, metrics(全量快照), checkpoint}`。
- 文件名对路径非法字符做安全化（`@` 保留，`\ / : * ? " < > |` 与空白替换为 `_`）。

### 3. 保存与历史

- 训练中某 epoch 刷新某指标最优时，立即保存该 epoch 的模型为
  `{指标}-best-model.pt`；两个指标同 epoch 刷新则各存一份（不去重）。
- `latest.pt`（最终 epoch）行为不变。
- `history.json` 结构：

```json
{
  "best": {
    "recall@20": {"epoch": 71, "value": 0.214,
                  "metrics": {"recall@20": 0.214, "ndcg@20": 0.198},
                  "checkpoint": "recall@20-best-model.pt"},
    "ndcg@20":  {"epoch": 58, "value": 0.198,
                  "metrics": {"recall@20": 0.212, "ndcg@20": 0.198},
                  "checkpoint": "ndcg@20-best-model.pt"}
  },
  "history": [{"epoch": 1, ...}, ...]
}
```

- 兼容：`--skip-train` 加载顺序改为 `{首指标}-best-model.pt` → `best.pt`
  （旧实验）→ `latest.pt`。

### 4. 应用范围

- `models/mf/train.py`、`models/lightgcn/train.py`：FullRankingCallback 接入
  BestTracker，best 文件写入 `outputs/{tag}/checkpoints/`。
- 四个攻击的 `fit.py`（pgd / bandwagon / tpa / random）：
  `train_poisoned_model` 接入 BestTracker，best 文件写入
  `outputs/{dataset}/{model}/{tag}/checkpoints/`；指标配置优先取攻击 config
  `evaluation.metrics`，缺省取模型自身 config。
- skill 模板：`assets/attack-imp-direct-poison/fit.py`、SKILL.md、
  `references/config_template.yaml`、`references/attack_imp_direct_poison.md`
  同步更新。
- 模型 config 与攻击 config 的 `evaluation` 增加
  `metrics`（带方向标注）与 `checkpoint_mode`。

## 明确不做（YAGNI）

- 不做指标两两/多组合联合最优（A∧B 同时最优）的存档。
- 不改变 `latest.pt` 语义，不改变 run_tag 实验隔离机制。
- 不引入新第三方依赖（单测用 stdlib unittest）。

## 测试策略

- `TPA/tests/test_training_metrics.py`（unittest）：
  - `parse_metrics`：三种写法、默认表、显式覆盖、非法方向报错；
  - `BestTracker`：upper/lower 方向、per_metric/single 模式、同 epoch 双指标
    各存一份、历史 best 结构、文件命名。
- 端到端冒烟：攻击 fit 用 1-2 epoch 跑通，检查
  `checkpoints/*-best-model.pt` 与 `history.json` 的 `best` 段。
