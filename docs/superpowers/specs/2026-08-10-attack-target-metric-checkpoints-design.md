# 攻击实验按目标物品指标选优 Checkpoint —— 设计文档

> 日期：2026-08-10
> 状态：待用户审阅

## 1. 背景与问题

TPA 项目包含四个攻击模块（`tpa` / `pgd` / `bandwagon` / `random`）与
`paper-code-implementation` 技能的 attack-imp-direct-poison 模板。四个攻击的
`fit.py` 在训练中毒模型时，用 `BestTracker` 按 `evaluation.metrics` 配置跟踪
并保存最优 checkpoint；但当前默认配置只包含**整体测试集**的 `recall@K` /
`ndcg@K`（模型效用指标），目标物品的攻击效果指标（`hr@k` / `ndcg@k`）只在
训练结束后的 `compare_models` 对比报告里计算，不参与选优。

后果：`--skip-train` 与最终对比报告加载的是"整体指标最优"的中毒模型，而不是
"对被攻击目标物品攻击效果最优"的模型，与攻击实验的评估目标不一致。

## 2. 目标与成功标准

目标：攻击实验的 checkpoint 选优以**被攻击目标物品**的指标为准。

成功标准：

1. 四个攻击配置的 `evaluation.metrics` 中 `target_ndcg@10` 排第一
   （primary metric），`--skip-train` 加载 `target_ndcg@10-best-model.pt`
   （存在时）。
2. 训练产出 `target_ndcg@10-best-model.pt` 与 `target_hr@10-best-model.pt`
   （per_metric 模式），整体 `recall@10` / `ndcg@10` checkpoint 仍保存，
   仅作投毒代价参考。
3. `history.json` 的每个评估 epoch 条目包含目标物品明细
   （`hr@k` / `ndcg@k` / `hit_users` / `mean_rank` / `mean_rank_all` / `n_elig`）。
4. 新增共享攻击评估层，四个攻击与模板的评估逻辑不再各存一份业务实现；
   以后新增指标只改共享模块与配置。
5. 单测全绿；无 `target_*` 指标的配置（向后兼容）行为与旧版一致。

## 3. 方案概览

采用方案 A（用户已确认）+ 共享评估层重构（用户已确认）：

- 新建共享模块 `TPA/evaluation/attack_eval.py`，收敛四个攻击 `evaluate.py`
  的全部业务实现，并以参数控制攻击名相关的标题/输出文件名。
- 每个攻击的 `evaluate.py` 改为纯 re-export 薄壳。
- 攻击 `fit.py` 训练循环在评估 epoch 调用共享的 `build_attack_eval_metrics`，
  把 `target_hr@K` / `target_ndcg@K` 并入 BestTracker 选优，并把目标物品明细
  写入 history。
- 四个攻击 config + 模板 config 的 `evaluation.metrics` 改为 target 指标在前。

## 4. 架构与组件

### 4.1 共享模块 `TPA/evaluation/attack_eval.py`（新建）

从四个攻击的 `evaluate.py` 收敛以下函数（以 `tpa` 副本为基准，保留其完整
docstring；`pgd` 版本中删除的 docstring 全部恢复）：

- `ranking_scores(model, test_pairs)`：全量排序评分，返回
  `(scores, test_user_ids, test_pos)`。
- `compute_target_metrics(scores, user_ids, clean_user_items, target_items, k)`：
  目标物品攻击指标，返回 `{target: {hr@k, ndcg@k, hit_users, mean_rank,
  mean_rank_all, n_elig, exposure, ndcg}}`。
  - 相比旧版**新增 `n_elig` 字段**（合格用户数），供聚合跳过无合格用户的目标；
  - `exposure` / `ndcg` 旧别名保留，兼容已有 JSON 消费者。
- `aggregate_target_metrics(target_metrics, target_items, k)`（新）：
  返回 `{"target_hr@k": ..., "target_ndcg@k": ...}`。
  - 等权均值：对每个目标物品的 `hr@k` / `ndcg@k` 取平均；
  - 跳过 `n_elig == 0` 的目标；所有目标均无合格用户时返回 `0.0`。
- `build_attack_eval_metrics(scores, user_ids, user_items, test_pos,
  clean_user_items, targets, ks, metric_names)`（新）：
  - 按 `ks` 计算整体指标 `compute_metrics(...)`；
  - 若 `metric_names` 中存在 `target_` 前缀指标，则对每个 K 计算目标指标并
    聚合进 `res_by_k[K]`；
  - 返回 `(res, target_details)`：`res` 为与 `metric_names` 对齐的扁平指标字典，
    `target_details` 为最大 K 下的 `{target: {...}}` 明细（供 history）。
- `compare_models(clean_model, poisoned_model, clean_meta, poisoned_meta,
  target_items, k, report_utility=True)`：与旧版一致。
- `format_report(report, title="投毒攻击对比报告")`：与旧版一致，但：
  - 标题由参数控制（tpa/random 用默认，pgd/bandwagon 传各自标题）；
  - **结论段泛化**：四个攻击的报告统一附带结论段（旧版仅 pgd 有），去掉
    "PGD" 专属措辞，改为"投毒显著提升了目标物品曝光"通用文案；
  - 结论段对 `recall@` 键做存在性判断，避免缺 recall 时 KeyError。
- `save_report(report, out_dir, name="attack")`：输出
  `{name}_comparison.md` 与 `{name}_comparison.json`，保持各攻击现有文件名：
  tpa/random → `attack`，pgd → `pgd`，bandwagon → `bandwagon`。

依赖方向：`evaluation/attack_eval.py` 只 import `evaluation.metrics` 与
`numpy` / `torch`；`evaluation/metrics.py` 不反向依赖 `attack_eval`，无循环。

### 4.2 各攻击 `evaluate.py` 薄壳化

`TPA/attacks/{tpa,pgd,bandwagon,random}/evaluate.py` 与技能模板
`assets/attack-imp-direct-poison/evaluate.py` 全部替换为：

```python
"""{攻击名} 攻击效果评估 —— 共享实现见 evaluation/attack_eval.py"""
from evaluation.attack_eval import (  # noqa: F401
    build_attack_eval_metrics,
    compare_models,
    compute_target_metrics,
    format_report,
    ranking_scores,
    save_report,
)
```

现有消费方无需改动导入：各 `fit.py` 的
`from attacks.{name}.evaluate import compare_models, ranking_scores, save_report`
与 `tpa/train_surrogate.py` 的 `ranking_scores` 继续可用。

### 4.3 攻击 `fit.py` 集成

四个攻击 `fit.py` 与模板 `fit.py` 做相同修改：

1. import 增加 `build_attack_eval_metrics`（从其自身 `evaluate` 薄壳导入）。
2. `train_poisoned_model` 新增参数 `targets: List[int]` 与
   `clean_user_items: Dict[int, set]`。
3. 评估 epoch 中，把

   ```python
   res_by_k = {K: compute_metrics(...) for K in ks}
   res = match_metric_values(list(tracker.directions), res_by_k)
   ```

   替换为：

   ```python
   res, target_details = build_attack_eval_metrics(
       scores, users, user_items, test_pos_local,
       clean_user_items, targets, ks, list(tracker.directions),
   )
   entry.update(res)
   if target_details:
       entry["targets"] = target_details
   ```

   `user_items` 仍为中毒数据 `poisoned_meta["user_items"]`（整体指标口径不变）；
   `clean_user_items` 仅用于目标人群过滤（与 `compare_models` 口径一致）。
4. `main()` 传入 `targets=targets`（来自 stats.json）与
   `clean_user_items=clean_meta["user_items"]`。
5. `save_report(report, out_dir)` 改为 `save_report(report, out_dir,
   name=attack_name)`，保持输出文件名不变。

`--skip-train` 加载链不变：`{primary}-best-model.pt` → `best.pt` →
`latest.pt`；primary 随配置变为 `target_ndcg@10`。旧 run 目录无 target
checkpoint 时自动回退，无需迁移。

### 4.4 配置改动

四个攻击 `config.yaml` 与模板 `config.yaml`、`references/config_template.yaml`
的 `evaluation.metrics` 改为（K 取各攻击当前评估 K，tpa 为 10；pgd/bandwagon/
random 沿用各自当前 K）：

```yaml
evaluation:
  metrics:
    - target_ndcg@10: upper
    - target_hr@10: upper
    - recall@10: upper
    - ndcg@10: upper
  checkpoint_mode: per_metric
```

`target_*` 排最前 ⇒ `BestTracker.primary_metric == "target_ndcg@10"`。

## 5. 数据流

```
clean meta.pkl ──(main)──▶ clean_user_items ──┐
stats.json ──(main)──▶ targets ───────────────┤
                                              ▼
                            train_poisoned_model（fit.py）
                              └─ 评估 epoch:
                                   scores = ranking_scores(...)
                                   res, details = build_attack_eval_metrics(
                                       scores, ..., clean_user_items, targets, ks,
                                       metric_names)
                                   tracker.update(res, epoch) → 存
                                     target_ndcg@10-best-model.pt /
                                     target_hr@10-best-model.pt / recall@10-... /
                                     ndcg@10-...
                                   entry["targets"] = details → history.json
                               训练结束
compare_models(clean, poisoned=target_ndcg@10 最优) → {name}_comparison.md/json
```

## 6. 接口定义

```python
# evaluation/attack_eval.py
def ranking_scores(model, test_pairs) -> tuple[torch.Tensor, list[int], dict[int, set]]: ...

def compute_target_metrics(scores, user_ids, clean_user_items,
                           target_items, k) -> dict[int, dict[str, Any]]: ...

def aggregate_target_metrics(target_metrics, target_items, k) -> dict[str, float]:
    """→ {"target_hr@k": float, "target_ndcg@k": float}"""

def build_attack_eval_metrics(scores, user_ids, user_items, test_pos,
                              clean_user_items, targets, ks, metric_names
                              ) -> tuple[dict[str, float], dict[int, dict[str, Any]]]: ...

def compare_models(clean_model, poisoned_model, clean_meta, poisoned_meta,
                   target_items, k, report_utility=True) -> dict[str, Any]: ...

def format_report(report, title="投毒攻击对比报告") -> str: ...

def save_report(report, out_dir, name="attack") -> Path: ...
```

`training/metrics.py` 的 `BestTracker` / `match_metric_values` /
`eval_ks_from_metrics` / `safe_checkpoint_name` **不改动**，已支持任意指标名。

## 7. 边界与错误处理

- `targets` 为空：`compute_target_metrics` 返回 `{}`，聚合返回
  `{"target_hr@k": 0.0, "target_ndcg@k": 0.0}`；训练正常进行。
- 某目标 `n_elig == 0`：该目标不参与均值（避免把 0 拉低均值），明细仍写入
  history。
- metrics 配置无 `target_*`：`build_attack_eval_metrics` 只计算整体指标，
  与旧版行为一致（向后兼容）。
- 旧 run 目录无 target checkpoint：`--skip-train` 回退 `best.pt` / `latest.pt`。
- 结论段在 `model_utility` 缺失或无 `recall@` 键时跳过效用结论，不抛异常。

## 8. 测试计划（TDD）

新增 `TPA/tests/test_attack_eval.py`（unittest，无新依赖），直接测共享函数：

- `compute_target_metrics`：合成 scores 矩阵；命中 rank=1 与 rank=k；合格用户
  过滤（训练集已交互目标的用户被排除）；无合格用户（`n_elig==0`、`mean_rank`
  为 None、hr/ndcg 为 0）；`mean_rank_all`；`exposure`/`ndcg` 别名。
- `aggregate_target_metrics`：单目标；多目标等权均值；跳过 `n_elig==0` 目标；
  全部无合格用户返回 0.0。
- `build_attack_eval_metrics`：含 `target_*` 指标时 res 含整体 + target 键、
  `target_details` 正确；不含 `target_*` 时无 target 键；`targets` 为空仍正常。
- `format_report`：标题参数生效；含结论段；缺 recall 键不抛异常。
- `save_report`：写入 `{name}_comparison.md` 与 `.json`（用临时目录）。

回归：现有 `test_training_metrics.py` / `test_modes.py` 全绿；
对四个攻击与模板的 fit.py / evaluate.py 做 `python -m py_compile` 语法检查。

## 9. 技能模板与文档同步

- 技能 `assets/attack-imp-direct-poison/`：`evaluate.py` 薄壳化、`fit.py` 同改、
  `config.yaml` metrics 重排。
- 技能 `SKILL.md`："多指标最优 checkpoint"约定段落补充——
  "攻击实验的选优主指标为 `target_ndcg@K`（被攻击目标物品的 NDCG），
  整体 recall/ndcg 仅作投毒代价参考"。
- 技能 `references/config_template.yaml`：metrics 示例加 `target_*`。
- 技能 `references/attack_imp_direct_poison.md`：交付清单与约定同步。
- 项目 `TPA/attacks/{tpa,pgd,bandwagon,random}/docs/USAGE.md`：evaluation 段
  说明 target 指标与主选优口径；`TPA/attacks/tpa/docs/DESIGN.md` 评估协议段
  补充一句。

## 10. git 提交策略

按任务分提交（沿用现有 `feat`/`docs` 前缀风格，参考最近一次
metric-best-checkpoints 计划的提交粒度）：

1. `docs: 攻击按目标物品指标选优设计 spec` —— 本文件。
2. `feat(eval): 共享攻击评估层 attack_eval.py 与薄壳化 + 单测` ——
   `evaluation/attack_eval.py`、四个攻击 `evaluate.py`、模板 `evaluate.py`、
   `tests/test_attack_eval.py`。
3. `feat(attacks): fit.py 接入目标物品选优（4 攻击 + config）` ——
   四个攻击 `fit.py` 与 `config.yaml`。
4. `feat(skill): paper-code-implementation 模板同步目标物品选优` ——
   模板 `fit.py` / `config.yaml` / `SKILL.md` / `references/*`。
5. `docs: 更新攻击 USAGE/DESIGN 说明` —— 攻击文档。

## 11. 范围外（Out of scope）

- 不重构四个 `fit.py` 训练循环本身的共享化（指标逻辑已共享，新增指标无需改
  fit.py；训练循环骨架属模板定制区域）。
- 不改 `models/*` 干净模型训练的选优逻辑（无目标物品概念）。
- 不改变报告 JSON schema 的既有字段（只新增 `n_elig` 等键）。
- 不做论文功能（PGD 对抗优化、路径消融等）。
