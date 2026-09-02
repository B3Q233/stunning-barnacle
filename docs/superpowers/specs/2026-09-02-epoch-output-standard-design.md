# 每 epoch 输出与产物格式统一——设计文档

> 日期：2026-09-02
> 状态：待人工审阅（brainstorming 已确认：攻击侧 + models 全部纳入，参照 bandwagon 格式）

## 1. 背景与目标

仓库内攻击/模型的“每 epoch 输出”与落盘产物格式不一致：bandwagon/random/pgd/tpa
已采用 `【Epoch i/N开始】 → [epoch i/N] train/val loss → [eval] 指标 → [结束 耗时]`
并写标准 `history.json`；但 advinject 生成阶段每轮写入 8 个内部字段且无耗时；
revisit 系简单模型（cml/itemae/itemcf/multvae/ncf）只 `print(metrics)`、无逐 epoch
输出与标准 history。

目标：

1. 所有攻击与模型的每 epoch 控制台输出、history.json、eval_log.csv 采用同一规范；
2. 每个 epoch 必须输出/记录耗时（控制台 `耗时X分Y秒` + history `epoch_seconds`）；
3. 评测时输出全部指标（不截断），无评测的轮次不伪造指标；
4. 规范固化到根 `AGENTS.md` 与 `.codex/skills/paper-code-implementation` 相关模板。

## 2. 统一标准（single source of truth）

### 2.1 控制台每 epoch 输出

```text
【Epoch 1/30开始】
  [epoch 1/30] train_loss=0.4452 val_loss=0.3821
    [eval] recall@10=0.1588, ndcg@10=0.2194, target_hr@10=0.0640, target_ndcg@10=0.0274, target_avg_rank=222.9
    [ckpt] best → …/target_ndcg@10-best-model.pt (target_ndcg@10=0.0274)
[Epoch 1/30结束 耗时0分3.2秒]
```

规则：
- epoch 行前缀为两个空格 `[epoch i/N]`，eval/ckpt 行再缩进两格；
- 指标行输出该轮实际计算的全部指标，键值 `name=value`，逗号+空格分隔；
- epoch 无评测（`epoch % eval_every != 0 and epoch != 1`）时不输出 eval 行；
- 每个 epoch 结束必须 `[Epoch i/N结束 耗时X分Y秒]`（复用 `training/timing.py`）。

### 2.2 history.json 统一 schema

```json
{
  "history": [
    {
      "epoch": 1,
      "epoch_seconds": 3.2,
      "train_loss": 0.4452,
      "val_loss": 0.3821,
      "recall@10": 0.1588,
      "ndcg@10": 0.2194,
      "target_hr@10": 0.0640,
      "target_ndcg@10": 0.0274,
      "target_avg_rank": 222.9,
      "targets": []
    }
  ],
  "best": {
    "target_ndcg@10": {"epoch": 7, "value": 0.0400, "metrics": {}, "checkpoint": "…"}
  }
}
```

规则：
- 顶层固定 `history` / `best`；history 元素必含 `epoch`、`epoch_seconds`；
- 训练循环必含 `train_loss`，有验证集时含 `val_loss`；
- 评测轮把全部指标展开为顶层键（指标名为唯一权威），目标物品明细放在
  `targets`（若存在）；
- 无评测轮不写指标键，不写空 `recall@10` 等占位；
- 文件路径：模型 `outputs/{run_tag}/history.json`、攻击
  `outputs/{dataset}/{model}/{run_tag}/history.json`（沿用现有位置）。

### 2.3 eval_log.csv

- 沿用现有 lightgcn/mf/wmf 的 `eval_log.csv` 模式：表头 `epoch,<指标名…>`，
  只在评测轮追加一行；所有模型与攻击统一产出（攻击路径同 history.json）。

### 2.4 攻击生成阶段（advinject generate）特例

生成优化阶段的“每轮”不是训练轮：history 元素只保留
`{epoch, epoch_seconds, loss}`，删除 `model_name/epochs/pretrain_steps/
unroll_steps/gradient_l2/changed` 冗余字段；控制台按标准 epoch 行打印
（`train_loss=` 处显示该轮 `loss`，不伪造 val_loss/eval）。最终攻击效果仍由
对比报告（bandwagon 样式 md/json）承载。

### 2.5 指标可扩展性：注册表驱动 + 结果矩阵

- 所有评测指标本质是对同一张 **User×Item 预测分数矩阵**（ranking scores）的
  后处理：统一评测入口以该矩阵 + 用户/测试正样本为输入，返回
  `dict[指标名, 数值]`。
- 新增 `evaluation/metrics.py`（或同目录 `metrics_registry.py`）的
  **指标注册表**：`METRICS: dict[str, Callable]`，键 = 指标名
  （如 `recall@10`、`target_ndcg@10`、`expected_percentile_rank`）。
  新增指标 = 注册实现 + 在配置 `evaluation.metrics` 中加入指标名，
  **不需要改动 epoch 打印/history/csv 代码**。
- 每 epoch 的打印、history.json、eval_log.csv 全部以评测返回的 dict 键为准
  （动态键），禁止在日志层硬编码指标名单；新增指标注册后自动出现在
  `[eval]` 行、history 元素与 csv 表头中。
- 可选分数矩阵缓存：配置 `evaluation.save_scores: false`（默认关闭）；
  开启时每评测轮把分数矩阵存到
  `outputs/{run_tag}/eval_scores/epoch_{i}.npy`，便于离线补算/新增指标，
  避免重复全量排序。默认关闭防止大数据集磁盘膨胀。
- 回归测试：注册一个临时假指标（如 `fake_metric@5`），断言其自动出现在
  `[eval]` 行、history 与 eval_log.csv，卸载后消失。

## 3. 共享模块设计

新增 `TPA/training/epoch_log.py`：

```python
def log_train_line(epoch, total, train_loss, val_loss=None) -> None
def log_eval_line(metrics: dict) -> None
def write_history(out_dir: Path, history: list, best: dict) -> None
def write_eval_log(out_dir: Path, rows: list, metric_names: list) -> None
def epoch_section(epoch, total) -> "context manager（复用 SectionTimer）"
```

- 控制台文案与 §2.1 一致；
- `write_history` 输出 §2.2 schema（不修改入参）；
- `write_eval_log` 输出 §2.3 csv；
- 内部复用 `training/timing.py` 的 `section_enter/section_exit/format_duration`。
- 各调用方把评测返回的完整 dict 直接交给 log_eval_line / history / csv，
  不自行筛选指标名（指标集合由注册表 + 配置决定）。

## 4. 应用改动清单

### 4.1 共享组件与攻击

- 新增 `TPA/training/epoch_log.py` + 单测 `TPA/tests/test_epoch_log.py`。
- 新增指标注册表（evaluation/metrics.py 或新文件）与
  `tests/test_metric_registry.py`，保证“加指标自动同步输出”。
- `bandwagon/random/pgd/tpa`：`fit.py` 每 epoch 循环与 history 写入改为调用共享
  组件；`history` 元素补 `epoch_seconds`。
- `advinject`：
  - `generate.py`：历史精简 + 标准 epoch 行 + `epoch_seconds`；`__main__`
    改 `load_config`（此前遗漏）；
  - `evaluate.py`/`fit.py`：victim 重训循环（`retrain_victim`）接入标准
    epoch 输出与 history；最终对比报告保留（另见 §5）。

### 4.2 models

- `lightgcn/mf/wmf`：训练循环/回调改用共享组件，补 `epoch_seconds`；`history`
  元素字段与 §2.2 对齐（lightgcn/mf 现有 eval 回调已近似，收敛写法）。
- `cml/itemae/itemcf/multvae/ncf`：
  - 入口实现逐 epoch 训练行 + 耗时；
  - 周期性全量评估（eval_every，默认每轮）并写 `eval_log.csv`；
  - `save_training_artifacts` 输出统一 history.json schema；
  - itemae/multvae 若模型内部 `fit` 无法逐 epoch 返回，先按其返回值记录
    最小 `{epoch, loss}` 并打印，不修改模型内部数学。
- `revisit_training.py`：`save_training_artifacts` / `ranking_report` 对齐
  §2.2/§2.3 并支持传入 `epoch_seconds`。

## 5. 对比报告（保留 bandwagon 样式）

对比报告 md/json 不并入本规范重构范围，但保持一致要求：标题标注 K；含模型
效用（Clean/Poisoned/Δ）、目标物品明细（HR/NDCG/平均排名）、结论。

## 6. 测试

- 新增 `tests/test_epoch_log.py`：行格式、history schema、csv 表头、无评测轮
  不写指标、epoch_seconds 非负。
- 更新受影响测试（`test_history_completeness`、wmf/lightgcn/mf 相关、攻击 fit
  测试）以新字段为准。
- 全量回归：`python -m unittest discover -s tests -p "test_*.py" -v`。

## 7. AGENTS.md 与 skill 固化

AGENTS.md §3 代码与工程规范新增“输出与产物规范（必须）”条款，内容为 §2.1–§2.3
与“新建模型/攻击的每 epoch 输出必须参照 `TPA/docs/epoch-output-standard.md`
（或本规范）”。条款同时要求：新增评测指标必须注册到指标注册表并以配置
`evaluation.metrics` 启用；epoch 打印/history/csv 不得硬编码指标名单，
须随评测返回的指标 dict 自动同步。

`.codex/skills/paper-code-implementation` 同步：
- `SKILL.md`：训练实现章节追加“每 epoch 输出规范”；
- `references/usage_doc_template.md`：产物说明增加 history.json/eval_log.csv
  与耗时说明；
- `assets/` 模板 fit/train 若含 epoch 打印则按本规范收敛。

## 8. 明确不做（Out of scope）

- 不改动任何训练/优化算法的数学与默认超参；
- 不重构各模块目录结构、run_tag/输出路径层级；
- 不统一对比报告 md 之外的画图格式；
- 不改 batch aggregate 的汇总表（其字段已含指标）。

## 9. 风险

- itemae/multvae 内部 fit 若不可逐 epoch 观测，需在 wrapper 层按模型返回值
  降级为最小历史并明确注释；不得为了输出格式改动模型内部实现。
- lightgcn/mf 现有 eval_log.csv 与 history.json 写入位置需要保留兼容，
  仅统一字段/写法。
