# 每 epoch 输出与产物格式统一——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立指标注册表驱动的每 epoch 输出规范（console/history.json/eval_log.csv，含耗时与全部指标），应用到全部攻击与模型，并固化进 AGENTS.md 与 paper-code-implementation skill。

**Architecture:** 新增 `training/epoch_log.py`（标准行/耗时/write_history/write_eval_log）与指标注册表（evaluation 域）；所有训练/生成循环以评测返回的指标 dict 驱动输出；advinject 精简生成历史；revisit 系模型入口补逐 epoch 输出。

**Tech Stack:** Python 3.12（仓库 .venv）、unittest、yaml。

## Global Constraints

- 指标打印/history/csv 不得硬编码指标名单；以评测返回 dict 键为准。
- 每个 epoch：console 有 `[Epoch i/N结束 耗时X分Y秒]`，history 有 `epoch_seconds`。
- 无评测轮不写指标键；不伪造指标。
- 新增指标 = 注册表登记 + 配置 `evaluation.metrics` 加入名字，日志自动同步。
- 不改算法数学与默认超参；不改输出路径层级。
- 全量测试（PowerShell，TPA 根）：
  `python -m unittest discover -s tests -p "test_*.py" -v`。
- 每 Task 结束跑相关测试并提交。

---

### Task 1：共享 epoch 输出组件 + 指标注册表（TDD）

**Files:**
- Create: `TPA/training/epoch_log.py`、`TPA/evaluation/metrics_registry.py`
- Create: `TPA/tests/test_epoch_log.py`、`TPA/tests/test_metric_registry.py`

**Interfaces:**
- Produces:
  - `epoch_log.log_train_line(epoch,total,train_loss,val_loss=None)`
  - `epoch_log.log_eval_line(metrics: dict)`
  - `epoch_log.write_history(out_dir, history, best)`
  - `epoch_log.write_eval_log(out_dir, rows, metric_names)`
  - `epoch_log.epoch_section(epoch,total)`（复用 timing.SectionTimer）
  - `metrics_registry.METRICS: dict[str, Callable]`、`register_metric(name, fn)`、`compute_metrics_by_name(name, scores, ...)`

- [ ] 先写失败测试（test_epoch_log / test_metric_registry），断言行格式、schema、csv、动态指标键。
- [ ] 实现 epoch_log.py 与 metrics_registry.py（初始登记 recall@K/ndcg@K/expected_percentile_rank 由既有实现包装或后续 Task 接入）。
- [ ] 测试通过后提交 `feat(output): epoch 输出组件与指标注册表`

---

### Task 2：模板攻击（bandwagon/random/pgd/tpa）

**Files:**
- Modify: 各攻击 `fit.py`（epoch 循环/history 写入段）

- [ ] `fit.py`：epoch 循环改用 `epoch_section/log_train_line/log_eval_line`；
  history 元素补 `epoch_seconds`；`write_history` 统一落盘。
- [ ] 运行各攻击相关单测与全量回归。
- [ ] 提交 `refactor(output): 模板攻击 epoch 输出统一`

---

### Task 3：advinject

**Files:**
- Modify: `attacks/advinject/generate.py`、`evaluate.py`、`fit.py`、`common.py`（如需）

- [ ] generate：每轮 record 精简为 `{epoch, epoch_seconds, loss}`；用 epoch_log
  打印标准行；`__main__` 改用 `load_config`。
- [ ] evaluate/fit：`retrain_victim` 的逐 epoch 训练循环接入 epoch_log 与
  history schema。
- [ ] 跑 `tests.test_advinject_contract` 与全量回归。
- [ ] 提交 `refactor(output): advinject epoch 输出规范化`

---

### Task 4：lightgcn / mf / wmf

**Files:**
- Modify: `models/{lightgcn,mf}/train.py`、`models/wmf/{train.py,report.py}` 与回调

- [ ] 训练循环/回调统一调用 epoch_log（保留 eval_log.csv 路径语义），
  history 补 `epoch_seconds`，schema 对齐。
- [ ] 相关测试（test_history_completeness / wmf_*）+ 全量回归。
- [ ] 提交 `refactor(output): lightgcn/mf/wmf epoch 输出统一`

---

### Task 5：revisit 系模型（cml/itemae/itemcf/multvae/ncf）

**Files:**
- Modify: `models/revisit_training.py`、各 `models/*/train.py`

- [ ] `revisit_training.save_training_artifacts` 输出标准 history/eval_log；
  `ranking_report` 保持返回指标 dict。
- [ ] 各 train.py：逐 epoch 训练行 + 耗时；eval_every 周期全量评估写
  eval_log.csv；模型内部 fit 无逐 epoch 观测时记录最小 `{epoch, loss}`。
- [ ] 相关模型单测 + 全量回归。
- [ ] 提交 `refactor(output): revisit 系模型 epoch 输出统一`

---

### Task 6：AGENTS.md 与 skill 固化

**Files:**
- Modify: `AGENTS.md`、`.codex/skills/paper-code-implementation/{SKILL.md,references/usage_doc_template.md,assets/**}`

- [ ] AGENTS §3 新增“输出与产物规范（必须）”条款（每 epoch 耗时/全部指标/
  指标注册表驱动/history schema）。
- [ ] skill SKILL.md 训练章节与 usage_doc_template 产物说明补规范。
- [ ] 提交 `docs(output): AGENTS 与 skill 固化 epoch 输出规范`

---

### Task 7：全量验证

- [ ] 全量 unittest 通过；
- [ ] 冒烟：wmf/lightgcn preprocess、batch `--mode generate --dry-run`、
  任一攻击 `--mode data`；
- [ ] git grep：`history`/`eval` 打印不出现硬编码指标名单残留（除 registry）。
- [ ] 收尾提交（如有）并交付总结。

---

## Self-Review

- spec §2.1–§2.5、§3–§7 均有对应 Task；
- Task 1 接口名在后续 Task 复用，命名一致；
- 无占位步骤；每 Task 带测试与提交。
