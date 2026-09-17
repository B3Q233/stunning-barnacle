# 配置文件同义键统一——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按 spec（`docs/superpowers/specs/2026-09-02-config-canonical-design.md`）把 16 份活跃 config 物理统一为 canonical 键，并实现 `training/config_utils.py` 的 canonicalization pipeline，业务代码只读 canonical，AGENTS.md 增加复现必须参考统一模板的要求。

**Architecture:** 中心 `canonicalize_config`（rename / restructure / semantic conversion / conditional fallback 四类显式处理器）→ schema 校验（合法叶子路径集合 + owner/domain + 语义唯一性）→ 业务代码只读 canonical。配置按 Phase 顺序迁移：规范 → AdvInject → 简单攻击 → batch → models。

**Tech Stack:** Python 3.12（仓库 `.venv`）、yaml、unittest（仅标准库）。

## Global Constraints

- canonical 唯一事实源：`TPA/docs/config-template.unified.yaml`；新配置键必须同步模板（schema 测试强制）。
- canonical 与 legacy 并存：canonical 生效、legacy 忽略并 WARNING；多 legacy 值不同 → `ValueError`。
- 业务代码只读 canonical；legacy 只允许出现在 `training/config_utils.py` 与 tests。
- `n_fakes`：>1 → `attack.num_fake_users`；0<值≤1 → `attack.ratio`。
- percentile/zone：纯改名不换算（percentile_head/upper_torso/lower_torso；target_items.zone）。
- `data.val_ratio` 可选、缺省 0.05、不机械添加；不改默认行为。
- surrogate 训练超参唯一 canonical：`surrogate.training.*`（含 weight_decay/unroll_steps）。
- 全量测试命令（PowerShell）：`python -m unittest discover -s tests -p "test_*.py" -v`（在 `TPA`）。
- 提交信息 Conventional Commits 中文；只 add 明确路径；每 Phase 提交一次。

---

### Phase 1：建立规范与兼容层

**Files:**
- Modify: `TPA/training/config_utils.py`
- Create: `TPA/tests/test_config_canonical.py`

**Interfaces:**
- Produces: `training.config_utils.canonicalize_config(cfg: dict) -> dict`、`load_config(path) -> dict`、`SCHEMA_LEAF_PATHS: set[str]`

- [ ] **Step 1: 冻结 canonical schema**
  解析 `TPA/docs/config-template.unified.yaml`（去掉注释）生成合法叶子路径集合；
  手工补充 batch 的 `batch.*` 与显式评分预留块若已在模板内则无需补。
- [ ] **Step 2: 实现 canonicalize_config（TDD 先写测试）**
  处理器按 spec §3.1：rename（model_name→name、l2→training.weight_decay、
  lambda_reg→weight_decay、output_dir→output.dir）、restructure
  （data/experiment.dataset→顶层 dataset、experiment.seed→seed、
  surrogate 平铺→surrogate.training.*、adv_*→attack.adv.*、
  evaluation.eval_every→training.eval_every、legacy k→顶层 k、
  clean_checkpoint/classification.checkpoint→checkpoint.clean）、semantic
  conversion（use_cuda→training.device、n_fakes 按值分支）、fallback
  （warm_start.checkpoint or checkpoint.clean）。
  实现为显式小函数列表 `_PIPELINE`，不用巨型 dict。
- [ ] **Step 3: 实现 load_config**：`yaml.safe_load` → `canonicalize_config`。
- [ ] **Step 4: 单测**
  - legacy→canonical：dataset/use_cuda/eval_every/output_dir/lambda_reg/
    clean_checkpoint/surrogate 平铺/adv_* 全部映射；
  - canonical 优先 + WARNING；多 legacy 冲突 ValueError；
  - n_fakes 分支；percentile/zone 纯改名；
  - path/domain schema（training.k 拒绝）；
  - 语义唯一性（surrogate.weight_decay/unroll_steps 不在合法集合）；
  - 活跃 config 全部叶子 ⊆ schema（先在 Phase 2-5 配置改写后逐步全绿）。
- [ ] **Step 5: 验证**
  `python -m unittest tests.test_config_canonical -v` 通过；全量回归仍绿
  （本 Phase 不改业务 config）。
- [ ] **Step 6: 提交** `feat(config): canonicalization pipeline 与 schema 校验`

---

### Phase 2：AdvInject（压力测试）

**Files:**
- Modify: `TPA/attacks/advinject/config.yaml`、`common.py`、`generate.py`、`data.py`、`classify.py`、`run.py`、相关 tests

- [ ] config.yaml 物理改为 canonical：dataset/seed/k/run_tag/output.dir/
  checkpoint.clean/training.*/evaluation.*/attack.{num_fake_users|ratio,
  target_items.{count,zone,strategy,ids},adv.*}/classification.percentile_*/
  surrogate.{name,training.*}。
- [ ] 代码读取点改用 canonical；Percentile 与 zone 保留原值语义。
- [ ] 全量测试 + `python -m unittest tests.test_advinject_contract -v`。
- [ ] 提交 `refactor(config): advinject 迁移 canonical 配置`

---

### Phase 3：简单攻击

**Files:**
- Modify: random / bandwagon / pgd / tpa 的 `config.yaml`、`fit.py`、`generate.py`、`classify.py`

- [ ] config：顶层 dataset/seed/k；`checkpoint.clean`；删除 clean_checkpoint/
  classification.checkpoint；training.eval_every 保留；其余键不变。
- [ ] 代码：`checkpoint.clean` 读取 + warm_start 回退；统一 load_config。
- [ ] 测试 + 提交 `refactor(config): random/bandwagon/pgd/tpa 迁移 canonical 配置`

---

### Phase 4：batch

**Files:**
- Modify: `TPA/attacks/batch/config.yaml`、`config.yelp2018_fast.yaml`、`generator.py`、`runner.py`、`aggregate.py`、`utils.py`

- [ ] batch 配置：experiment.dataset/seed → 顶层 dataset/seed；override.*
  内部键保持 canonical（不删除 override 协议）。
- [ ] 代码：读取边界映射；原子配置输出仍扁平 canonical。
- [ ] 测试（test_batch_*）+ 提交 `refactor(config): batch 配置 canonical 化`

---

### Phase 5：models

**Files:**
- Modify: models/{cml,itemae,itemcf,lightgcn,mf,multvae,ncf,wmf}/{config.yaml,dataset.py,train.py,report.py,config_keys.py(仅 wmf)}、`models/revisit_common.py`、`models/revisit_training.py`

- [ ] config：data.dataset→顶层 dataset；evaluation.k→顶层 k；
  evaluation.eval_every→training.eval_every；wmf lambda_reg→weight_decay。
- [ ] 代码读取 canonical（顶层 dataset 优先、eval_every 从 training）。
- [ ] 测试 + 提交 `refactor(config): models 配置 canonical 化`

---

### Phase 6：AGENTS.md 与全局收口

**Files:**
- Modify: `AGENTS.md`、`TPA/docs/config-template.unified.yaml`（如有遗漏键）

- [ ] AGENTS.md §3 增加配置规范条款（见 spec §5）；§6.2 补参考模板行。
- [ ] 全局 grep 验收：
  - 活跃 config 无 `use_cuda|output_dir:|lambda_reg|popular_percentile|
    torso_percentile|tail_percentile|n_fakes|clean_checkpoint|
    evaluation.eval_every|data.dataset:|experiment.dataset:`；
  - 业务代码无 legacy 读取（仅 config_utils.py 与 tests 允许）。
- [ ] 全量 unittest 通过；wmf/lightgcn preprocess 冒烟；batch
  `--mode generate --dry-run` 冒烟。
- [ ] 提交 `docs(config): AGENTS 增补统一配置模板引用`（若有模板修正同批）。

---

## Self-Review

- spec §1-§9 均有对应 Phase；schema/兼容层测试覆盖 §4.3 全部断言。
- 无占位步骤；每 Phase 结尾有可运行测试与提交。
- 接口命名在 Phase 1 定义并被后续 Phase 引用（canonicalize_config / load_config）。
