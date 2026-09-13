# UBA 攻击复现 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 `executing-plans`（或
> `subagent-driven-development`）按任务逐条实施本计划。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 在 `TPA/attacks/uba/` 复现论文 UBA（uplift 预算分配）攻击框架，遵守仓库攻击模块硬模板与统一配置 schema。

**Architecture:** 在模板 classify → data → model 三阶段上追加 `estimate` 阶段：`uplift.py` 提供纯算法（目标用户选择、A'³ 三跳路径代理、分组背包 DP、三种分配策略），`generate.py` 负责分配 + 画像注入（纯数据），`estimate.py` 负责 w/ S_φ 的代理模型模拟实验，`fit.py` 负责 warm-start 投毒训练与对比评估。

**Tech Stack:** Python 3.12、PyTorch 2.5、numpy、scipy.sparse、stdlib unittest、仓库 `training/` 与 `evaluation/` 公共层。

**设计依据:** `docs/superpowers/specs/2026-09-13-uba-reproduction-design.md`

## Global Constraints

- 注释中文，来源标注沿用 `[paper] / [ai] / [unreported] / [官方代码]`；配置每个键都要写"为什么/是什么/公式或出处/取值举例"。
- 只用仓库既有 canonical 键 + 新增 `attack.uba.*`；新增键必须同步写入 `TPA/docs/config-template.unified.yaml`，否则 `test_config_canonical.py` 失败。
- 运行命令统一：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_xxx -v`（工作目录 `G:\Idea\TPA`）。
- 实验隔离：数据 `attacks/uba/data/poisoned/{dataset}/{model}/{tag}/`，输出 `attacks/uba/outputs/{dataset}/{model}/{tag}/`，均写 config.yaml 快照；data 阶段写 `latest.json`。
- 不改 `models/*` 代码；模型选择一律经 `models/registry.py`。
- 不新增第三方依赖（`requirements.txt` 已有 torch/numpy/scipy/yaml）。

---

## Task 0: 清理与基线

**Files:** 无（只读验证）

- [ ] 确认 `.venv`、ml100k 预处理产物与干净 checkpoint 存在：
      `G:\Idea\.venv\Scripts\python.exe -c "import torch;print(torch.cuda.is_available())"`
- [ ] 记录回归基线：`G:\Idea\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v`（记录通过数量）
- [ ] 允许失败（已有历史失败项要先记录，避免与本次引入的失败混淆）

## Task 1: uplift.py 纯算法层（DP + 三跳路径 + 目标用户选择 + 分配策略）

**Files:**

- Create: `TPA/attacks/uba/uplift.py`
- Create: `TPA/attacks/uba/__init__.py`
- Test: `TPA/tests/test_uba_uplift.py`

**Interfaces（后续任务依赖的精确签名）:**

```python
def interaction_sets(train_pairs) -> Dict[int, set]                     # uid -> item set
def item_interaction_sets(train_pairs) -> Dict[int, set]                # iid -> user set
def cooccurrence_neighbors(item_users, target_item, size) -> List[int]   # 含 target_item，按共现数降序、id 升序
def select_target_users(meta, target_item, cfg, seed) -> Dict[str, Any]  # {users, strategy, category_items, ...}
def three_hop_path_effect(user_items, target_users, target_item, max_per_user, num_items, alpha, beta) -> np.ndarray  # (|U_t|, H+1)
def dp_allocate(values: np.ndarray, budget: int, max_per_user: int) -> Tuple[np.ndarray, float]  # (T*, best_value)
def allocate(values, users, strategy, budget, max_per_user, seed, all_users=None) -> Dict[str, Any]
def expand_allocation(allocation: Dict[int, int], num_users: int) -> List[Tuple[int, int]]  # [(uid, t)]
```

- [ ] **Step 1: 写失败测试**（DP 对暴力枚举 / 对官方 pack5 口径；三跳路径对朴素稠密 A'³；目标用户选择约束；分配策略预算）
- [ ] **Step 2: 运行确认失败**：`python -m unittest tests.test_uba_uplift -v` → ImportError
- [ ] **Step 3: 实现 `uplift.py`**（含矩阵 shape 注释、公式来源、边界检查）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交** `feat(attacks): UBA uplift 算法层（DP 分配 + 三跳路径处理效应）`

## Task 2: config.yaml + 统一模板同步 + registry/classify

**Files:**

- Create: `TPA/attacks/uba/config.yaml`
- Create: `TPA/attacks/uba/registry.py`
- Create: `TPA/attacks/uba/classify.py`
- Create: `TPA/attacks/uba/evaluate.py`
- Modify: `TPA/docs/config-template.unified.yaml`（新增 `attack.uba.*` canonical 块）

**Interfaces:**

- `classify.main(config) -> dict`，写 `attacks/uba/data/rec_freq/{dataset}/{model}_top{k}.json`（与 bandwagon 同 schema）。
- `evaluate.py` 转出 `evaluation/attack_eval.py` 的 `ranking_scores / compute_target_metrics / build_attack_eval_metrics / compare_models / format_report / save_report / aggregate_target_metrics`。

- [ ] **Step 1: 写 config.yaml**（每个键四要素注释；默认 `attack.num_fake_users=100`、`filler_size=36`、`uba.treatment.method=path`、`uba.treatment.max_per_user=6`、`hit_k=20`、`target_users.count=50`）
- [ ] **Step 2: 同步模板**：在 `config-template.unified.yaml` 的 `attack` 块内新增 `uba:` 子树（含注释与来源）
- [ ] **Step 3: 跑 schema 测试**：`python -m unittest tests.test_config_canonical -v` → PASS
- [ ] **Step 4: 写 registry.py / classify.py / evaluate.py**（薄壳）
- [ ] **Step 5: 跑 classify**：`python attacks/uba/run.py --mode classify` → 打印三档数量并落缓存
- [ ] **Step 6: 提交** `feat(attacks): UBA 配置与分类入口（同步统一模板）`

## Task 3: generate.py（data 阶段：分配 + 画像注入）

**Files:**

- Create: `TPA/attacks/uba/generate.py`
- Test: `TPA/tests/test_uba_generate.py`

**Interfaces（batch/generate 调度依赖）:**

```python
DEFAULT_RAW_META: Path            # models/{source_model}/data/processed/{dataset}/meta.pkl
DEFAULT_OUT_DIR: Path             # attacks/uba/data/poisoned/{dataset}/{model}
def load_meta(path) -> dict
def load_yaml_config(path) -> dict
def build_fake_profiles(allocation, meta, targets, cfg, rng) -> List[Dict[str, Any]]
def inject(meta, profiles) -> dict
def main(config, raw_meta=None, out_dir=None) -> dict   # 返回 stats
def effect_cache_path(config, model_name, target_item, method) -> Path
def load_or_build_effect(config, meta, target_item, target_users) -> dict
```

- [ ] **Step 1: 写测试**（合成 meta：注入数量断言、假 uid 起始、目标物品在每个画像、filler 不重复、stats schema）
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现 generate.py**（含 run_tag 隔离、config 快照、latest.json、边界检查）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 真实数据冒烟**：`python attacks/uba/run.py --mode data --tag uba-smoke`
- [ ] **Step 6: 提交** `feat(attacks): UBA 数据阶段（预算分配 + 假档案注入）`

## Task 4: estimate.py（w/ S_φ 支路：代理模型模拟实验）

**Files:**

- Create: `TPA/attacks/uba/estimate.py`

**Interfaces:**

```python
def main(config) -> dict                       # 计算 Y 并写缓存
def treatment_effect_surrogate(config, meta, target_item, target_users) -> dict
```

- [ ] **Step 1: 实现模拟实验**（t=0..H × E 次：造 D_f → 重训代理 → 目标用户 Top-K 命中）
- [ ] **Step 2: 用 `<dataset>/<model>` 的小规模配置跑一次**（`epochs=1, repeats=1, max_per_user=2`）确认可跑通并落缓存
- [ ] **Step 3: 提交** `feat(attacks): UBA 代理模型处理效应估计阶段`

## Task 5: fit.py（model 阶段：投毒训练 + 对比评估）

**Files:**

- Create: `TPA/attacks/uba/fit.py`
- Modify: `TPA/evaluation/attack_eval.py`（`REPORT_NAMES`/`REPORT_TITLES` 增加 `uba`）
- Modify: `TPA/tests/test_attack_fit_consistency.py`（ATTACK_DIRS 增加 uba）

**Interfaces:** `def main(config, skip_train=False, tag=None) -> dict`

- [ ] **Step 1: 实现 fit.py**（warm-start 迁移断言 + per_metric checkpoint + 报告；追加"目标用户群"指标段）
- [ ] **Step 2: 跑 `python attacks/uba/run.py --mode model --tag uba-smoke --skip-train` 之外的完整 model 阶段（epochs 小值）**
- [ ] **Step 3: 校验产物**：`history.json` 含 `{history, best}`、`eval_log.csv`、`uba_comparison.md`
- [ ] **Step 4: 提交** `feat(attacks): UBA 中毒训练与对比评估`

## Task 6: run.py 编排 + 批量注册 + 文档

**Files:**

- Create: `TPA/attacks/uba/run.py`
- Modify: `TPA/attacks/batch/registry.py`（注册 uba）
- Modify: `TPA/tests/test_batch_registry.py`（断言含 uba）
- Create: `TPA/attacks/uba/docs/DESIGN.md`、`TPA/attacks/uba/docs/USAGE.md`
- Modify: `papers/UBA/UBA_understanding.md`（回填交叉验证表，非入库文件）

- [ ] **Step 1: 写 run.py**（`classify|estimate|data|model|both|all`，`--tag` 优先）
- [ ] **Step 2: 注册 batch 并断言**
- [ ] **Step 3: 写 DESIGN/USAGE**（含 H 差异、hit_k=20、后端范围、批量限制）
- [ ] **Step 4: 全流程 `--mode all`（path 分支）跑通一次并记录产物**
- [ ] **Step 5: 提交** `feat(attacks): UBA 编排入口、批量注册与文档`

## Task 7: 回归与交付

- [ ] `python -m unittest discover -s tests -t . -v` 全量通过（与 Task 0 基线对比）
- [ ] `git status` / `git diff --stat` 自查，只提交与任务相关文件
- [ ] 汇总：三阶段门禁结果、与论文的差异清单、后续可做项（AIA/AUSH/Leg-UP 后端、多目标、w/ S_φ 全量实验）
