# 模型指标 K 统一装配——设计文档

> 日期：2026-09-17
> 状态：方案 A3 已人工确认（范围与边界冻结），待落实施计划
> 前置文档：`docs/superpowers/specs/2026-09-02-config-canonical-design.md`
> 参考模板：`TPA/docs/config-template.unified.yaml`

## 1. 背景与事故

### 1.1 现象

在 `TPA/models/lightgcn/config.yaml` 声明 `k: 10` 的前提下启动 LightGCN 训练，
落盘产物却出现 `recall@20` / `ndcg@20`：`eval_log.csv` 表头、`history.json` 的
`best` 键、checkpoint 文件名（`ndcg@20-best-model.pt`）全部是 20。

### 1.2 根因

2026-09-02 的 canonical 化（commit `1999423`）把 `evaluation.k` 提到顶层 `k`、
把 `data.dataset` 提到顶层 `dataset`，但 `models/{lightgcn,mf,wmf}/train.py` 的
模型主入口仍保留旧展平写法：

```python
for section in ["data", "model", "training", "evaluation"]:
    flat.update(raw[section])   # 顶层 k / dataset 不在其中，被静默丢弃
flat = apply_k(flat)            # 解析不到 k → resolve_k(default=20)
```

后果有两条，第二条更危险：

1. `k` 丢失 → `apply_k` 回退 `default=20`，`{k}` 模板展开成 `@20`，评测、选优、
   checkpoint 命名全部按 20 执行。
2. `dataset` 丢失 → `LightGCNDataset` 的兜底默认是 `gowalla`
   （`models/lightgcn/dataset.py:73`），训练可能静默换到另一个数据集。

### 1.3 同类风险（本次要一并消灭的类别）

| 位置 | 问题 |
|---|---|
| `models/{lightgcn,mf,wmf}/train.py` | 内联展平，丢失顶层 canonical 键 |
| `models/wmf/report.py:107` | 复制同一份内联展平 |
| `models/wmf/report.py:54` | 硬编码 `("rank", "recall@20", "ndcg@20")` 画曲线 |
| `models/registry.py::load_model_config` | 返回未经 canonicalize / `apply_k` 的原始 YAML，调用方拿到未展开的 `{k}` 模板 |
| `attacks/{random,bandwagon,pgd,tpa,uba}/fit.py` | `ev_defaults.get("k", 20)` 与兜底 `["recall@20", "ndcg@20"]` 形成第二权威 |
| `models/{lightgcn,mf}/train.py` 回调 | `config.get("k", 20)` 再次形成裸 20 兜底 |

历史证据：`TPA/attacks/pgd/outputs/ml100k/lightgcn/2026-08-09-23-30/pgd_comparison.md`
的配置为 `classification.k: 10` / `evaluation.metrics: recall@10, ndcg@10`，报告却是
`Top-20 / HR@20`，JSON `"k": 20`。

## 2. 目标：4 条不变量（可验收条款）

### I1 唯一权威

**K 只由 canonical 顶层 `k` 决定。** `evaluation.k / training.k / classification.k`
仅作为兼容输入被读取一次，映射为顶层 `k` 后即消失，不得成为后续链路的数据来源。

验收：任意 entry point 解析后的配置里，K 的取值等于该文件顶层 `k`；攻击侧优先级为
"攻击配置顶层 k > 模型配置顶层 k"，且两者都缺时只允许有一个中心兜底常量。

### I2 单次展开

`{k}` 模板只在统一装配入口展开一次。展开发生在配置加载阶段，训练回调、history、
eval_log、checkpoint 命名、攻击报告只接受已展开的指标名（如 `recall@10`）。

验收：进入训练/评估/报告的对象中不存在含 `{k}` 的指标名；同一 run 内出现的所有
`@K` 中 K 恒等于该 run 解析出的 `k`。

### I3 无静默回退

`k` 缺失按兼容默认 20 处理，但必须 `UserWarning`；`dataset` 缺失直接 `ValueError`，
不再回退 `gowalla`；无 `evaluation.metrics` 的旧模型允许按已解析的 K 派生
`recall@K / ndcg@K`，并给出可见告警。

验收：缺 `k` 的配置加载时 `assertWarns(UserWarning)`；缺 `dataset` 时 `assertRaises(ValueError)`；
派生兜底产生的指标名中的 K 等于解析出的 K，且不等于任何硬编码常量。

### I4 产物自证

同一 run 的 `config.yaml` 快照、`history.json`、`eval_log.csv`、checkpoint 文件名
必须互相自证：快照里的 `k` 与全部 `@K` 指标名一致，且该一致性由自动化测试守住。

验收：见 §5 的 L2 契约测试；任意模型声明 `k=X`，其解析后 metrics 中每个 `@K` 的
K 必须等于 X，`k: 10 + recall@20` 这类配置必须使测试变红。

## 3. 范围

### 3.1 在范围内（A3 = A2 + L2 契约测试）

- `TPA/training/config_utils.py`：新增统一装配入口（展平 + 展开 + 缺省策略）。
- `TPA/models/{lightgcn,mf,wmf}/train.py`：内联展平改为调用统一入口。
- `TPA/models/wmf/report.py`：同一份内联展平改为统一入口；曲线指标名不再硬编码。
- `TPA/models/registry.py`：`load_model_config` 返回 canonical + `apply_k` 后的配置。
- `TPA/attacks/{random,bandwagon,pgd,tpa,uba}/fit.py`：去掉作为 K 来源的裸 20，
  兜底指标按已解析 K 派生。
- `TPA/models/{lightgcn,mf,wmf}/docs/USAGE.md`：指标表改为 `{k}` 语义并注明 K 唯一来源。
- 新增 `TPA/tests/test_metric_k_unification.py`（L1–L4）。

### 3.2 明确不在范围内

- 不修改历史产物（`outputs/` 下已生成的 run 目录一律不动）。
- 不修改攻击算法、训练算法、指标计算语义（`evaluation/metrics.py`、
  `evaluation/attack_eval.py` 的数值口径不变）。
- 不修改 `training/framework.py` 的抽象接口（`TrainingConfig` 结构不变）。
- 不为缺 `evaluation.metrics` 的 5 个旧模型（itemcf/itemae/ncf/multvae/cml）
  补配置；它们继续走按 K 派生的告警兜底。
- 不引入新的第三方依赖；不新增 canonical 配置键。

## 4. 设计

### 4.1 组件与职责

```text
TPA/training/config_utils.py
├── DEFAULT_K = 20                      # 全仓库唯一的 K 兼容兜底常量
├── resolve_k(cfg, default=DEFAULT_K)   # 既有，默认值改为引用常量
├── flatten_model_config(raw, ...)      # 新增：canonical 嵌套 → 扁平，且不丢顶层键
└── build_training_config_from_yaml(p)  # 新增：唯一装配点（load_config → flatten → TrainingConfig）
```

接口签名（后续计划按此实现，不得改名）：

```python
DEFAULT_K: int = 20

def flatten_model_config(
    raw: Dict[str, Any],
    *,
    top_level_keys: Sequence[str] = ("dataset", "k", "seed", "mode", "run_tag"),
    require_k: bool = True,
    require_dataset: bool = True,
) -> Dict[str, Any]: ...

def build_training_config_from_yaml(
    path: str | Path,
    *,
    extra_overrides: Dict[str, Any] | None = None,
) -> "TrainingConfig": ...
```

### 4.2 数据流（改后唯一路径）

```text
config.yaml
   │ load_config()                 # yaml.safe_load + canonicalize（legacy section k → 顶层 k）
   ▼
canonical dict（顶层 dataset / k / run_tag + 分片）
   │ flatten_model_config()        # 分片先合并，顶层 canonical 键后覆盖（顶层赢）
   ▼
flat dict（dataset / k / metrics 已展开为 @10）
   │ apply_k()
   ▼
TrainingConfig → 训练回调 / history.json / eval_log.csv / checkpoint 名 / 报告
```

攻击侧是同规则的第二个入口：`run.py` 已执行 `canonicalize + apply_k`，
`fit.py` 只允许通过 `resolve_k`（默认取模型配置已解析的 k）读取 K，
不得再在攻击代码里写裸 20 作为 K 来源。

### 4.3 优先级与冲突规则

1. 分片键（`data/model/training/evaluation`）先合并进扁平结果。
2. 顶层 canonical 键（`dataset/k/seed/mode/run_tag`）后写入 → 顶层赢。
3. `canonicalize_config` 已负责"canonical 顶层 k 存在时忽略 legacy section k 并发警告"，
   本设计不改变该语义，只用测试锁住。
4. 攻击侧 K 优先级：攻击配置顶层 k > 模型配置顶层 k > `DEFAULT_K`（带告警）。

### 4.4 错误处理

| 情形 | 行为 |
|---|---|
| 缺顶层 `k` | `UserWarning`（提示声明 canonical 顶层 k），随后按 `DEFAULT_K` 继续 |
| 缺顶层 `dataset` | `ValueError`，消息含"缺少 canonical 顶层 dataset"与文件路径 |
| 指标名仍含 `{k}` | 契约测试判红（表示有入口绕过了统一装配） |
| 指标 `@K` 与顶层 `k` 不一致 | 契约测试判红（第二权威出现） |
| `evaluation.metrics` 缺失 | 按已解析 K 派生 `recall@K / ndcg@K` + 告警 |

### 4.5 与既有实现的关系

- `apply_k` 对扁平 dict 只展开 `metrics`，不会凭空创建 `training/evaluation`
  子字典（`test_flat_shape_metrics_expanded` 已锁定），因此扁平结果不会被污染。
- `canonicalize_config`、`load_config`、schema 校验均不改语义，只被复用。
- `load_model_config` 变为"canonicalize → merge overrides → apply_k"，返回值仍是
  同一份嵌套结构，调用方读取路径（`model.* / training.* / evaluation.*`）不变。

## 5. 测试设计（L1–L4，TDD 先 RED）

新增 `TPA/tests/test_metric_k_unification.py`，全部 stdlib `unittest`。

### L1 统一装配函数（单元）

| 测试 | 断言 |
|---|---|
| `test_flatten_keeps_top_level_k` | 顶层 `k:10` → `flat["k"] == 10` |
| `test_flatten_keeps_top_level_dataset` | 顶层 `dataset` 原样保留 |
| `test_flatten_expands_metric_templates` | `metrics == [{"recall@10":"upper"},{"ndcg@10":"upper"}]` |
| `test_top_level_k_beats_section_k` | `{k:10, evaluation:{k:20}}` → 10 |
| `test_legacy_section_k_mapped` | 仅 `evaluation.k:10` → 10 |
| `test_missing_k_warns` | `assertWarns(UserWarning)` |
| `test_missing_dataset_raises` | `assertRaises(ValueError)` |
| `test_does_not_mutate_input` | 入参 raw 不变 |

### L2 全模型契约（A3 核心）

契约测试必须走**运行时的真实解析链**，而不是扫描 YAML 文本：

```text
raw（models/*/config.yaml）
  → canonicalize_config
  → resolve_k
  → apply_k
  → 检查最终 metrics
```

- `test_every_model_config_metric_k_matches_resolved_k`：遍历 `models/*/config.yaml`，
  对每个解析后指标名 `m`：若 `metric_k(m)` 非空，则必须等于 `resolve_k(raw)`；
  若名字里仍残留 `{k}`，判红。`recall@{k} / ndcg@{k}` 模板合法（展开后比较）。
- `test_contract_catches_hardcoded_metric_k`：用 `{"k": 10, metrics: [recall@20]}`
  构造**必须判红**的配置，证明契约测试真的能抓住本次事故，而不是恒真。
- `test_lightgcn_entry_yields_k10_yelp2018`：`build_training_config_from_yaml(
  models/lightgcn/config.yaml)` → `k==10`、`dataset=="yelp2018"`、
  metrics 为 `recall@10/ndcg@10`。
- `test_mf_wmf_entry_preserve_declared_k`：mf 声明 20、wmf 声明 10 各自保留
  （证明统一的是"听顶层 k"，不是"统一改成 10"）。

### L3 攻击侧一致性

- `test_load_model_config_expands_k`：`load_model_config("lightgcn")` 的
  `evaluation.k == 10`，指标不含 `{k}`。
- `test_attack_fit_k_follows_config`：对 random/bandwagon/pgd/tpa/uba 逐个
  `build_training_config`，攻击配置 `k:5` → 结果 `k==5`，
  `eval_ks_from_metrics(...) == [5]`。
- `test_metrics_fallback_uses_resolved_k`：模型无 `evaluation.metrics` 时，
  兜底指标为 `recall@5 / ndcg@5`，且结果中不含 `@20`。

### L4 入口可测性守卫

- `test_model_entrypoints_use_shared_assembler`：用 `unittest.mock.patch`
  断言三个模型入口的 `main()` 调用统一装配入口（防止内联展平回流）。
- `test_wmf_report_metric_names_follow_history`：`report._metric_names_from_history`
  是纯函数（不绘图），从 history 的键推导 `["rank", "recall@10", "ndcg@10"]`，
  对只含 `@10` 的 history 不得产出 `@20`；`plot_training_curve` 缺省调用它。

## 6. 验收判据

1. §2 的 I1–I4 全部有对应自动化测试，且测试先 RED 后 GREEN。
2. 用户点名的 6 条事故测试在修复前确实失败：
   `test_flatten_keeps_top_level_k`、`test_flatten_keeps_top_level_dataset`、
   `test_flatten_expands_metric_templates`、`test_top_level_k_beats_section_k`、
   `test_lightgcn_entry_yields_k10_yelp2018`、`test_metrics_fallback_uses_resolved_k`。
3. `ml100k` + `epochs=1` smoke：`lightgcn` 主入口产出的 `eval_log.csv` 表头为
   `epoch,recall@10,ndcg@10`，checkpoint 为 `ndcg@10-best-model.pt`，
   快照 `dataset` 为执行时显式声明的数据集。
4. 全量回归 `python -m unittest discover -s tests -t . -v` 全绿。
5. 最终一句话验收：**任意模型声明 `k=X`，从模型配置进入训练、攻击、报告和产物后，
   都只能出现 `@X`；不存在 `@20` 的隐式第二权威。**

## 7. 风险与回滚

| 风险 | 缓解 |
|---|---|
| `load_model_config` 返回值变化影响调用方 | 只新增 canonicalize/apply_k，不改变嵌套结构与 overrides 语义；`tests/test_model_registry.py` 与 `pre` 侧回归覆盖 |
| `DEFAULT_K` 常量替换字面量时漏改 | 用测试锁定"攻击侧不得出现裸 20 作为 K 来源"的兜底行为 |
| 三模型入口行为变化影响历史命令 | 历史 run 产物不改；新 run 若缺顶层 `k` 会显式告警而非静默 |
| 回滚 | 本次仅涉及 6 个源码文件 + 1 个测试文件 + 3 份文档，可按文件粒度 revert |
