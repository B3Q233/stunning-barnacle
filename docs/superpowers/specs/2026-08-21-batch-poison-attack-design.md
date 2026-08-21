# 批量投毒攻击系统（v1）设计文档（最终版）

> 日期：2026-08-21 ｜ 版本：v1.1（按用户 v1.0 Design Freeze 重写）
> 设计原则：**插件化（Plugin）**、**配置继承（Config Deep Merge）**、**单一职责（SRP）**
> 关联计划：`docs/superpowers/plans/2026-08-21-batch-poison-attack.md`

## 1. 项目目标

仓库已有四种原子攻击（Bandwagon / Random / PGD / TPA），每个攻击通过一个 `config.yaml`
独立完成 classify → generate → fit → evaluation。Batch 系统**不重写攻击**，在不修改任何
现有攻击代码的前提下实现：批量生成原子攻击配置、按 popular/normal/cold 分层采样、
批量调度多个原子实验、统一整合实验结果。

两层架构：

| 层级 | 职责 |
|---|---|
| Atomic Attack | 一个目标物品的一次完整投毒实验 |
| Batch | 配置生成、批量调度、结果整合 |

```
Batch Config ──> Generator（配置生成）──> Atomic Config × N
      ──> Bandwagon / Random / PGD / TPA（Generate Poison Data → Fit Model）
      ──> Aggregate Results
```

Batch 完全不知道攻击算法，只通过 **Registry** 获取攻击插件。

## 2. 目录结构

### 2.1 攻击插件（现有四个，统一规范）

```
attacks/{bandwagon,random,pgd,tpa}/
├── config.yaml     # 默认配置（= 独立运行配置，无需维护 default_config.yaml）
├── registry.py     # 模型注册（受害模型）
├── classify.py / generate.py / fit.py / run.py
```

### 2.2 Batch 模块（新增）

```
attacks/batch/
├── config.yaml       # Batch 配置模板
├── registry.py       # Attack 插件注册器（AttackSpec）
├── generator.py      # 配置生成（四层 Deep Merge + 分层采样）
├── runner.py         # 调度器
├── aggregate.py      # 结果整合
├── run.py            # CLI
├── utils.py          # 路径/命名/JSON/deep_merge 公共工具
├── cache/classification/{dataset}/{model}/top{k}/   # 公共分类缓存（不入库）
├── docs/DESIGN.md / docs/USAGE.md
└── output/           # 批量实验产物（不入库）
```

## 3. Batch 输出结构

每次 Batch 以时间 `YYYY-MM-DD-HH:MM` 为唯一标识（batch_tag）：

```
attacks/batch/output/2026-08-21-15-30/
├── config.yaml        # Batch 配置快照
├── meta.json          # Batch 元信息
├── configs/bandwagon_ml100k_lightgcn_top10/
│   ├── popular/item32.yaml, item87.yaml
│   ├── normal/item510.yaml
│   └── cold/item251.yaml, item1203.yaml
├── runs/bandwagon_ml100k_lightgcn_top10/
│   ├── popular/item32/{checkpoints/, history.json, attack_comparison.md, config.yaml}
│   ├── normal/...
│   └── cold/item251/...
├── results.csv
├── summary.md
└── logs/runner.log
```

命名规则（公共信息只保留一次，不用超长文件名）：

| 对象 | 示例 |
|---|---|
| Batch | `2026-08-21-15-30` |
| 实验组 | `bandwagon_ml100k_lightgcn_top10` |
| 分层 | `popular` |
| 配置 | `item251.yaml` |
| 实验目录 | `item251/` |

## 4. Attack Registry（插件注册）

Batch 不直接 import 攻击，统一查询注册器：

```python
@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_path: str
    classify: Callable
    generate: Callable
    fit: Callable

register(name, config_path, classify, generate, fit)
get(name) -> AttackSpec
registered_names() -> List[str]
```

- `attacks/batch/registry.py` 启动时自动注册现有四个攻击
  （import 各攻击模块的 `classify.main / generate.main / fit.main`，不改攻击源码）；
- 新攻击 X 只需三步：建目录 `attacks/x/`（config.yaml + 四个阶段文件）→
  `register(...)` → Batch 配置 `attack.name: x`；Generator/Runner/Aggregate 零修改。

## 5. 配置系统（四层继承）

采用 **Deep Merge** 四层继承，优先级从高到低：

| 优先级 | 来源 |
|---|---|
| P1 | Generator 运行时字段（`target_items` / `run_tag` / `output.dir`，自动写入） |
| P2 | `override`（用户自定义覆盖） |
| P3 | Batch Config |
| P4 | Attack Config（`attacks/{attack}/config.yaml` 默认配置） |

Deep Merge 规则：嵌套 dict 递归合并，list / 标量整体覆盖；不修改入参。
**现有攻击 config 为扁平结构（`dataset` 在顶层）；Batch 配置使用 `experiment:` 包裹，
合并时 `experiment.*` 展开到顶层再参与合并，输出原子配置保持攻击期望的扁平结构。**

合并示例：

```
Attack 默认: attack: {ratio: 0.03, filler_size: 20}, training: {epochs: 30}
Batch:      training: {epochs: 10}
Override:   attack: {filler_size: 40}
最终:       attack: {ratio: 0.03, filler_size: 40}, training: {epochs: 10}
```

## 6. Generator（generator.py）

输入 `BatchConfig`，输出 `AtomicConfig[]`：

```
读 Batch Config → 查询 Registry → 加载 attacks/{attack}/config.yaml
→ Deep Merge(Batch) → Deep Merge(Override) → 读分类缓存
→ 每层采样 K 个目标物品 → 写入 configs/{group}/{tier}/item{id}.yaml
```

原子配置运行时字段（P1，Generator 自动写入）：
`attack.target_items = {strategy: specified, ids: [item]}`；
`run_tag = {batch_tag}-{tier}-item{id}`（经 sanitize，供攻击管线数据隔离）；
`output.dir = attacks/batch/output/{batch_tag}/runs`。
`batch` 与 `override` 段不会出现在原子配置中。

## 7. Runner（runner.py）

只负责调度，不负责攻击：

```
Generator → Atomic Configs → Classify（一次）→ for each config:
      generate.py → fit.py → 整理到 runs/{group}/{tier}/item{id}/
```

- classify 通过 `spec.classify` 执行一次（公共缓存，与攻击算法无关）；
- generate + fit 串行，单 GPU；
- 支持 `--dry-run`（只生成 configs/meta，不执行）；`--skip-classify`；`--max-targets`；
- fit 产物先落到 staging（fit.py 固定拼接 `{dataset}/{model}/{run_tag}`），
  runner 移动整理到分层目录（不改 fit.py）；
- 每次 Batch 写 `logs/runner.log`。

CLI：`python attacks/batch/run.py --mode all|generate|run|aggregate`
参数：`--batch-tag` / `--dry-run` / `--skip-classify` / `--max-targets`。

## 8. 公共分类缓存

```
attacks/batch/cache/classification/{dataset}/{model}/top{k}/
├── rec_freq.json   # {"popular": [...], "normal": [...], "cold": [...]}
└── meta.json       # {dataset, model, topk, checkpoint, generated_at}
```

- Generator 只读缓存，不重算；Runner 在缓存缺失时调 `spec.classify` 生成一次，
  再归一化拷贝到公共缓存（攻击侧 `ordinary` → 公共 `normal`）；
- 缓存与攻击算法无关，可跨攻击复用。

## 9. Aggregate（aggregate.py）

### results.csv（每原子实验一行）

| 列 | 说明 |
|---|---|
| attack / dataset / model | 实验组信息 |
| tier / item | 分层与目标物品 |
| target_hr@{k} / target_ndcg@{k} | 攻击效果（最优值） |
| recall@{k} / ndcg@{k} | 模型效用（投毒训练后整体指标） |

### summary.md（按层统计 + clean 基线）

| Tier | HR@{k} | NDCG@{k} |
|---|---|---|
| Popular | 0.214 ± 0.03 | 0.183 ± 0.02 |
| Normal | 0.336 ± 0.04 | 0.291 ± 0.03 |
| Cold | 0.487 ± 0.05 | 0.421 ± 0.04 |

底部附 `Clean Model Utility`（w_clean 在 clean 数据上的 recall@{k} / ndcg@{k}）。

## 10. meta.json

```json
{"batch_tag": "2026-08-21-15-30", "attack": "bandwagon", "dataset": "ml100k",
 "model": "lightgcn", "topk": 10, "tiers": ["popular","normal","cold"],
 "per_tier": 3, "total_runs": 9, "seed": 42}
```

作为实验索引、快速检索与结果复现依据。

## 11. 测试策略（stdlib unittest，CPU，fixture）

| 测试 | 内容 |
|---|---|
| test_registry | 插件注册 / get / 重复注册报错 / 未知攻击报错 |
| test_merge | Deep Merge：嵌套合并、三层优先级、标量/list 覆盖、不改入参 |
| test_generator | 采样确定性、四层合并结果、运行时字段、yaml 写出 |
| test_cache | ordinary→normal 归一化、缓存 meta、已有缓存复用 |
| test_runner | dry-run 生成 configs/meta 不执行；非 dry-run 调用 N×2 并整理目录、写 logs |
| test_aggregate | results.csv 列、mean±std、缺 history 跳过 |
| test_meta | meta.json 字段完整 |

E2E（手工）：ml100k，epochs=1，tiers=[cold]，per_tier=1。

## 12. Sprint v1 验收标准

| 模块 | 验收条件 |
|---|---|
| Registry | 新攻击零修改接入 |
| Config Merge | 四层配置正确合并 |
| Generator | 生成 `configs/{group}/cold/item251.yaml` |
| Runner | classify 一次，全部原子实验完成 |
| Cache | 公共分类缓存可复用 |
| Aggregate | 自动生成 results.csv 与 summary.md |
| Compatibility | Bandwagon / Random / PGD / TPA 无需修改源码 |

核心设计原则（最终）：插件化——每个攻击维护自己的 config.yaml 与执行入口，通过
Registry 注册；配置继承——攻击默认 → Batch 公共 → Override 用户覆盖 → Generator
运行时注入；单一职责——Generator 管配置生成、Runner 管调度、Aggregate 管汇总、
Atomic Attack 管具体攻击逻辑。
