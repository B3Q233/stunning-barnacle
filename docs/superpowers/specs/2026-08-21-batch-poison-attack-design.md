# 批量投毒攻击系统（v1）设计文档 —— 优化版

> 日期：2026-08-21
> 状态：待用户确认（已按用户提供的《Batch 投毒攻击系统（v1）完整优化方案》重写）
> 关联计划：`docs/superpowers/plans/2026-08-21-batch-poison-attack.md`

## 1. 目标与设计原则

**目标**：在不修改现有 Bandwagon / TPA / PGD / Random 原子攻击代码的前提下，新增一个
Batch 配置生成与调度系统，实现分层采样、批量生成配置、批量训练与结果整合。

**两层架构**：

| 层级 | 职责 | 现状 |
|---|---|---|
| 原子攻击（Atomic） | 一个目标物品的一次完整投毒实验 | 已有 |
| Batch（批量） | 生成多个原子配置 + 调度运行 + 汇总结果 | 新增 |

Batch **不实现任何攻击算法**，只调用已有攻击入口。调用关系：

```
Batch Config ──> Generator（分层采样）──> Atomic Config × N
      ──> Bandwagon / TPA / PGD ──> Runs ──> Aggregate
```

## 2. 目录结构

```
TPA/attacks/batch/
├── config.yaml              # Batch 配置模板
├── generator.py             # 分层采样 + 配置生成
├── runner.py                # 调度器
├── aggregate.py             # 结果整合
├── run.py                   # CLI
├── utils.py                 # 路径/命名/JSON/meta 公共工具
├── cache/                   # 公共分类缓存（不入库）
│   └── classification/{dataset}/{model}/top{k}/rec_freq.json + meta.json
├── docs/DESIGN.md
├── docs/USAGE.md
└── output/                  # 批量实验产物（不入库）
```

## 3. Batch 输出结构（分层目录，避免超长文件名）

每次 Batch 是独立实验，以时间 `YYYY-MM-DD-HH:MM` 为唯一标识（batch_tag）：

```
attacks/batch/output/2026-08-21-15-30/
├── config.yaml                          # Batch 配置快照
├── meta.json                            # Batch 元信息（实验索引）
├── configs/
│   └── bandwagon_ml100k_lightgcn_top10/ # 实验组 = 公共信息只保留一次
│       ├── popular/item32.yaml, item87.yaml
│       ├── normal/item510.yaml
│       └── cold/item251.yaml, item1203.yaml
├── runs/
│   └── bandwagon_ml100k_lightgcn_top10/
│       ├── popular/item32/{checkpoints/, history.json, attack_comparison.md, config.yaml}
│       ├── normal/item510/...
│       └── cold/item251/...
├── results.csv
└── summary.md
```

命名规则（路径即语义）：

| 层级 | 示例 |
|---|---|
| 实验组 | `bandwagon_ml100k_lightgcn_top10` |
| 分类 | `cold` |
| 原子配置 | `item251.yaml` |
| 原子结果 | `item251/` |

## 4. Batch 配置 Schema

Batch 配置 = 原子攻击配置 + Batch 扩展。`experiment:` 包裹原原子配置顶层的
dataset/mode/seed；`batch:` 为批量扩展段，**生成原子配置时被自动删除**，
`experiment.*` 被展开回原子配置顶层（保持与现有攻击 config 兼容）。

```yaml
experiment:
  dataset: ml100k
  mode: all
  seed: 42

model:
  name: lightgcn
  overrides: {}

classification:
  k: 10
  popular_ratio: 0.2
  checkpoint: models/lightgcn/checkpoints/best.pt   # w_clean

attack:
  name: bandwagon
  ratio: 0.03
  filler_size: 20
  target_items: {strategy: specified, ids: []}       # 由 Generator 覆写

warm_start:
  enabled: true
  checkpoint: models/lightgcn/checkpoints/best.pt    # w_clean

training:
  epochs: 30
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  neg_ratio: 1
  device: cuda

evaluation:
  k: 10
  report_model_utility: true

output:
  dir: attacks/batch/output

batch:
  tiers: [popular, normal, cold]
  per_tier: 3
  strategy: random            # random（seed 固定）| first
  seed: 42
```

## 5. Generator 设计（generator.py）

输入 `BatchConfig`，输出 `AtomicConfig[]`。

流程：读 Batch Config → 读公共分类缓存 → 得到 popular/normal/cold 三层 →
每层采样 K 个 Item → 复制公共配置 → 覆写 `attack.target_items.ids=[item]` →
写入 `configs/{group}/{tier}/item{id}.yaml`。

原子配置生成规则：
- 深拷贝 Batch 配置，删除 `batch:` 段，展开 `experiment.*` 到顶层；
- 仅覆写 `attack.target_items = {strategy: specified, ids: [item]}`；
- `run_tag = {batch_tag}-{tier}-item{id}`（经 sanitize，用于攻击管线数据隔离）；
- `output.dir = attacks/batch/output/{batch_tag}/runs`（fit 阶段再整理到分层目录）。

## 6. Runner 设计（runner.py）

Runner 负责调度，不负责攻击：

```
Batch Config ──> Generator ──> Atomic Configs
      ──> Classify（仅一次）──> for item in configs:
            Data Generate ──> Model Train ──> runs/
```

- classify 只执行一次（公共缓存，与攻击算法无关）；
- Data + Model 顺序执行，单 GPU 串行；
- 支持 `--dry-run`（只生成 configs 与 meta.json，不执行攻击）；
- 由于 fit.py 的 out_dir 固定拼接 `{dataset}/{model}/{run_tag}`，runner 在
  model 阶段完成后把该 staging 目录**移动**到
  `runs/{group}/{tier}/item{id}/`，得到用户要求的层次结构（不改 fit.py）。

CLI：`python attacks/batch/run.py --mode all`
其他模式：`--mode generate` / `--mode run` / `--mode aggregate`；
附加参数：`--batch-tag` / `--dry-run` / `--skip-classify` / `--max-targets`。

## 7. 公共分类缓存

分类结果与攻击算法无关，作为公共缓存：

```
cache/classification/{dataset}/{model}/top{k}/
├── rec_freq.json   # {"popular": [1,5,8,...], "normal": [...], "cold": [...]}
└── meta.json       # dataset/model/k/checkpoint/生成时间
```

- Generator 只读缓存，不重算；
- Runner 在缓存不存在时：调用 `attacks.{attack}.classify.main`（复用已有实现，
  生成攻击侧缓存），再归一化拷贝到公共缓存；v1 支持 bandwagon/pgd/random，
  tpa 待 v2 适配；
- `attacks/batch/cache/` 加入 .gitignore（不入库）。

## 8. 结果整合（aggregate.py）

### results.csv（每原子实验一行）

| 列 | 说明 |
|---|---|
| attack | 攻击方法 |
| dataset | 数据集 |
| model | 受害模型 |
| tier | 分层 |
| item | 目标物品 id |
| target_hr@{k} | 攻击效果：目标 HR（最优值） |
| target_ndcg@{k} | 攻击效果：目标 NDCG（最优值） |
| recall@{k} | 模型效用（中毒训练后整体 recall） |
| ndcg@{k} | 模型效用（整体 ndcg） |

### summary.md（按层统计 + clean 基线）

| Tier | HR@{k} | NDCG@{k} |
|---|---|---|
| Popular | 0.214 ± 0.03 | 0.183 ± 0.02 |
| Normal | 0.336 ± 0.04 | 0.291 ± 0.03 |
| Cold | 0.487 ± 0.05 | 0.421 ± 0.04 |

底部附 `Clean Model Utility`（w_clean 在 clean 数据上的 recall@{k} / ndcg@{k}），
便于观察投毒代价。

## 9. meta.json

```json
{
  "batch_tag": "2026-08-21-15-30",
  "attack": "bandwagon",
  "dataset": "ml100k",
  "model": "lightgcn",
  "topk": 10,
  "tiers": ["popular", "normal", "cold"],
  "per_tier": 3,
  "total_runs": 9,
  "seed": 42
}
```

## 10. 测试策略（stdlib unittest，CPU）

| 测试 | 内容 |
|---|---|
| test_batch_config | 配置合法性（experiment/batch 段、per_tier>0、tier 白名单） |
| test_generator | 分层采样确定性、原子配置字段、experiment 展开、batch 删除 |
| test_yaml_writer | `configs/{group}/{tier}/item251.yaml` 正确生成 |
| test_runner | dry-run 生成 configs+meta 且不执行；非 dry-run 调用 N×2 次并整理目录 |
| test_aggregate | results.csv 列、mean±std 正确、缺 history 跳过 |
| test_meta | meta.json 字段完整 |

E2E（手工）：ml100k，epochs=1，tiers=[cold]，per_tier=1，验证整条链路。

## 11. Sprint v1 验收标准

| 功能 | 验收 |
|---|---|
| Generator | 能生成 `configs/{group}/cold/item251.yaml` |
| Runner | classify 一次，顺序完成全部原子实验 |
| Output | runs/ 与 configs/ 数量一致 |
| Aggregate | 自动生成 results.csv 与 summary.md |
| Isolation | 每次 Batch 独立时间目录，不覆盖历史 |
| Compatibility | 不修改任何现有攻击代码 |
