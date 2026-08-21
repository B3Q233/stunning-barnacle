# Batch 批量投毒攻击系统（v1）使用文档

## 1. 项目结构

```
attacks/batch/
├── config.yaml       # 批跑配置（experiment + 攻击差异项 + batch 扩展 + override）
├── registry.py       # Attack 插件注册器（bandwagon/random/pgd/tpa 自动注册）
├── generator.py      # 四层配置合并 + 分层采样 + 原子配置生成
├── runner.py         # 调度：公共分类缓存 + 逐个 data/model + 目录整理 + 日志
├── aggregate.py      # results.csv + summary.md + clean 基线
├── run.py            # CLI（generate/run/aggregate/all）
├── utils.py          # deep_merge / 路径 / 命名 / JSON
├── cache/            # 公共分类缓存（不入库）
├── docs/             # 本文档 + DESIGN.md
└── output/           # 批量实验产物（不入库）
```

## 2. 前置条件

- 受害模型（如 lightgcn）已完成六步复现：`models/{model}/data/processed/{dataset}/meta.pkl`；
- clean 模型权重（w_clean）已训练并记录路径，填入 `classification.checkpoint` 与
  `warm_start.checkpoint`；
- 数据集 meta 存在：`models/{model}/data/processed/{dataset}/meta.pkl`。

## 3. 运行方式

```powershell
cd G:\Idea\TPA

# 一键全流程（classify 一次 → 生成原子配置 → 逐个训练 → 整合）
G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --mode all

# 只生成配置（不训练）
G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --mode generate --dry-run

# 分步：先 generate，再 run，最后 aggregate
G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --mode generate
G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --mode run
G:\Idea\.venv\Scripts\python.exe attacks/batch/run.py --mode aggregate
```

可选参数：

| 参数 | 作用 |
|---|---|
| `--config PATH` | 批跑配置路径（默认 `attacks/batch/config.yaml`） |
| `--mode all\|generate\|run\|aggregate` | 执行阶段 |
| `--batch-tag NAME` | 指定 Batch 名（默认当前时间 YYYY-MM-DD-HH:MM） |
| `--dry-run` | 只生成 configs/meta，不执行训练 |
| `--skip-classify` | 复用已有公共分类缓存，不重新生成 |
| `--max-targets N` | 只跑前 N 个原子实验（冒烟） |

## 计时输出

全流程各环节会打印计时（格式：`【xx开始】` / `[xx结束 耗时X分Y秒]`），包括：
原子配置生成、批量执行、数据注入、中毒模型拟合（含每个 Epoch）、分类缓存生成、
结果整合与 Clean 基线计算。可用 `--max-targets 1` 先跑单个原子实验估算 yelp2018
等大数据集单轮耗时。

## 大数据集提速与复现注意事项

### 快速变体配置

`config.yelp2018_fast.yaml`：batch_size=1024、eval_every=5、epochs=10、
num_workers=4。默认 `config.yaml` 保持不变，保证论文复现口径稳定。

### batch 缩放实测（yelp2018，本机 RTX 3050）

LightGCN 的 step 时间由全图传播主导，与 batch 大小基本无关（实测 256/512/1024/2048
均约 53ms/step），因此 epoch 耗时随 batch 线性缩短：256→4.1min、1024→1.0min、
2048→0.5min。加大 batch 是收益最大的配置项。

### 评估语义（复现一致性）

`mean_rank_all` 采用论文 rank_ui 定义：**严格高于目标分的候选数 + 1，
并列分数共享同一排名**。旧版逐行 argsort 在并列时可能给出不同位次，因此相同数据下
指标与旧版本可能出现细微差异；复现/对比历史结果时请注意这一点。

### 评估频率建议

普通训练 `eval_every=5` 足够；攻击实验关注早期有效性（attack success curve），
建议 `eval_every=2~5`，最终复现时单独做一次全量评估。

### num_workers 说明

LightGCN 训练瓶颈在 GPU 传播与评估，负采样数据加载占比很小，`num_workers=4` 收益
有限（实测几乎无变化）；保留配置项便于在 CPU 训练等场景使用。

### 评估分数缓存（后续优化方向）

每轮全量评估会重新计算 User×Item 分数矩阵；如需画攻击曲线或做调试分析，可将
`ranking_scores` 的输出按 epoch 缓存到磁盘（如 `scores_epoch{n}.pt`）复用，
避免每轮重复计算。

## 4. 输出目录

```
attacks/batch/output/{batch_tag}/
├── config.yaml            # 批跑配置快照
├── meta.json              # 实验索引（attack/dataset/model/topk/per_tier/total_runs...）
├── configs/{攻击}_{数据集}_{模型}_top{k}/{层}/item{id}.yaml
├── runs/{同上}/{层}/item{id}/   # checkpoints + history.json + attack_comparison.md
├── results.csv            # 每原子实验一行 + 每层均值行（item=avg）
├── tier_stats.json        # 按层统计（四项指标 mean/std/n）
├── summary.md             # 按层 mean±std + Clean Model Utility
└── logs/runner.log
```

## 5. 配置说明（四层继承）

优先级从高到低：Generator 运行时字段（`target_items` / `run_tag` / `output.dir`）>
`override` > Batch 配置 > 攻击默认配置（`attacks/{attack}/config.yaml`）。

批跑配置**只写与攻击默认不同的项**（P3 差异），相同项由 P4 继承，避免同义配置重复维护；
临时覆盖用 `override` 段（Deep Merge 生效）。

### 统一评估 K

`k` 只需在**最外层定义一次**（如批跑配置顶层的 `k: 10`），
`classification.k` / `training.k` / `evaluation.k` 与指标名会自动绑定：
指标写 `{k}` 模板（如 `target_ndcg@{k}: upper`），加载时展开为 `target_ndcg@10`。
模型与攻击配置文件同样支持（模型取 `evaluation.k`，攻击取顶层 `k`）。

`batch` 段控制分层采样：

| 键 | 说明 |
|---|---|
| `tiers` | 分层列表：`popular` / `normal` / `cold` |
| `per_tier` | 每层采样目标物品数 K |
| `strategy` | `random`（seed 固定）或 `first`（层内前 K） |
| `seed` | 采样随机种子 |

`override` 段可覆写任意攻击参数（如 `attack.ratio`、`training.lr`），Deep Merge 合并。

## 6. 常见问题

- **classify 报 checkpoint 不存在**：确认 `classification.checkpoint` 指向已存在的
  w_clean 权重。
- **原子配置出现 batch/override 字段**：不应出现；Generator 会自动剔除。
- **runs/ 里没有历史**：检查 `attacks/{attack}/data/poisoned/{dataset}/{model}/`
  下是否有对应 run_tag 的中毒数据（fit 依赖 generate 阶段产物）。
- **换攻击**：改 `attack.name` 即可（pgd/random/tpa 已注册；tpa 分类缓存格式如不兼容
  需 v2 适配）。
