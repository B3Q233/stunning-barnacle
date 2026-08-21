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

## 4. 输出目录

```
attacks/batch/output/{batch_tag}/
├── config.yaml            # 批跑配置快照
├── meta.json              # 实验索引（attack/dataset/model/topk/per_tier/total_runs...）
├── configs/{攻击}_{数据集}_{模型}_top{k}/{层}/item{id}.yaml
├── runs/{同上}/{层}/item{id}/   # checkpoints + history.json + attack_comparison.md
├── results.csv            # 每原子实验一行
├── summary.md             # 按层 mean±std + Clean Model Utility
└── logs/runner.log
```

## 5. 配置说明（四层继承）

优先级从高到低：Generator 运行时字段（`target_items` / `run_tag` / `output.dir`）>
`override` > Batch 配置 > 攻击默认配置（`attacks/{attack}/config.yaml`）。

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
