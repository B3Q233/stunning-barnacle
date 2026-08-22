# PGD（投影梯度上升投毒攻击）使用文档

## 1. 项目结构

```
attacks/pgd/
├── config.yaml        # 唯一配置入口（dataset / model / classification / attack / training / evaluation）
├── registry.py        # 受害模型注册表（mf / lightgcn）
├── classify.py        # 第 1 步：推荐频次分类（流行/普通/冷门）
├── generate.py        # 第 2 步：PGD 梯度上升生成假用户画像 + 注入
├── fit.py             # 第 3 步：warm-start 投毒训练 + 对比评估
├── evaluate.py        # HR@K / NDCG@K / 模型效用报告
├── run.py             # classify / data / model / both / all 编排
├── data/
│   ├── rec_freq/{dataset}/{model}_top{k}.json   # classify 缓存
│   └── poisoned/{dataset}/{model}/{tag}/        # 中毒数据（meta.pkl / profiles.json / stats.json / config.yaml 快照）
├── outputs/{dataset}/{model}/{tag}/             # 中毒模型 + pgd_comparison.md（run_tag 隔离）
└── docs/             # DESIGN.md（本实现设计依据）+ USAGE.md
```

## 1.5 run_tag 实验隔离

每次实验带一个 run_tag：`--tag` > config `run_tag` > 自动当前时间
（`2026-08-07-14:20` 形式；Windows 目录名中 `:` 自动替换为 `-`）。
数据与输出都按 `{dataset}/{model}/{tag}/` 分层，目录内保存 config.yaml 快照，
互不覆盖；`data/poisoned/{dataset}/{model}/latest.json` 记录最近一次 data 生成的
tag，单独跑 model 阶段（不带 --tag）时会自动读取该 tag 衔接同一实验。

## 2. 环境准备

使用项目虚拟环境（已安装 torch 等依赖）：

```powershell
G:\Idea\.venv\Scripts\python.exe --version
```

## 3. 数据准备

受害模型（MF / LightGCN）需要先交付：预处理好的 `meta.pkl` + 训练好的干净
checkpoint。当前项目已有：

- LightGCN：`models/lightgcn/data/processed/ml100k/meta.pkl` +
  `models/lightgcn/outputs/checkpoints/latest.pt`
- MF：`models/mf/data/processed/ml100k/meta.pkl` +
  `models/mf/outputs/checkpoints/latest.pt`

## 4. 完整流程

### 第 1 步：交互数分类（可选但推荐）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\pgd\run.py --mode classify
```

产出 `data/rec_freq/{dataset}/lightgcn_top20.json`（目标选择与 filler 池来源）。

### 第 2 步：PGD 生成中毒数据

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\pgd\run.py --mode data
```

产出 `data/poisoned/{dataset}/lightgcn/{tag}/{meta.pkl, profiles.json, stats.json, config.yaml}`。

### 第 3 步：投毒训练 + 对比评估

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\pgd\run.py --mode model
```

或一条命令全流程：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\pgd\run.py --mode all
```

产出 `outputs/{dataset}/lightgcn/{tag}/pgd_comparison.md`（含目标物品 Clean/Poisoned
HR@K、NDCG@K 与模型效用）。

## 5. 怎么换被攻击的模型（MF / LightGCN）？

编辑 `attacks/pgd/config.yaml` 的 4 处：

```yaml
model:
  name: mf                       # lightgcn → mf
classification:
  checkpoint: models/mf/outputs/checkpoints/latest.pt
warm_start:
  checkpoint: models/mf/outputs/checkpoints/latest.pt
clean_checkpoint: models/mf/outputs/checkpoints/latest.pt
```

PGD 引擎自动切换：`mf → engine=als`（论文 §4.1 精确 KKT 梯度），
`lightgcn → engine=neighbor`（一阶线性化代理，见 DESIGN.md §2.2）。
不同模型的中毒数据与输出分目录存放（`data/poisoned/{dataset}/{model}`、
`outputs/{dataset}/{model}`），互不覆盖。仓库内也提供现成的 MF 变体配置
`tmp/pgd_mf_config.yaml` 可直接 `--config` 引用。

## 6. 配置详解（attacks/pgd/config.yaml）

| 参数 | 默认 | 说明 |
|------|------|------|
| dataset | ml100k | 受害数据集 |
| model.name | lightgcn | 受害/打分模型（mf / lightgcn）|
| classification.k / popular_ratio / medium_ratio | 20 / 0.05 / 0.40 | 交互数分类参数 |
| classification.checkpoint | lightgcn latest.pt | 仅聚合报告使用 |
| attack.ratio | 0.01 | 假用户比例（假用户数 = ratio × 真实用户数）|
| attack.filler_size | 20 | B：每个假用户交互的物品数（PGD 预算）|
| attack.target_items.strategy | specified | 目标选择策略 |
| attack.target_items.ids | [251] | strategy=specified 时的目标物品 ID |
| attack.pgd.iterations | 10 | PGD 外层迭代数 |
| attack.pgd.step_size | 0.2 | 梯度上升步长 s_t |
| attack.pgd.lambda_rating | 1.0 | 评分约束 Λ |
| attack.pgd.lambda_u / lambda_v | 0.05 | ALS/代理正则 |
| attack.pgd.engine | auto | als（MF）\| neighbor（LightGCN）\| auto |
| attack.pgd.init | popularity | M̃ 初始化池（popularity / random）|
| attack.pgd.include_target | true | 画像固定包含目标物品 |
| attack.pgd.utility.mu1 / mu2 | -1.0 / 1.0 | 混合效用权重（Eq.9）|
| attack.pgd.utility.w_target | 2.0 | 目标物品权重 |
| attack.pgd.utility.cross_target | true | 一阶交叉项开关 |
| warm_start.enabled / checkpoint | true / lightgcn latest.pt | 嵌入迁移初始化 |
| training.* | 30 epoch / 256 / 1e-3 | 投毒训练超参 |
| evaluation.report_model_utility | true | 是否报告投毒代价 |
| evaluation.metrics | target_ndcg@10: upper 等 | 指标与方向标注（upper=越高越好 / lower=越低越好；指标名中的 @K 是评估 K 的唯一权威；target_* 为攻击选优指标，整体 recall/ndcg 仅作投毒代价参考）|
| evaluation.checkpoint_mode | per_metric | 每个指标各存一份 `{指标}-best-model.pt`（默认）\| single=仅第一指标 |

训练完成后，`outputs/{dataset}/{model}/{tag}/checkpoints/` 下会出现
`target_ndcg@10-best-model.pt`、`target_hr@10-best-model.pt`、
`recall@10-best-model.pt`、`ndcg@10-best-model.pt` 等多份最优模型
（每指标一份，同 epoch 不去重），`history.json` 的 `best` 段记录每个指标最优时的
完整指标快照与对应 checkpoint 文件名；`--skip-train` 会按
`{首指标}-best-model.pt`（主指标 = `target_ndcg@10`）→ `best.pt`（旧）→
`latest.pt` 顺序加载。

### 目标物品怎么确定？

- 首选 `strategy: specified` + `ids`，直接把目标物品 ID 填进配置；
- 或 `strategy: category` + `category: popular/ordinary/cold`，从三档中自动挑；
- 查看各档有哪些物品：读 `data/rec_freq/{dataset}/{model}_top{k}.json` 的
  `categories`。

## 7. 常见问题

**Q1：为什么 data 模式会加载模型？**
PGD 需要用干净模型权重计算隐式梯度（白盒攻击）；这与 bandwagon 的纯数据层不同，
是 PGD 攻击的本质要求（见 run.py 注释与 DESIGN.md §2.2）。

**Q2：目标物品平均排名提升了但 HR@20 还是低？**
目标越冷门，初始 HR@20 越低（可能为 0），需要更多假用户（加大 ratio）或更多
训练轮数才能打进 Top-20；平均排名是更灵敏的指标。

**Q3：想只优化目标评分、不优化 filler？**
设 `attack.pgd.utility.cross_target: false`，回到论文式梯度（只把目标评分推向 +Λ）。

**Q4：换数据集怎么办？**
先把数据集预处理到 `models/{model}/data/processed/{dataset}/meta.pkl`
（参考 `models/mf/scripts/preprocess.py`），训练对应干净模型，再把
`config.yaml` 的 `dataset` 与 checkpoint 路径改掉。
