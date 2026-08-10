# TPA（传递式路径投毒攻击）—— 使用文档

> 基于 attack-imp-direct-poison 攻击模板（no-subgoal）实现。
> 攻击语义：假用户画像 = 平庸基座 + 传递路径 + 目标物品，
> 路径沿物品共现图以 CF 距离最短路径自然过渡（无 PGD 版）。

## 1. 环境准备

使用项目虚拟环境（已安装 torch 等依赖）。所有命令在任意目录执行均可
（脚本内部已处理项目根目录路径）。

## 1.5 代理模型（黑盒模式）准备

TPA 默认是白盒（`surrogate.enabled: false`，用受害模型嵌入构造路径，攻击效果上界）。
要做真实的黑盒迁移攻击，先训练一个**攻击者自己的代理模型**：

```powershell
# 1) config.yaml 中设置 surrogate.enabled: true
# 2) 训练代理模型（不同划分种子，保存到 surrogate.checkpoint）
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\train_surrogate.py --config G:\Idea\TPA\attacks\tpa\config.yaml
```

之后 classify / paths 会自动使用代理模型嵌入，受害模型只出现在评估里。
要恢复白盒对照，把 `surrogate.enabled` 改回 `false` 并重跑 classify + paths + data + model。

## 2. 快速开始

### 第 1 步：推荐频次分类（classify）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --mode classify
```

产出 `attacks/tpa/data/rec_freq/{dataset}/{model}_top{k}.json`
（流行/普通/冷门三档，用于目标选择与统计）。

### 第 2 步：路径画像构造（paths，TPA 核心）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --mode paths
```

产出 `attacks/tpa/data/paths/{dataset}/profiles.json`：

```
├── targets        # 本次攻击目标
├── profiles       # 每个假用户画像（items = 基座 + 路径 + 目标）
│                  # 每画像含 path: {start, middle, hops, cf_dist_start_target}
└── path_stats     # 路径命中率 / 平均跳数 / 起终点平均 CF 距离
```

### 第 3 步：注入中毒数据（data）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --mode data
```

产出 `attacks/tpa/data/poisoned/{dataset}/meta.pkl + profiles.json + stats.json`。

### 第 4 步：投毒训练 + 对比评估（model）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --mode model
```

一条命令跑全流程：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\tpa\run.py --mode all
```

产出 `attacks/tpa/outputs/{dataset}/`：checkpoints / history.json /
attack_comparison.md（含目标物品 HR@K / NDCG@K）。

## 3. 配置文件详解（`attacks/tpa/config.yaml`）

```yaml
dataset: ml100k            # 受害数据集（与基线同口径）
mode: all                  # classify | paths | data | model | both | all
seed: 42

model:
  name: lightgcn           # 被攻击 / 打分的模型
  overrides: {}

surrogate:
  enabled: false           # true=代理（黑盒）；false=白盒（对照）
  model_name: lightgcn     # 代理架构
  checkpoint: models/lightgcn/outputs/surrogate/latest.pt
  training:
    epochs: 30
    split_seed_offset: 1   # 与受害模型不同的划分种子
    eval_every: 5

classification:
  k: 10                    # 推荐频次分类 Top-K（默认取 training.k）
  popular_ratio: 0.2
  batch_size: 1024
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt

attack:
  name: tpa
  ratio: 0.03              # 假用户数 = ratio × 真实用户数（与基线一致）
  base_size: 10            # 平庸基座物品数（高流行度池采样）
  path:
    strategy: shortest     # shortest（random_walk 预留）
    max_bridge_items: 3    # 中间桥接物品数上限（IndirectAD=1）
    per_hop_tau: null      # 每跳距离阈值（语义阶段启用）
    fallback: direct       # 无路径时回退：基座 + 目标
  target_items:
    strategy: specified
    category: cold
    count: 1
    ids: [251]             # 与基线相同目标

warm_start:
  enabled: true
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt

training:
  epochs: 30
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  neg_ratio: 1
  eval_every: 5
  k: 10
  device: cuda

evaluation:
  report_model_utility: true

> 攻击选优：`evaluation.metrics` 的 `target_ndcg@K` / `target_hr@K` 是中毒模型
> checkpoint 的选优指标（主指标 = `target_ndcg@K`，`--skip-train` 与对比报告
> 都加载按它选出的最优模型）；整体 `recall@K` / `ndcg@K` 仅作投毒代价参考。
> 每个评估 epoch 的目标物品明细（hr/ndcg/命中人数/平均排名）写入 `history.json`。

output:
  dir: attacks/tpa/outputs
```

### 目标物品怎么确定？

与基线一致：`strategy: specified` + `ids` 手动指定；或 `strategy: category`
从流行/普通/冷门中挑。路径构造会以目标为终点搜索。

### 怎么调整路径？

- `max_bridge_items`：控制多跳程度（消融 1/2/3/5/8 时改这里 + 重新跑 paths）
- `per_hop_tau`：语义融合阶段启用每跳距离约束
- `base_size`：平庸基座规模，越大画像越"大众化"但注入交互越多

## 4. 常见问题

**Q1：为什么有些画像显示"回退直连"？**
共现图稀疏时基座物品到目标可能没有 ≤ 上限的路径；回退为基座 + 目标，
等价两跳直连。可增大 `max_bridge_items` 或换数据集。

**Q2：改配置后需要重新跑哪些阶段？**
改 `target_items` / `base_size` / `path` 后必须重跑 `paths` + `data` + `model`；
只改 `training` 则从 `model` 开始即可。

**Q3：白盒和代理模式有什么区别？**
白盒（`surrogate.enabled: false`）用受害模型自己的嵌入构造路径，效果是上界但假设
攻击者能拿到受害模型；代理（`true`）只用攻击者自训模型，受害模型仅评估，结果才
代表真实迁移攻击。论文需要同时报两个版本。

**Q4：想加语义距离怎么办？**
当前数据无物品文本/图像特征；接入多模态数据集后，在 `path_builder.py` 的
`weighted_adj` 中把权重改为 λ·d_sem + (1-λ)·d_CF 并启用 `per_hop_tau`。
