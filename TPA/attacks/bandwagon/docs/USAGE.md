# Bandwagon（从众）攻击 —— 使用文档

## 1. 环境准备

使用项目虚拟环境（已安装 torch 等依赖）：

```powershell
G:\Idea\.venv\Scripts\python.exe --version
```

所有命令在任意目录执行均可（脚本内部已处理项目根目录 `G:\Idea\TPA` 的路径）。

## 2. 快速开始

### 第 1 步：推荐频次分类（classify，可选但推荐）

先有训练好的干净模型 checkpoint，然后对全量用户做 Top-K 推荐，统计每个物品的
出现次数并划分为 **流行（前 20%）/ 普通 / 冷门**，结果缓存后供目标选择和
filler 采样使用：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\bandwagon\run.py --mode classify
```

产出：

```
attacks/bandwagon/data/rec_freq/{dataset}/lightgcn_top20.json
├── counts        # 每个物品在 Top-K 中出现的次数
├── categories    # popular（流行）/ ordinary（普通）/ cold（冷门）三类 ID
└── summary       # 各档数量、流行阈值、热门/冷门样例
```

### 只生成中毒数据（不建模型）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\bandwagon\run.py --mode data
```

产出：

```
attacks/bandwagon/data/poisoned/{dataset}/
├── meta.pkl          # 中毒后的训练数据（num_users 已 +k）
├── profiles.json     # 每个假用户的画像（目标 + 热门物品列表）
└── stats.json        # 注入统计（目标物品、假用户数、交互增减）
```

### 拟合中毒模型 + 对比评估

```powershell
# 先有中毒数据（上一步），然后：
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\bandwagon\run.py --mode model
```

或在配置中设 `mode: all`，一条命令串联分类 + 数据生成 + 训练评估：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\bandwagon\run.py --mode all
```

产出（`attacks/bandwagon/outputs/{output.dir}/{dataset}/`）：

```
├── checkpoints/best.pt        # 中毒模型（最优 recall）
├── checkpoints/latest.pt      # 最终模型
├── history.json               # 每个 epoch 的 train/val loss + recall/ndcg
└── bandwagon_comparison.md    # clean vs poisoned 对比报告（含目标物品 HR@K / NDCG@K）
```

### 复用已有 checkpoint 快速出报告

训练过但想调整评估或换指标时，跳过训练直接评估：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\bandwagon\fit.py --config <配置> --skip-train
```

## 3. 配置文件详解（`attacks/bandwagon/config.yaml`）

```yaml
dataset: ml100k            # 受害数据集（ml100k / gowalla / yelp2018 / amazon-book）
mode: all                  # classify | data | model | both | all
seed: 42                   # 随机种子（假用户/目标选择/训练划分均依赖）

model:
  name: lightgcn           # 被攻击 / 打分的模型（注册表见 registry.py，当前仅 lightgcn）
  overrides: {}            # 可选：覆盖模型超参（emb_dim / n_layers / init_method ...）

classification:
  k: 20                    # 每用户取 Top-K 推荐（默认取 training.k）
  popular_ratio: 0.2       # 流行物品 = 推荐频次前 20%
  batch_size: 1024         # 评分矩阵分批大小（大数据集防爆显存）
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt  # 打分用干净模型

attack:
  name: bandwagon
  num_fake_users: 6        # 显式假用户数（优先）
  ratio: 0.01              # 或按比例：假用户数 = ratio × 真实用户数
  filler_size: 20          # 每个假用户交互的流行物品数
  target_items:
    strategy: specified    # specified=手动指定 | category=按分类挑 | coldest | random
    category: cold         # strategy=category 时：popular | ordinary | cold
    count: 3               # 目标物品数（假用户均分给各目标）
    ids: [90, 110, 146]    # strategy=specified 时填写（目标物品由你自行确定）

warm_start:
  enabled: true            # 用干净模型初始化中毒模型（原用户/物品嵌入迁移）
  checkpoint: models/lightgcn/outputs/checkpoints/latest.pt

clean_checkpoint: models/lightgcn/outputs/checkpoints/latest.pt  # 对比用的干净模型（缺省=warm_start.checkpoint）

training:
  epochs: 30
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  neg_ratio: 1
  eval_every: 1                 # 每轮全量评估并写入 history（可视化需要完整每轮数据）
  k: 20                    # Top-K 评估
  device: cuda             # 无 CUDA 时自动回退 CPU

evaluation:
  report_model_utility: true    # 是否对比 clean/poisoned 的模型效用（投毒代价检查）

> 攻击选优：`evaluation.metrics` 的 `target_ndcg@K` / `target_hr@K` 是中毒模型
> checkpoint 的选优指标（主指标 = `target_ndcg@K`，`--skip-train` 与对比报告
> 都加载按它选出的最优模型）；整体 `recall@K` / `ndcg@K` 仅作投毒代价参考。
> 每个评估 epoch 的目标物品明细（hr/ndcg/命中人数/平均排名）写入 `history.json`。

output:
  dir: attacks/bandwagon/outputs
```

### 目标物品怎么确定？

- 首选：`strategy: specified` + `ids`，直接把攻击目标物品 ID 填进配置，
  想攻击谁由你决定；
- 想从某档物品里挑：`strategy: category` + `category: popular / ordinary / cold`，
  会分别从模型推荐频次最高的流行物品、普通物品、冷门物品中挑相对最冷的 `count` 个；
- 想看每档具体有哪些物品：读 classify 产出的
  `data/rec_freq/{dataset}/lightgcn_top{k}.json` 里的 `categories`。

### 怎么换被投毒的模型？

`model.name` 指定受害模型（当前注册表只有 `lightgcn`）；新增模型时在
`attacks/bandwagon/registry.py` 的 `AVAILABLE_MODELS` 登记模型类、数据集类和
自身 `config.yaml` 即可，攻击模块会自动使用该模型的默认超参。

## 4. 嵌入矩阵扩容说明（重要）

注入 k 个假用户后，新模型的嵌入矩阵自动变为 `(M+k+N) × D`：

- 原 M 个用户行 + N 个物品行：warm-start 时从干净模型迁移（物品行整体后移 k 位）
- 新增 k 个假用户行：按 `init_method` 随机初始化（默认 N(0, 0.1)），**不**被 warm-start 覆盖
- 若 `warm_start.enabled: false`，则整个 `(M+k+N) × D` 矩阵全部随机初始化、从头训练

## 5. 常见问题

**Q1：为什么目标物品平均排名大幅提升，但 Top-20 命中还很少？**
目标物品默认选最冷门（流行度仅 1）的物品，从 ~3800 名提升到 ~550 名已是 7 倍曝光提升；
要打进 Top-20 可增大 `num_fake_users`（如 30）、增加 `epochs`，或把 `target_items.strategy`
改为 `random` 选稍热一点的物品。

**Q2：数据模式和模型模式的区别？**
`data` 模式只做数据注入（不 import 任何模型代码）；`model` 模式才新建模型并训练。
两者解耦：可以先生成多份不同投毒比例的中毒数据，再分别训练对比。

**Q3：跑 `train.py` 报 ModuleNotFoundError: training？**
已修复：`train.py` / `model.py` 内置了项目根目录路径引导，可直接运行。

**Q4：想换数据集？**
先把对应数据集预处理到 `models/lightgcn/data/processed/{dataset}/`（见 LightGCN 的
`scripts/preprocess.py`），再把 `config.yaml` 的 `dataset` 改成目标数据集。
