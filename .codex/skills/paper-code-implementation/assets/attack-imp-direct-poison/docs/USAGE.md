# 攻击模板（attack-imp-direct-poison）—— 使用文档

> 本目录是从 bandwagon 实现提取的**攻击模板**（no-subgoal：不走六步模型复现顺序）。
> 复制为 `attacks/attack_imp_direct_poison/` 可直接运行；要改攻击名，全局替换
> `attack_imp_direct_poison` 即可（config 中的 `attack.name` 会自动驱动数据/输出路径）。

## 1. 环境准备

使用项目虚拟环境（已安装 torch 等依赖）。所有命令在任意目录执行均可
（脚本内部已处理项目根目录路径）。

## 2. 快速开始

### 第 1 步：推荐频次分类（classify，可选但推荐）

先有训练好的干净模型 checkpoint，然后对全量用户做 Top-K 推荐，统计每个物品的
出现次数并划分为 **流行（前 20%）/ 普通 / 冷门**，结果缓存后供目标选择和
filler 采样使用：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\attack_imp_direct_poison\run.py --mode classify
```

产出：

```
attacks/attack_imp_direct_poison/data/rec_freq/{dataset}/{model}_top{k}.json
├── counts        # 每个物品在 Top-K 中出现的次数
├── categories    # popular（流行）/ ordinary（普通）/ cold（冷门）三类 ID
└── summary       # 各档数量、流行阈值、热门/冷门样例
```

### 只生成中毒数据（不建模型）

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\attack_imp_direct_poison\run.py --mode data
```

产出：

```
attacks/attack_imp_direct_poison/data/poisoned/{dataset}/
├── meta.pkl          # 中毒后的训练数据（num_users 已 +k）
├── profiles.json     # 每个假用户的画像（目标 + filler 物品列表）
└── stats.json        # 注入统计（目标物品、假用户数、交互增减）
```

### 拟合中毒模型 + 对比评估

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\attack_imp_direct_poison\run.py --mode model
```

或一条命令跑全流程：

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\attack_imp_direct_poison\run.py --mode all
```

产出（`attacks/attack_imp_direct_poison/outputs/{output.dir}/{dataset}/`）：

```
├── checkpoints/best.pt        # 中毒模型（最优 recall）
├── checkpoints/latest.pt      # 最终模型
├── history.json               # 每个 epoch 的 train/val loss + recall/ndcg
└── attack_comparison.md       # clean vs poisoned 对比报告（含目标物品 HR@K / NDCG@K）
```

### 复用已有 checkpoint 快速出报告

```powershell
G:\Idea\.venv\Scripts\python.exe G:\Idea\TPA\attacks\attack_imp_direct_poison\fit.py --config <配置> --skip-train
```

## 3. 配置文件详解（`attacks/attack_imp_direct_poison/config.yaml`）

```yaml
dataset: ml100k            # 受害数据集
mode: all                  # classify | data | model | both | all
seed: 42                   # 随机种子

model:
  name: lightgcn           # 被攻击 / 打分的模型（注册表见本目录 registry.py）
  overrides: {}            # 可选：覆盖模型超参

classification:
  k: 10                    # 每用户取 Top-K 推荐（默认取 training.k）
  popular_ratio: 0.2       # 流行物品 = 推荐频次前 20%
  batch_size: 1024         # 评分矩阵分批大小（大数据集防爆显存）
  checkpoint: <干净模型 checkpoint 路径>

attack:
  name: attack_imp_direct_poison  # 攻击名，驱动数据/输出路径与日志前缀
  num_fake_users: 6        # 显式假用户数（优先）
  ratio: 0.01              # 或按比例：假用户数 = ratio × 真实用户数
  filler_size: 20          # 每个假用户交互的 filler 物品数
  target_items:
    strategy: specified    # specified=手动指定 | category=按分类挑 | coldest | random
    category: cold         # strategy=category 时：popular | ordinary | cold
    count: 3               # 目标物品数（假用户均分给各目标）
    ids: [251]             # strategy=specified 时填写（示例目标，可改成你的目标物品）

warm_start:
  enabled: true            # 用干净模型初始化中毒模型
  checkpoint: <干净模型 checkpoint 路径>

clean_checkpoint: <对比用的干净模型，缺省=warm_start.checkpoint>

training:
  epochs: 30
  batch_size: 256
  lr: 0.001
  weight_decay: 0.0001
  neg_ratio: 1
  eval_every: 5
  k: 20                    # Top-K 评估
  device: cuda             # 无 CUDA 时自动回退 CPU

evaluation:
  report_model_utility: true    # 是否对比 clean/poisoned 的模型效用（投毒代价检查）

output:
  dir: attacks/attack_imp_direct_poison/outputs
```

### 目标物品怎么确定？

- 首选：`strategy: specified` + `ids`，直接把攻击目标物品 ID 填进配置；
- 想从某档物品里挑：`strategy: category` + `category: popular / ordinary / cold`；
- 想看每档具体有哪些物品：读 classify 产出的 `data/rec_freq/` 下 JSON 的 `categories`。

### 怎么换被投毒的模型？

`model.name` 指定受害模型；新增模型时在本目录 `registry.py` 的 `AVAILABLE_MODELS`
登记模型类、数据集类和自身 `config.yaml` 即可。

### 怎么换攻击语义？

模板默认画像 = "流行 filler + 指定目标"（bandwagon 语义）。要换成随机/平均/多跳路径
等画像，只改 `generate.py` 的 `generate_fake_profiles` 一处，其余流程不变。

## 4. 常见问题

**Q1：为什么目标物品平均排名大幅提升，但 Top-K 命中还很少？**
目标物品如果选最冷门（流行度仅 1）的物品，排名提升明显但打进 Top-K 需要更大的
假用户数或更多 epoch；可增大 `num_fake_users`、增加 `epochs`，或改选稍热的目标。

**Q2：data 模式和 model 模式的区别？**
`data` 模式只做数据注入（不 import 任何模型代码）；`model` 模式才新建模型并训练。
两者解耦：可以先生成多份不同投毒比例的中毒数据，再分别训练对比。

**Q3：想换数据集？**
先把对应数据集预处理到 `models/{model.name}/data/processed/{dataset}/`（见模型复现
的预处理脚本），再把 `config.yaml` 的 `dataset` 改成目标数据集。
