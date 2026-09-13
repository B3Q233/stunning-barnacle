# UBA 攻击 使用文档

面向第一次接触该模块的使用者：项目结构 → 环境准备 → 数据准备 → 复现流程 →
配置详解 → 常见问题。

## 1. 项目结构

```
TPA/attacks/uba/
├── config.yaml            唯一配置入口（canonical 键；每个键都有来源标注）
├── registry.py            模型注册薄壳（转发 models/registry.py）
├── classify.py            第 1 步：按训练集交互数分档（流行/普通/冷门）
├── estimate.py            第 2 步（可选）：代理模型模拟实验估计处理效应 Y
├── generate.py            第 3 步：预算分配 + 假档案注入 → 中毒 meta
├── fit.py                 第 4 步：warm-start 投毒训练 + 对比评估
├── evaluate.py            评估薄壳 + UBA 专属「目标用户群」指标
├── uplift.py              纯算法层：目标用户选择 / A'³ 三跳路径 / 分组背包 DP
├── run.py                 编排入口（--config/--mode/--tag）
├── data/                  实验数据（rec_freq / estimate / poisoned，均不入库）
├── outputs/               实验输出（checkpoints / history / 报告，均不入库）
└── docs/                  DESIGN.md（设计）/ USAGE.md（本文档）
```

## 2. 环境准备

使用仓库根虚拟环境（**不要**用系统 Python）：

```powershell
cd G:\Idea
G:\Idea\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

依赖已在 `requirements.txt`（torch / numpy / scipy / pyyaml），本模块不新增依赖。

## 3. 数据准备

UBA 复用仓库既有预处理产物，不需要额外下载：

| 需要的东西 | 路径 | 说明 |
|---|---|---|
| 干净 meta | `models/{model}/data/processed/{dataset}/meta.pkl` | 含 `num_users/num_items/train_pairs/test_pairs/user_items`；缺失时回退 `models/lightgcn/...` |
| 干净 checkpoint | `config.checkpoint.clean` | warm-start 与对比评估都用它；必须先训练好受害模型 |
| 数据集原始交互 | `data/implicit/raw/{dataset}/train.txt|test.txt` | 仅当需要重新预处理时使用 |

若要用论文数据集 ML-1M，可按下表核对规模（论文 Table 2）：5,950 用户 / 3,702 物品 /
567,533 交互；仓库当前的 `ml100k` 为 608 用户 / 6,298 物品 / 38,614 训练交互。

## 4. 复现流程（按真实操作顺序）

```powershell
cd G:\Idea\TPA

# ① 物品分层（目标选择与 filler 池用）
G:\Idea\.venv\Scripts\python.exe attacks/uba/run.py --mode classify --tag uba-ml100k

# ② 处理效应 Y
#   - method=path（默认）：可跳过本步，data 阶段会自动补算并缓存
#   - method=surrogate：必须先跑本步（很慢，训练 (H+1)×E 次代理模型）
G:\Idea\.venv\Scripts\python.exe attacks/uba/run.py --mode estimate --tag uba-ml100k

# ③ 预算分配 + 中毒数据
G:\Idea\.venv\Scripts\python.exe attacks/uba/run.py --mode data --tag uba-ml100k

# ④ 中毒训练 + 对比评估
G:\Idea\.venv\Scripts\python.exe attacks/uba/run.py --mode model --tag uba-ml100k

# 一条命令跑完整闭环（classify + estimate + data + model）
G:\Idea\.venv\Scripts\python.exe attacks/uba/run.py --mode all --tag uba-ml100k
```

注意：若步骤 ②③④ 分开跑，`--tag` 必须一致（不传 tag 时 `latest.json` 指针会把
data/model 衔接起来）；`config.yaml` 的 `mode` 字段等价于 `--mode` 的默认值。

产物：

| 阶段 | 路径 | 关键文件 |
|---|---|---|
| classify | `attacks/uba/data/rec_freq/{dataset}/{model}_top{k}.json` | 三档分类 + 交互数 |
| estimate | `attacks/uba/data/estimate/{dataset}/{model}/item{i}_h{H}_...json` | 处理效应 Y |
| data | `attacks/uba/data/poisoned/{dataset}/{model}/{tag}/` | `meta.pkl` / `profiles.json` / `stats.json` / `config.yaml` |
| model | `attacks/uba/outputs/{dataset}/{model}/{tag}/` | `history.json` / `eval_log.csv` / `checkpoints/*.pt` / `uba_comparison.md` / `target_user_metrics.md` |

## 5. 配置文件详解（canonical 键）

来源标注：`[paper]` 论文明确写出 / `[官方代码]` 官方实现取值 / `[ai]` 论文未给、
按依据推断 / `[unreported]` 论文未提及、沿用仓库惯例。

### 5.1 顶层

| 键 | 默认 | 含义与调整影响 | 来源 |
|---|---|---|---|
| `dataset` | `ml100k` | 数据集名；必须在 `training/paths.py` 的登记表中 | [ai] |
| `mode` | `all` | `classify/estimate/data/model/both/all`；`--mode` 优先 | [ai] |
| `seed` | 42 | 目标用户采样 / filler 采样 / DP 并列取小用的随机种子 | [unreported] |
| `k` | 10 | 统一评估 K，指标名里的 `{k}` 由它展开 | [paper] |
| `run_tag` | `null` | 实验隔离标签；缺省取当前时间；`--tag` 优先 | [ai] |
| `model.name` | `lightgcn` | 受害模型（`lightgcn/mf/wmf/ncf/itemae/itemcf/cml/multvae`） | [paper] |
| `model.overrides` | `{}` | 覆盖模型自身 config 的 `model` 段（如 `emb_dim`） | [ai] |

### 5.2 classification

| 键 | 默认 | 含义 | 来源 |
|---|---|---|---|
| `popular_ratio` | 0.05 | 交互数前 5% 记为流行（filler 池与 category 策略用） | [ai] |
| `medium_ratio` | 0.40 | 5%~40% 记为普通，其余为冷门 | [ai] |

### 5.3 attack

| 键 | 默认 | 含义与调整影响 | 来源 |
|---|---|---|---|
| `name` | `uba` | 攻击名（目录/产物/报告前缀） | [ai] |
| `num_fake_users` | 100 | 总预算 N；**改大**会同时放大攻击强度与投毒代价 | [paper] |
| `ratio` | `null` | `num_fake_users` 缺省时按 `ratio × 真实用户数` 计算 | [ai] |
| `filler_size` | 36 | 每个假用户除目标物品外的物品数；改大会让画像更"稠密" | [官方代码] |
| `target_items.strategy` | `specified` | `specified/category/coldest/random` | [ai] |
| `target_items.count` | 1 | **必须为 1**（多目标未实现，>1 报错） | [paper] |
| `target_items.ids` | `[251]` | `specified` 时的目标物品 id | [ai] |

### 5.4 attack.uba.treatment

| 键 | 默认 | 含义与调整影响 | 来源 |
|---|---|---|---|
| `method` | `path` | `path`=A'³ 三跳路径（分钟级，纯数据）；`surrogate`=代理模型模拟（小时级） | [paper] |
| `max_per_user` | 6 | 单用户预算上限 H；改大→DP 搜索空间更大、单用户最多能分到更多假用户 | [paper] |
| `repeats` | 10 | 仅 `surrogate`：每档位模拟重复次数 E；改小加速但估计噪声变大 | [paper] |
| `hit_k` | 20 | 命中判据 `rank ≤ hit_k`（官方 DPA 用 20） | [官方代码] |
| `alpha`/`beta` | 1.0/1.0 | 路径代理变换 $Y=\alpha\cdot(A'^3)^\beta$ | [paper] |

### 5.5 attack.uba.allocation

| 键 | 默认 | 含义 | 对应论文/官方 |
|---|---|---|---|
| `strategy` | `uba` | `uba`=DP 最优；`uniform_target`=目标用户均分；`random_all`=随机模板用户 | +UBA / +Target / baseline(way=1) |

### 5.6 attack.uba.target_users

| 键 | 默认 | 含义与调整影响 | 来源 |
|---|---|---|---|
| `strategy` | `cooccurrence` | 选人规则；`specified` 用 `ids` 完全指定 | [paper] + [ai] 代理 |
| `count` | 50 | 目标用户数；改大→每个用户平均预算变小 | [paper] |
| `category_size` | 20 | 共现类别代理大小；改大→候选更宽松 | [ai] |
| `max_category_interactions` | 10 | 类别交互数上界（轻交互用户优先） | [paper] |
| `ids` | `[]` | `strategy=specified` 的目标用户列表 | [ai] |
| `accessible_ratio` | 1.0 | `random_all` 的模板池比例；设 0.2 对齐论文"只能访问 20% 交互" | [ai] |

### 5.7 attack.uba.profile

| 键 | 默认 | 含义 | 来源 |
|---|---|---|---|
| `filler_source` | `template_user` | filler 来源：`template_user`=模板用户自身交互（论文"最相似用户"语义）；`popular`=流行池（bandwagon 语义）；`random`=全量随机（random 语义） | [paper] |

### 5.8 其余段

`checkpoint.clean`（干净权重）、`warm_start.enabled`（是否迁移原用户/物品嵌入）、
`surrogate.*`（代理模型名/权重/训练超参）、`training.*`（中毒训练超参）、
`evaluation.metrics`（指标与方向）、`output.dir`（输出根目录）与仓库其它攻击一致，
语义见 `TPA/docs/config-template.unified.yaml`。

**改 k 时只需改顶层 `k`**：`classification/training/evaluation` 与指标名里的 `{k}`
会自动绑定，不要在多处重复写 K。

## 6. 常见问题

1. **`treatment.method=surrogate` 报"需要 surrogate.enabled=true"**
   `surrogate` 段默认关闭。打开它并把 `surrogate.name` 设成支持 mini-batch 训练的模型
   （`mf`/`lightgcn`/`ncf`）；纯 ALS 的 `wmf` 没有 `train_step`，会直接报错。
2. **surrogate 支路太慢**
   训练次数 = (H+1) × E：默认 7×10 = 70 次代理重训。冒烟时把
   `treatment.max_per_user=2`、`treatment.repeats=1`、`surrogate.training.epochs=1`，
   或用 `treatment.method=path`。
3. **目标用户候选为空**
   说明没有用户"与该类别有少量交互且未交互目标物品"。放宽
   `max_category_interactions`（如 20）或 `category_size`，或改用
   `target_users.strategy=specified` 显式给出用户 id。
4. **DP 结果把大部分预算留给少数用户**
   这是论文 Algorithm 1 的正常行为：$Y_{u,i}(t)$ 通常随 t 边际递减，DP 会把预算集中到
   边际收益最高的用户。可看 `stats.json` 的 `allocation.histogram` 核对分配形状。
5. **CUDA 显存不足**
   先降 `training.batch_size`；surrogate 支路还会反复建模型，必要时
   `training.device=cpu`（慢但稳）。
6. **改了 `alpha/beta/H` 后结果没变**
   处理效应缓存文件名已编码这些参数，正常会重算；若手工改过缓存目录，请删除
   `attacks/uba/data/estimate/{dataset}/{model}/` 下对应文件再跑。
