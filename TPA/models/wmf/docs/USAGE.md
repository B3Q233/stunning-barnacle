# WMF（加权矩阵分解）模型使用文档

WMF = 论文《Collaborative Filtering for Implicit Feedback Datasets》
（Hu, Koren & Volinsky, ICDM 2008）的隐式反馈加权矩阵分解，ALS 交替最小二乘
求解。本实现集成在仓库 `TPA/models/wmf/`，复用仓库 `training.framework`
的 TrainableModel / DatasetProtocol 契约。

## 1. 项目结构

```
models/wmf/
├── config.yaml            # 唯一配置入口（每项带来源标注 [paper]/[ai]/[unreported]）
├── main.py                # 训练入口（--config / --resume / --tag / --epochs）
├── train.py               # 训练组装：ALS 循环 + 全量排序评估 + 实验归档
├── model.py               # WMFModel(TrainableModel) + ALS 求解（Eq.4/5/6）
├── dataset.py             # WMFDataset / WMFDataLoader（DatasetProtocol 五方法）
├── config_keys.py         # 配置键名常量（唯一定义来源，禁止各文件手写字符串）
├── scripts/preprocess.py  # 步骤①数据处理：成对文件 → meta.pkl
├── data/processed/ml100k/ # 预处理产物（meta.pkl 等，随仓库提交）
├── docs/
│   ├── IMPLEMENTATION_DOCS.md   # 六步实现文档
│   └── USAGE.md                 # 本文档
└── outputs/               # 实验结果（不入库）：{run_tag}/ 实验目录 +
                           # 稳定指针（latest.pt / history.json / 曲线 / 对比表）
```

## 2. 环境准备

使用仓库根虚拟环境（已含 numpy / scipy / torch / pandas / matplotlib / pyyaml）：

```powershell
G:\Idea\.venv\Scripts\python.exe -m pip install -r G:\Idea\requirements.txt
```

## 3. 数据集准备

数据集使用仓库本地 `TPA/data/raw/ml100k/`（成对格式 `user item`，
train.txt + test.txt），无需下载。若换数据集：把新数据放入
`TPA/data/raw/{dataset}/`（同名 train.txt / test.txt），再改 config 的
`data.dataset` 并重跑预处理。

## 4. 复现完整流程

```powershell
cd G:\Idea\TPA

# ① 数据处理（生成 models/wmf/data/processed/ml100k/meta.pkl）
..\.venv\Scripts\python.exe models\wmf\scripts\preprocess.py --dataset ml100k

# ②-⑤ 训练（ALS 15 轮，默认 f=100；完成后自动出曲线与对比表）
..\.venv\Scripts\python.exe models\wmf\main.py

# 可选参数：
#   --tag 2026-08-15-a    指定实验标签（缺省=当前时间）
#   --epochs 5            覆盖训练轮数（冒烟测试）
#   --resume              从 outputs/checkpoints/latest.pt 断点续训

# ⑥ 查看结果
notepad models\wmf\outputs\comparison_table.md
models\wmf\outputs\training_curve.png
models\wmf\outputs\rank_cdf.png   # 论文 Fig.2 对应物：观看节目秩累积分布
```

## 5. 配置文件详解

来源标注：`[paper]`=论文明确写出；`[ai]`=论文未给但可推断（注释给依据）；
`[unreported]`=论文未提及，用社区/框架默认值。

| 模块 | 参数 | 默认值 | 说明 / 可调范围 |
|---|---|---|---|
| data | `dataset` | ml100k | 数据集名；换数据集需重跑预处理 |
| data | `val_ratio` | 0.05 | 训练集随机划出验证集比例；0~0.5 均可，改后需重跑 |
| model | `factors` | 100 | 因子数 f；任意正整数，论文推荐取可行上限（200 更优但更慢），改变后需重训 |
| model | `alpha` | 40 | 置信度缩放 α；论文值 40，≥0 |
| model | `epsilon` | 1e-8 | log-scaling 常数；仅 `confidence_scheme=log-scaling` 生效 |
| model | `confidence_scheme` | minimal | `minimal`（1+αr）/ `log-scaling`（1+αlog(1+r/ε)） |
| model | `init_method` | normal | `normal` / `uniform`；论文未提及 |
| model | `init_std` | 0.01 | 初始化标准差；>0 |
| training | `lambda_reg` | 0.01 | L2 正则 λ；论文未给（recpack/implicit 默认），≥0 |
| training | `epochs` | 15 | ALS 交替轮数；论文约 10 轮收敛，加多不一定更好 |
| training | `device` | cpu | `cpu` / `cuda`；ALS 为 CPU 线性代数，GPU 收益有限 |
| training | `save_every_n_epochs` | 5 | checkpoint 保存间隔 |
| evaluation | `metrics` | rank / recall@20 / ndcg@20 | 指标名中的 @K 是评估 K 的唯一权威 |
| evaluation | `k` | 20 | 仅当指标无 @K 时回退 |
| evaluation | `eval_every` | 1 | 全量排序评估间隔（epoch） |
| evaluation | `checkpoint_mode` | per_metric | `per_metric`（每指标一份最优）/ `single` |

## 6. 常见问题

- **loss 很大（约 1e5）正常吗？** 正常。Eq.(3) 是全体 (u,i) 对的加权平方损失
  （含 c=1 的未观测项），量级天然很大；关注下降趋势与 rank̄ 即可。
- **一轮训练很慢？** 默认 f=100 一轮约 1.3 秒；若 f=200 约为 4 倍，属正常
  复杂度 O(f²N + f³n)。调小 `model.factors` 可显著加速。
- **换数据集后报找不到 meta.pkl？** 先跑 preprocess 生成
  `models/wmf/data/processed/{dataset}/meta.pkl`，并确认
  `data.processed_data_path` 指向正确。
- **rank 显示 NaN？** 说明测试正样本全部被训练集过滤（无有效观测），
  检查 train/test 划分是否有交集。

## 7. 与攻击流程集成（注册表 + ALS 训练分支）

模型注册统一收口到公有注册表 `models/registry.py`（四个攻击目录下的
`registry.py` 均为薄壳）。WMF 与 MF 已登记，攻击配置 `model.name` 可写
`wmf` / `mf` / `lightgcn`。

- **可用的接口**：`model_cls(cfg, num_users, num_items, edge_index)` 构造、
  `get_user_embeddings() / get_item_embeddings()`、checkpoint
  `model_state_dict` 加载、全量评分（classify 推荐频次分类 / 对比评估）。
- **fit 阶段（中毒训练）已适配**：`attacks/*/fit.py` 检测到 `WMFModel` 时
  自动走共享分支 `models.wmf.train.train_wmf_from_meta`——中毒数据对 ALS
  就是普通数据，直接构造全量训练矩阵（含假用户）逐轮 sweep，评估复用
  攻击模板的 recall@K / ndcg@K / 目标 HR@K / NDCG@K。
- **注意**：WMF 是纯 ALS 模型，没有 `embedding` 属性；攻击配置里的
  `warm_start.enabled` 会被分支自动跳过（打印警告并随机初始化），无需手动改。
