# TPA/pre —— Preliminary Representation Recovery Experiment

## 这个实验在问什么

**当一个目标物品的真实交互被逐渐删除后，现有投毒攻击还能否把它的表示重新推向原始表示？**

不是比较哪个攻击更强。目标物品取高交互物品（有足够交互可删），一次选定后
所有模型、所有攻击、所有删除比例都用同一批目标。

## 实验矩阵

```
Model(LightGCN, MF) × Attack(全部现有攻击) × DeletionRatio(0.1/0.2/0.3/0.5/0.8/0.9) × TargetItem(K)
```

K=20、A=6 时规模为 2×6×6×20 = 1440 个条件；但原始模型只有 2 份、
删除数据按 (model,item,ratio) 缓存复用，真正要跑的是 degraded + attacked 两类训练。

## 目录

```
pre/
├── configs/default.yaml      唯一配置入口
├── runners/                  编排：目标选择 / 退化 / 训练 / 调攻击
│   ├── common.py             路径、种子锚定、meta 读写、模型与 Dataset 统一构造
│   ├── targets.py            Phase 1 目标物品选择
│   ├── degrade.py            Phase 3 交互退化（确定性、缓存删除结果）
│   ├── train_model.py        Phase 2/4/5 训练与缓存（四件套）
│   ├── run_attack.py         Phase 5 调用既有攻击接口
│   └── pipeline.py           Phase 编排
├── analysis/
│   ├── align.py              Phase 11 统一空间映射（正交 Procrustes）
│   ├── distances.py          距离与恢复率定义
│   └── aggregate.py          Phase 12 汇总 CSV/JSON
├── visualization/plot_results.py   Phase 13（只读 CSV）
├── tests/test_pre_pipeline.py      stdlib unittest
└── outputs/                  产物（.gitignore 已覆盖 outputs/）
```

## Phase 0 接口审查结论（重要，都是实跑验证过的）

接入既有攻击时遇到并解决/记录的四个接口事实：

| # | 事实 | pre 的处理 |
| --- | --- | --- |
| 1 | `random / bandwagon / pgd / tpa / uba` 的 `generate.main(config, raw_meta=...)` **都接受自定义 meta** | 直接把"退化后的 meta"喂给攻击——这是本实验能成立的关键接口 |
| 2 | `advinject` 的入口是 **`generate(config)`**（不是 `main`），且通过 `config["data_path"]` 取数据 | 在 `ATTACK_SPECS` 里登记 per-attack 入口名与传参方式 |
| 3 | `advinject` 产出的中毒 meta 里 `user_items` 是**列表**，而 `LightGCNDataset/MFDataset` 要求 **dict**（`user_items.keys()`），二者开箱不兼容 | `common.normalize_user_items()` 做格式桥接（不改攻击、不改模型代码） |
| 4 | LightGCN 的邻接矩阵 `A_hat` **不在 state_dict 里** | 条件模型必须用当时那份 meta 重建；pre 为每个条件缓存 `meta_path` 引用 |

已知限制（不隐藏）：**TPA 的 `paths` 阶段（`path_builder`）把 meta 路径写死成干净数据**，
无法重定向。因此 TPA 的共现路径基于干净图构建，而被投毒的 meta 是退化后的。
本实验只删单个物品的交互，共现图几乎不变，但这一点记录在此。

另：仓库 WMF 训练路径**没有 `torch.manual_seed`**，pre 自己锚定种子，不依赖仓库默认行为。

## 用法

在 `TPA` 目录下执行：

```bash
# 单条件全链路验收（1 目标 × 1 比例 × 1 攻击 × 1 骨干）
python pre/run.py --mode all --tag smoke \
    --limit-items 1 --ratios 0.5 --attacks random --models lightgcn --epochs 5

# 分阶段
python pre/run.py --mode targets      # Phase 1  目标物品集合
python pre/run.py --mode original     # Phase 2  原始嵌入 e^0（真值）
python pre/run.py --mode degrade      # Phase 3  退化数据（缓存删除结果）
python pre/run.py --mode degraded     # Phase 4  退化嵌入 e^r（不攻击）
python pre/run.py --mode attack       # Phase 5  全部攻击 + 中毒训练
python pre/run.py --mode analyze      # Phase 11/12 统一空间映射与距离

# 全矩阵
python pre/run.py --mode all --tag full-run

# 可视化（独立，只读 CSV）
python pre/visualization/plot_results.py \
    --csv pre/outputs/full-run/results/embedding_distances.csv \
    --out pre/outputs/full-run/results/figures --alignment procrustes

# 单元测试
python -m unittest pre.tests.test_pre_pipeline -v
```

CLI 覆盖项：`--models / --attacks / --ratios / --epochs / --num-fake-users / --limit-items / --tag`。
所有阶段可重复执行，产物默认复用（`pre.cache.reuse`）。

## 核心定义

```
z^0     原始模型的目标物品嵌入（参考）
z^r     删掉比例 r 后、未攻击的嵌入
z^A,r   删掉 r 后、被攻击 A 推到的嵌入

d_deg = d(z^r,   z^0)
d_att = d(z^A,r, z^0)
RR    = (d_deg − d_att) / d_deg            ← 恢复率

RR > 0  攻击把物品推回原始表示
RR ≈ 0  攻击没有改变退化状态
RR < 0  攻击把物品推得更远
```

删除条数 = `floor(r × |I_i|)`，只删目标物品自身的真实交互，其余数据完全不动。

## 统一嵌入空间

每次训练都有自己的坐标帧（实测：同初始化重训的物品行平均余弦 0.956~0.975，
异初始化仅 −0.002）。pre 同时输出两套口径：

```
raw         条件模型自己的嵌入行
procrustes  正交对齐到原始模型后的嵌入行（R = UVᵀ，M = BᵀA，只在交互数 ≥ min_support 的行上拟合）
```

**对齐后的嵌入只能用于嵌入距离比较**：只旋转物品侧会破坏 `U·Vᵀ` 的自洽性，
不能拿去算曝光/排名。因此 pre 不提供曝光评估。

## 产物

```
pre/outputs/<tag>/
├── manifest.json            实验参数、git commit、argv
├── targets.json             目标物品集合（一次选定，全实验复用）
├── original/<model>/item_<id>/0/{model.pt,checkpoint.pt,embeddings.pt,metadata.json}
├── degraded/<model>/item_<id>/ratio_<pp>/
│   ├── train_data/meta.pkl          退化后的训练数据
│   ├── deleted_interactions.json    被删交互明细（可复现，不重新随机）
│   └── {model.pt,checkpoint.pt,embeddings.pt,metadata.json}
├── attacks/<model>/item_<id>/<attack>_ratio_<pp>/
│   ├── attack.json                  攻击状态 + 产物路径引用 + 攻击配置快照
│   └── {model.pt,checkpoint.pt,embeddings.pt,metadata.json}
└── results/
    ├── embedding_distances.csv      逐条件距离与恢复率（含两种对齐）
    ├── recovery_metrics.csv         按 model×attack×ratio 求均值的汇总
    └── summary.json
```

每个 (model, item, ratio, attack) 都留下"四件套"：目标物品计数、模型与配置、
训练/攻击文件路径、原始/退化/受攻击三种嵌入与距离。
攻击模块自己的大文件（中毒 meta、profiles）**只记路径引用，不复制**。

### CSV 字段

```
model, attack, item_id, deletion_ratio, alignment,
original_distance_cos, degraded_distance_cos, attacked_distance_cos, recovery_rate_cos,
original_distance_l2,  degraded_distance_l2,  attacked_distance_l2,  recovery_rate_l2,
attacked_norm, original_norm, degraded_norm,
remaining_interactions, original_interactions,
mean_row_cos_raw, mean_row_cos_aligned, procrustes_residual, item_row_cos_raw,
fake_users_added, poisoned_meta
```

其中 `attack=__none__` 是**未攻击的退化基线**（RR 恒为 0，作为参照线）。

## 验收状态（截至本次实现）

- 单元测试 7/7 通过（目标选择、删除条数=floor、删除确定性、距离、恢复率、Procrustes 正交还原）。
- 全链路冒烟通过：`LightGCN + MF × 6 个攻击 × 1 比例 × 1 目标`，
  28 条距离记录，每个条件都产出四件套。
- 冒烟结果为 3 epochs 的未收敛值，**不作为实验结论**。正式结论需按 `epochs: 30` 跑全矩阵。
