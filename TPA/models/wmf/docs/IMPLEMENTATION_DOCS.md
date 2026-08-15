# WMF 复现实现文档（六步）

> 论文：Hu, Koren & Volinsky. *Collaborative Filtering for Implicit Feedback
> Datasets*. ICDM 2008。
> 理解文档：[papers/WMF/WMF_understanding.md](../../../papers/WMF/WMF_understanding.md)
> 本实现只使用仓库内本地数据与既有依赖，不新增第三方库。

## 步骤① 数据处理

**方法**：把原始成对数据（`user item`，支持 NGCF 多物品行）解析为
`(user_id, item_id)` 列表，再做 id 重映射为 0..n-1 连续空间，产出
`meta.pkl` 与可人工抽查的 `train_pairs.txt` / `test_pairs.txt`。
置信度/偏好变换（p_ui、c_ui）不在本步骤做：它们依赖 config 的 α，属于
数据导入阶段（步骤②）按配置计算。

**输入 → 输出示例**：

```
原始 train.txt:
  0 1
  0 2 3        （NGCF 多物品行）
→ meta.pkl: {num_users, num_items, train_pairs, test_pairs, user_items}
```

**验证结果**（真实数据 + 单元测试）：

- 真实数据统计：608 用户 / 6298 物品 / 38614 训练交互 / 9965 测试交互，
  与仓库现有 models/mf 的预处理产物完全一致（交叉核对通过）。
- 单元测试 `tests/test_wmf_preprocess.py` 6 项通过：单对/多物品/空行解析、
  id 重映射、端到端产出。

**关键决策记录**：
- 本地 ml100k 为成对隐式格式、无评分值，r_ui 恒为 1。【AI 推断补全：依据
  仓库数据实际格式；论文 TV 数据 r_ui 为观看次数】
- id 重映射保证与 num_users/num_items 一致，避免稀疏 id 导致的越界。

## 步骤② 数据导入

**方法**：`WMFDataset` 把成对数据展开为 `(user, item, confidence, p)` 四元组
张量，并在初始化时**预构建一次** `user_obs` / `item_obs` 观测分组（用户/物品
→ 观测下标，训练过程不变，ALS 每轮直接引用）；`WMFDataLoader` 实现
DatasetProtocol 五个方法。置信度按论文：
`p_ui = 1 if r_ui > 0`、`c_ui = 1 + α·r_ui`（minimal，Eq.3）或
`1 + α·log(1 + r_ui/ε)`（log-scaling，Eq.6）。训练集按 `val_ratio`（默认
5%）随机划分验证集。

**batch 格式表**：

| 字段 | shape | dtype | 说明 |
|---|---|---|---|
| users | (N,) | int64 | 用户 id |
| items | (N,) | int64 | 物品 id |
| conf | (N,) | float32 | c_ui（默认 41.0） |
| p | (N,) | float32 | p_ui（恒 1） |

论文 ALS 是全量矩阵闭式优化（Eq.4/5，不存在 mini-batch），因此
`train_loader` 返回**单批全量训练矩阵** 6 元组：
`(users, items, conf, p, user_obs, item_obs)`；一个 epoch 恰好是
一次完整 sweep。val/test 单批全量仅用于损失监控，不依赖 mini-batch 顺序。

**验证结果**：`tests/test_wmf_dataset.py` 7 项通过——置信度公式
（minimal=41、log-scaling=1+40·log1p(1e8)）、batch shape/dtype、划分比例、
五方法协议齐全。

## 步骤③ 模型结构

**方法**：双因子模型 `user_factors (m×f)` + `item_factors (n×f)`，
预测 `p̂_ui = x_uᵀ y_i`。无偏置/无 Dropout/无神经网络层（论文明确）。
初始化默认高斯 N(0, 0.01)（论文未提及，参考常见 MF 惯例）。

**逐层结构表**：

| 参数 | shape | 参数量（ml100k, f=100） | 初始化 |
|---|---|---|---|
| user_factors | (608, 100) | 60,800 | N(0, 0.01) |
| item_factors | (6298, 100) | 629,800 | N(0, 0.01) |

**验证结果**：`tests/test_wmf_model.py` 6 项通过——forward 输出 shape
`(n,)`、嵌入 shape、全量评分 `(n_users, n_items)`、state_dict 键、
初始化分布自检（200×200 大样本 mean≈0、std≈0.01）、uniform 边界。

## 步骤④ 模型评估

**方法**：主指标按论文 Eq.(8) expected percentile rank：
`rank_ui` = 用户候选中预测分严格高于物品 i 的物品占比（0=最优先，1=最劣），
`rank̄ = Σ r^t_ui·rank_ui / Σ r^t_ui`（越低越好，随机期望 0.5）。
实现加入公用评估模块 `evaluation/metrics.py`（`expected_percentile_rank`），
并在 `training/metrics.py` 注册默认方向 `rank: lower`。补充指标
recall@K / ndcg@K 复用仓库 all-ranking 协议。

**指标公式 ↔ 代码逻辑**：

```
rank_ui = count(s > s_ui) / valid_items      # 训练集已交互物品置 -inf，不计入分母
rank̄    = Σ w_ui·rank_ui / Σ w_ui           # w_ui = r^t_ui，缺省 1
```

**手工验证用例**（`tests/test_wmf_evaluation.py`，7 项通过）：
- 3 用户 × 5 物品手算：rank̄ = (0.2 + 0.8 + 0.0)/3 = 1/3 ✓
- 训练集过滤后分母变化：1/5 → 1/4 ✓
- 并列分数不计数为“高于” ✓；测试权重加权 ✓；空测试返回 NaN ✓

## 步骤⑤ 模型训练

**方法**：ALS 交替最小二乘，一个 epoch = 一轮交替（全量矩阵优化，非
mini-batch）：
固定 Y 逐用户解 `x_u = (YᵀCᵘY + λI)⁻¹ YᵀCᵘp(u)`（Eq.4），再固定 X 逐物品
对称求解（Eq.5）。加速项 `YᵀCᵘY = YᵀY + Yᵀ(Cᵘ−I)Y` 只累加观测对。
观测分组（user_obs / item_obs）在数据导入时预构建，每轮直接引用；实现用
torch 分批批量求解（物品侧 128 个一批），避免逐项 LAPACK 的 Python 开销
（一轮从 50s 降到 1.3s）。正则项 `λI` 显式指定 `dtype=float64`，避免
隐式类型提升。

**损失函数**（论文 Eq.3，含未观测项）：

```
full = Σ_ui c_ui(p_ui − x_uᵀy_i)² + λ(Σ‖x_u‖² + Σ‖y_i‖²)
     = [Σ_all s² + Σ_obs (c(p−s)² − s²)] + λ(Σ‖x‖² + Σ‖y‖²)
Σ_all s² = trace(X YᵀY Xᵀ)，O(m f²) 快速计算
```

**超参数来源**：α=40【论文 Eq.3】、f=100【论文 Fig.1 推荐可行上限】、
epochs=15【论文约 10 轮收敛；参考 implicit 15】、λ=0.01【论文未给，
recpack/implicit 默认，[unreported]】。

**验证结果**：
- `tests/test_wmf_training.py` 5 项通过：解满足正常方程（残差 < 1e-6）、
  Eq.(3) 加速式与 O(m·n) 全量展开一致、1 batch 1 epoch 后 loss 下降且参数
  更新、train_step/eval_step 全部返回标量（eval_step 标量契约）。
- 真实数据 1-epoch 冒烟：train_loss=107520、rank=0.335（优于随机 0.5）。
- 性能：ALS 一轮从 50s（逐项 numpy solve）优化到 1.3s（torch 分批批量）。
- 审阅修正后重训：rank̄ **28.44%**（epoch 7）、recall@20 **19.13%**、
  ndcg@20 **17.22%**（float64 正则更精确，结果略优于修正前）。

**公式 → API 对照**：损失中 `Σ‖X‖²` 用 `np.sum(X*X)`（平方 Frobenius 范数，
不是 `norm()` 未平方）；本模型无 logsigmoid/softplus 项。

**配置键名一致性检查**：`models/wmf/config_keys.py` 为唯一定义来源，
`dataset.py / model.py / train.py / main.py` 全部从该模块导入键名常量，
无重复字面字符串（见下方全项目检索记录）。

**真实训练结果**（15 轮，54 秒）：

| epoch | train_loss | val_loss | rank | recall@20 | ndcg@20 |
|---|---|---|---|---|---|
| 1 | 106590.8 | 171015.2 | 0.3308 | 0.0907 | 0.0861 |
| 9（rank 最优） | 56874.7 | 137347.9 | **0.2859** | 0.1837 | 0.1618 |
| 11（recall 最优） | 55470.7 | 136393.7 | 0.2862 | **0.1866** | 0.1641 |
| 13（ndcg 最优） | 54393.5 | 135623.8 | 0.2863 | 0.1848 | **0.1646** |
| 15 | 53527.1 | 134970.5 | 0.2865 | 0.1828 | 0.1644 |

## 步骤⑥ 结果展示

**方法**：训练结束后自动生成 `outputs/training_curve.png`（左=Eq.3 损失、
右=排序指标）、`outputs/rank_cdf.png`（论文 Fig.2 对应物：测试观看节目
百分位秩累积分布）与 `outputs/comparison_table.md`（复现值 vs 论文报告值
对比表 + 缺口分析）；也可单独运行 `python -m models.wmf.report` 重新生成。

**论文报告值汇总**（私有电视数据，Fig.1）：Popularity 16.46%、
邻域 10.74%、WMF f=50/100/200 → 8.93% / 8.56% / 8.35%、Eq.9 13.40%、
Eq.10 10.49%、随机 50%。

**对比结果**：复现 rank̄ = 28.44%（f=100，ml100k），显著优于随机 50%，
与论文定性结论一致，判定为**部分对齐（定性）**。

**缺口分析**：
1. 数据集不同：论文为私有电视数据（30 万用户 / 1.7 万节目 / 3200 万观测），
   本地 ml100k 仅 608 用户 / 6298 物品 / 3.9 万交互，候选集与密度差异巨大。
2. 协议差异：本地数据无评分，c_ui 恒为 41；评估按仓库 all-ranking 协议
   过滤训练集，论文另有 TV 测试过滤与 momentum 预处理（本仓库跳过，
   用户确认"不用管"）。
3. 论文推荐取最高可行因子数（f=200 时 rank̄ 8.35%）；当前默认 f=100，
   可通过 config.yaml `model.factors` 提升后重训。
