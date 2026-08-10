# TPA（传递式路径投毒攻击）设计文档

> 对应研究方案《基于物品共现图的传递式多跳中毒攻击》，
> 基于 attack-imp-direct-poison 攻击模板（no-subgoal）实现。
> 本版本已冻结决策：**无 PGD**（纯图路径注入）、**最短路径法**、**数据集不变**
> （当前 ml100k，与 bandwagon / random 基线同口径）。

## 0. 提取信息

- 方法来源：研究方案 §2.3（路径构造）与 §2.2（平庸基座）；IndirectAD
  (arXiv:2511.05845) 为两跳桥接基线，TPA 将其扩展为多跳传递路径
- 受害模型：由 `config.model.name` 指定（注册表见 `registry.py`，当前 lightgcn）
- 本模块：`attacks/tpa`（与模型代码解耦）

## 1. 攻击定义

【研究方案明确写出】TPA 假用户画像采用三段式框架：

```
假用户画像 = 平庸基座（高流行度采样） + 传递路径（共现图最短路径） + 目标物品
```

与 bandwagon（随机热门 filler + 目标）不同，TPA 的 filler 由一条
**沿物品共现图自然过渡**的多跳路径构成，目标是在攻击有效性与行为隐蔽性之间
取得平衡（方案 §1.3 核心洞察：把"一次大跳变"拆成"多次小跳变"）。

## 2. 已冻结的实现决策（v1）

| 决策点 | 选择 | 说明 |
|--------|------|------|
| 对抗优化 | 不做 PGD | 纯图路径注入；复合损失 + PGD 留待下一阶段 |
| 路径策略 | 最短路径法（Dijkstra） | 确定、可复现，便于路径长度消融；random_walk / 锚点桥接预留 |
| 距离度量 | λ=0 纯 CF 距离 | `1 - cos(e_i, e_j)`（干净模型物品嵌入）；语义融合 d'=λ·d_sem+(1-λ)·d_CF 待多模态数据集 |
| 图定义 | 物品共现图 | 同一用户交互集合内两两连边，路径只走共现边 |
| 路径约束 | 跳数上限 = max_bridge_items+1 | 默认 3 个桥接物品（4 跳）；每跳 τ 阈值暂不启用（per_hop_tau: null） |
| 平庸基座 | 高流行度池 P(i) 采样 base_size 个 | 语义接近兴趣中心 c 的约束留待语义阶段；起点=基座中离目标 CF 最近且有可行路径者 |
| 无路径回退 | direct | 基座 + 目标（等价两跳直连），计入 path_stats |
| 模型访问假设 | 支持白盒 / 代理两种模式 | `surrogate.enabled: false`=白盒（用受害模型嵌入，攻击效果上限）；`true`=代理（黑盒，路径/分类只用代理模型嵌入，受害模型仅评估） |
| 数据集 | 当前 ml100k | 与基线同口径：608 用户 / 6298 物品，Top-10，目标 251，3% 假用户 |

## 3. 模块设计（四阶段，no-subgoal）

| 模式 | 职责 | 是否新建模型 |
|------|------|--------------|
| `classify`（classify.py） | 推荐频次分类（流行/普通/冷门），供目标选择与统计 | 否（只读 checkpoint） |
| `paths`（path_builder.py） | 共现图 + CF 最短路径 + 基座/路径/目标画像缓存 | 否（只读 checkpoint） |
| `data`（generate.py） | 读路径缓存 → 注入中毒数据 | 否（纯数据层） |
| `model`（fit.py） | 按 `model.name` 新建受害模型（warm-start）→ 训练 → 对比评估 | 是（新实例） |
| `train_surrogate.py` | 训练代理模型（不同划分种子），供黑盒模式使用 | 是（代理实例） |

### 白盒 vs 代理（黑盒）说明

- **白盒（v1 默认对照）**：`surrogate.enabled: false`，路径距离 / 频次分类使用
  受害模型自身嵌入——攻击效果是"信息上限"，论文中作为上界对照
- **代理（黑盒，推荐）**：`surrogate.enabled: true`，攻击者只用自己的代理模型
  （`train_surrogate.py` 以不同划分种子训练）构造路径与分类；受害模型只出现在
  评估（clean_checkpoint）与可选 warm-start 中
- 两种模式的缓存按 `_proxy` 后缀隔离，不会互相覆盖；`stats.json` 记录 surrogate 开关
- 后续 PGD 对抗优化同样应只对代理模型求梯度，再迁移到受害模型

数据流：

```
clean 模型（checkpoint）──classify──▶ rec_freq.json（流行/普通/冷门）
                          └──paths──▶ paths/profiles.json（基座+路径+目标）
                                          │
clean meta.pkl ──generate（读缓存注入）──▶ poisoned meta.pkl
                                          │
                                          ▼ fit（model.name）
                                      新受害模型（BPR 训练）
                                          ▼
                            clean 模型 ──对比评估──▶ attack_comparison.md
```

## 4. 评估协议

- 模型效用：all-ranking recall@K / ndcg@K，投毒代价检查
- 攻击效果：目标物品 Clean/Poisoned 的 **HR@K** 与 **NDCG@K**，外加平均排名
- checkpoint 选优：中毒模型按目标物品 `target_ndcg@K` / `target_hr@K` 选优
  （主指标 `target_ndcg@K`，`--skip-train` 与对比报告加载它），
  整体 recall/ndcg 仅作投毒代价参考
- 对比口径：与 bandwagon / random 相同（同一干净模型、同配置、同数据集）
- 路径质量（paths 阶段产出）：路径命中率、平均跳数、起终点平均 CF 距离

## 5. 缺口与风险（后续版本）

- PGD 对抗优化、时间衰减权重（方案 §2.4）未实现
- 代理模型目前是"同架构、不同划分种子"（LightGCN）；跨架构代理（BPR-MF / NGCF）
  与"代理只在受害训练集子集上训练"留待后续，迁移性结论需以跨架构实验为准
- 语义距离融合（λ>0）、每跳 τ 阈值未启用——当前数据无物品文本/图像特征
- random_walk / 锚点桥接两种路径策略尚未实现（代码预留 `strategy` 字段）
- 隐蔽性评估（KL/MMD、检测 AUC）不在 v1 范围
- 共现图稀疏导致长路径缺失时回退直连，可能稀释 TPA 效果——大数据集上需复查
