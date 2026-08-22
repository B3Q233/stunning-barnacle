# Random（随机）攻击设计文档

> 本模块由 attack-imp-direct-poison 攻击模板（no-subgoal）生成，已按
> Random（随机）攻击语义填写。来源等级：【论文明确写出】/【AI推断补全】/【用户指定】。

## 0. 提取信息

- 攻击定义来源：Lam & Riedl, *Shilling Recommender Systems for Fun and Profit*,
  WWW 2004——随机攻击是经典 shilling attack 基线之一
- 关联文献：IndirectAD (arXiv:2511.05845) 相关工作段落将 bandwagon / random 等
  列为经典托攻击成员
- 受害模型：由 `config.model.name` 指定（注册表见 `registry.py`）
- 本模块：`attacks/random`（与模型代码解耦）

## 1. 攻击定义

【文献惯例】Random（随机）攻击属于经典托攻击：攻击者构造一批假用户画像，
每个画像同时包含 **若干随机物品（filler，从全量物品中均匀采样，不利用流行度或
共现信息）** 和 **目标物品（push 对象）**，从而提升目标物品在 Top-K 推荐中的曝光。

```
假用户画像 = { 目标物品 } ∪ { K 个均匀随机采样的物品 }
```

与 bandwagon（流行 filler）相比，random 攻击不带任何语义信息，攻击效果通常更弱，
常作为"无信息攻击"的下界基线参与对比。

## 2. 参数与来源等级

| 参数 | 默认值 | 来源等级 | 说明 |
|------|--------|----------|------|
| 假用户数 | 18（≈3% 真实用户） | 【用户指定】 | `ratio: 0.03`，可调 |
| filler_size K | 20 | 【AI推断补全】 | 从全量物品均匀随机采样 K 个 |
| 目标物品 | 手动指定 | 【用户指定】 | `strategy: specified` 填 ID |
| 流行度划分 | 前 5% 流行 / 5~40% 普通 / 其余冷门 | 【用户指定】 | 基于训练集每物品交互次数 |
| warm-start | 开启 | 【用户指定】 | 用干净模型参数初始化中毒模型 |

## 3. 模块设计（三阶段，no-subgoal）

| 模式 | 职责 | 是否新建模型 |
|------|------|--------------|
| `classify`（classify.py） | 统计训练集交互数 → 按交互数排名划分流行/普通/冷门并缓存 | 否（纯数据层，零模型依赖） |
| `data`（generate.py） | 选目标（默认手动指定）→ 构造假画像 → 注入训练集 | 否（纯数据层） |
| `model`（fit.py） | 按 `model.name` 新建受害模型（可选 warm-start）→ 训练 → 对比评估 | 是（新实例） |

数据流：

```
训练集交互数 ──classify──▶ rec_freq.json（流行/普通/冷门）
                                          │
clean meta.pkl ──generate（指定目标 + filler）──▶ poisoned meta.pkl
                                                          │
                                                          ▼ fit（model.name）
                                                      新受害模型（BPR 训练）
                                                          ▼
                                            clean 模型 ──对比评估──▶ attack_comparison.md
```

## 4. 评估协议

- 模型效用：all-ranking recall@K / ndcg@K，用于检查投毒代价；
  不需要时配置 `evaluation.report_model_utility: false`
- 攻击效果：对"训练集未交互过目标物品"的用户，报告每个目标物品的
  **HR@K（命中率）** 与 **NDCG@K（单目标，IDCG=1）**，外加命中用户数和平均排名
- 对比口径：clean 与 poisoned 统一用干净训练集过滤，保证公平

## 5. 缺口与风险（需人工确认）

- 【AI推断补全】filler_size 与假用户比例的最优值：沿用模板默认 20 / 3%，可按实验调整
- 【文献惯例】filler 是否排除目标物品、是否允许重复采样：本实现每个假用户
  从"全量物品去掉目标"中无放回采样 K 个
- warm-start 假设攻击者能拿到干净模型参数（增量注入假设）；如不适用可设
  `warm_start.enabled: false` 从头训练
