# Bandwagon（从众）攻击设计文档

> 对应 paper-understanding 工作流的「理解文档」：本基线在项目中的定义、参数来源与评估协议。
> 项目：TPA（传递式路径投毒攻击）—— Bandwagon 是方案 §6.3 中列出的经典托攻击基线之一。
> 使用方法见同目录 `USAGE.md`。

## 0. 提取信息

- 攻击定义来源：Mobasher et al., *Toward trustworthy recommender systems* (2007)，首次系统化提出 bandwagon attack
- 关联文献：IndirectAD (arXiv:2511.05845) 相关工作段落明确列出 bandwagon 为经典 shilling attack 成员（提取文档第 364 行）
- 受害模型：LightGCN（已实现，`TPA/models/lightgcn`）；模型由配置
  `model.name` 指定，注册表见 `attacks/bandwagon/registry.py`
- 本模块：`TPA/attacks/bandwagon`（与模型代码解耦）

## 1. 攻击定义

【论文明确写出】Bandwagon（从众）攻击属于经典托攻击（shilling attack）：攻击者构造一批假用户画像，每个画像同时包含
**若干高流行度物品（filler，利用"从众"心理使其画像看起来正常）** 和 **目标物品（push 对象）**，
从而提升目标物品在 Top-K 推荐中的曝光。

隐式反馈（交互）场景下的具体形式（本项目采用的实例化）：

```
假用户画像 = { 目标物品 } ∪ { K 个流行物品 }
```

其中"流行物品"按**模型推荐频次**划分（不是训练集交互次数）：
加载干净模型 → 全量评分 → 每用户取 Top-K 推荐（过滤已交互）→ 统计物品出现次数，
出现次数最高的前 20% 为流行物品，其余有出现次数的为普通物品，出现次数为 0 的为冷门物品。

## 2. 参数与来源等级

| 参数 | 默认值 | 来源等级 | 说明 |
|------|--------|----------|------|
| 假用户数 | 6（≈1% 真实用户） | 【AI推断补全】 | 经典托攻击文献常用 1%~5%；ml100k 用户少，取 1% 便于观测 |
| filler_size K | 20 | 【AI推断补全】 | 从模型推荐频次 Top 20% 的流行物品池随机采样 K 个 |
| 目标物品 | 手动指定（默认 [90,110,146]） | 【用户指定】 | `strategy: specified` 填 ID；也可 `category: popular/ordinary/cold` 按分类挑 |
| 流行度划分 | 前 20% 流行 / 其余普通 / 零频次冷门 | 【用户指定】 | 基于干净模型每用户 Top-K 推荐的物品出现次数 |
| 假用户分配 | 均分给各目标 | 【AI推断补全】 | 每个假用户只 push 一个目标，与经典攻击一致 |
| warm-start | 开启 | 【用户指定】 | 用干净模型参数初始化中毒模型，再继续训练 |

## 3. 模块设计（模式分离）

| 模式 | 职责 | 是否新建模型 |
|------|------|--------------|
| `classify`（classify.py） | 加载干净模型 → 全量评分 → 每用户 Top-K → 统计推荐频次 → 划分流行/普通/冷门并缓存 | 否（只读 checkpoint） |
| `data`（generate.py） | 读分类缓存 → 选目标（默认手动指定）→ 构造假画像 → 注入训练集 | 否（纯数据层，零模型依赖） |
| `model`（fit.py） | 按 `model.name` 新建受害模型实例（可选 warm-start）→ 训练 → 对比评估 | 是（新实例，干净模型不受影响） |

数据流：

```
clean 模型（checkpoint）──classify──▶ rec_freq.json（流行/普通/冷门）
                                          │
clean meta.pkl ──generate（指定目标 + 流行 filler）──▶ poisoned meta.pkl
                                                          │
                                                          ▼ fit（model.name）
                                                      新受害模型（BPR 训练）
                                                          ▼
                                            clean 模型 ──对比评估──▶ bandwagon_comparison.md
```

## 4. 评估协议

- 模型效用：all-ranking recall@K / ndcg@K（与 LightGCN 一致，过滤训练集已交互物品）
  用于检查投毒代价——中毒模型推荐质量不应显著下降；如不需要可在配置中设
  `evaluation.report_model_utility: false` 关闭
- 攻击效果：对"训练集未交互过目标物品"的用户，报告每个目标物品的
  **HR@K（命中率）** 与 **NDCG@K（单目标，IDCG=1）**，外加命中用户数和平均排名
- 对比口径：clean 与 poisoned 统一用干净训练集过滤，保证公平

## 5. 缺口与风险（需人工确认）

- 【论文未提及】filler_size 与假用户比例的最优值：本实现用默认 20 / 1%，可按实验需要调整
- 【论文未提及】目标物品选择策略：默认 `specified` 由用户自行指定；
  `category` 可分别从流行/普通/冷门三类中挑；`coldest`（训练集最冷）仅作对照
- 【AI推断补全】流行/普通/冷门的三档划分（前 20% / 其余 / 零频次）来自用户确认的攻击流程，
  非论文原文；`popular_ratio` 可在配置中调整
- warm-start 会继承干净模型对目标物品已有的（微弱）偏好，属于攻击者"增量注入"假设；
  如需从头训练中毒模型，设置 `warm_start.enabled: false` 即可
