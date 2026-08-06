# 攻击设计文档模板（attack-imp-direct-poison）

> 本文件是生成攻击项目时的**设计文档模板**。TODO 处需要根据目标论文填写：
> 攻击定义、参数来源等级（【论文明确写出】/【AI推断补全】/【用户指定】）、
> 以及论文依据（公式/表格/章节位置），禁止把 AI 推断标成论文原文。

## 0. 提取信息

- 攻击定义来源：TODO（论文标题 / 作者 / 年份，首次提出该攻击的文献）
- 关联文献：TODO（如 IndirectAD / 其他 shilling attack 综述）
- 受害模型：由 `config.model.name` 指定（注册表见 `registry.py`）
- 本模块：`attacks/attack_imp_direct_poison`（与模型代码解耦）

## 1. 攻击定义

TODO：写清楚攻击画像的形式化定义，例如：

```
假用户画像 = { 目标物品 } ∪ { K 个 filler 物品 }
```

并注明 filler 的选取语义（流行/随机/平均/路径等）与论文依据。

## 2. 参数与来源等级

| 参数 | 默认值 | 来源等级 | 说明 |
|------|--------|----------|------|
| 假用户数 | TODO | TODO | TODO |
| filler_size K | TODO | TODO | TODO |
| 目标物品 | 手动指定 | 【用户指定】 | `strategy: specified` 填 ID |
| 流行度划分 | 前 20% 流行 / 其余普通 / 零频次冷门 | 【用户指定】 | 基于干净模型每用户 Top-K 推荐频次 |
| warm-start | 开启 | 【用户指定】 | 用干净模型参数初始化中毒模型 |

## 3. 模块设计（三阶段，no-subgoal）

| 模式 | 职责 | 是否新建模型 |
|------|------|--------------|
| `classify`（classify.py） | 加载干净模型 → 全量评分 → 每用户 Top-K → 推荐频次分类并缓存 | 否（只读 checkpoint） |
| `data`（generate.py） | 选目标（默认手动指定）→ 构造假画像 → 注入训练集 | 否（纯数据层） |
| `model`（fit.py） | 按 `model.name` 新建受害模型（可选 warm-start）→ 训练 → 对比评估 | 是（新实例） |

数据流：

```
clean 模型（checkpoint）──classify──▶ rec_freq.json（流行/普通/冷门）
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

- TODO：论文未给出而模板做了假设的参数（假用户比例、filler 大小、目标选择策略等）
- TODO：与论文实现有差异的细节（如 warm-start 假设攻击者能拿到干净模型）
