# AdvInject 阶段一 Victim Models 设计规格

## 目标

在现有 TPA 模型规范下，复现 `graytowne/revisit_adv_rec` 的 victim model 能力，暂不实现 `advinject` attack。

## 范围

新增独立模型模块：`itemcf`、`itemae`、`ncf`、`multvae`、`cml`；复用并改进已有 `wmf`、`mf`、`lightgcn`。所有模型通过 `TPA/models/registry.py` 公共注册表暴露给攻击模块。

## 统一契约

- 数据输入：复用现有 `models/*/data/processed/{dataset}/meta.pkl`，包含 `num_users`、`num_items`、`train_pairs`、`test_pairs`、`user_items`。
- 构造：兼容 `model_cls(cfg, num_users, num_items, edge_index=None)`。
- 评分：提供 `get_user_embeddings()`、`get_item_embeddings()` 或等价 `predict_full_ranking()`。
- 训练：每个模型拥有自己的 `config.yaml`、`dataset.py`、`model.py`、`train.py`、`main.py` 和中英文明确的 `docs/USAGE.md`、`docs/DESIGN.md`。
- 评估：统一 clean/poisoned 可用的 Top-K 全量排序指标，并保证 `eval_step` 返回标量。
- 实验：使用现有 run_tag、config snapshot、checkpoint 和输出目录约定。
- 环境：使用仓库根 `.venv`，不创建新环境，不修改 requirements.txt，除非验证发现缺失且用户另行批准。

## 模型实现边界

- `itemcf`：基于训练交互的 item-item 相似度与用户历史聚合，不引入神经参数。
- `itemae`：item-based autoencoder，输入用户-物品交互矩阵，输出重构评分并暴露用户/物品评分接口。
- `ncf`：GMF/MLP 兼容的隐式反馈 BPR 训练模型，默认采用 MLP+GMF 融合。
- `multvae`：用户-物品向量上的 Mult-VAE，训练重构项与 KL 项，输出全量推荐分数。
- `cml`：Collaborative Metric Learning，用户/物品嵌入与 margin ranking loss，距离越小表示偏好越强。
- `wmf`：保留现有 ALS 闭式训练，新增稳定的可微 SGD 训练路径、动态用户边界检查和统一全量评分接口。

## 验收

- 每个模型先完成数据加载最小验证，再完成结构/评分验证，再完成单 batch 或单 epoch 训练验证。
- 新模型注册表测试覆盖类、数据类和配置路径。
- 对动态伪用户 ID 执行越界边界测试。
- 运行相关 unittest，再运行全量 unittest。
- 不实现 `TPA/attacks/advinject/`，阶段二另行开始。