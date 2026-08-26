# ItemAE 设计说明

## 核心算法

ItemAE 使用用户-物品交互向量作为输入，经线性编码器和 `tanh` 隐藏层得到用户表示，再由线性解码器重构原始交互向量。

## 训练与评估

训练使用 Adam 和二元交叉熵重构损失；推理时使用解码 logits 作为物品排序分数。用户历史保存在模型 buffer 中，用于 `predict_full_ranking`。

## 统一接口

构造函数为 `model_cls(config, num_users, num_items, edge_index=None)`。`edge_index` 是兼容项目统一模型接口的保留参数，不参与 ItemAE 计算。
