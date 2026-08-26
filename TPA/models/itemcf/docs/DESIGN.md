# ItemCF 设计说明

## 核心算法

1. 根据训练交互构造用户-物品矩阵 `R`。
2. 计算 `R.T @ R` 得到 item-item 共现矩阵。
3. 按物品交互次数进行余弦归一化，并清零对角线。
4. 用用户历史向量乘以相似度矩阵，得到全量物品分数。

## 工程契约

模型实现 `TrainableModel` 所需的 `train_step`、`eval_step`、`predict_full_ranking`、embedding 获取接口。由于 ItemCF 是非参数模型，实际拟合由 `fit` 完成。
