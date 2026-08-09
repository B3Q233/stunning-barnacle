# MF（矩阵分解）模型实现文档

> 论文依据：Li et al., *Data Poisoning Attacks on Factorization-Based Collaborative
> Filtering* (NIPS 2016)，Eq. (2) 的矩阵分解形式 M ≈ U V^T；训练目标采用本项目
> LightGCN 同款 BPR 成对损失（隐式反馈），使 MF 与 LightGCN 可被攻击流程统一调用。

## ① 数据处理

- 输入：`G:\Idea\TPA\data\raw\{dataset}\train.txt / test.txt`
  - ml100k 为单对行格式 `user item`；gowalla/yelp2018/amazon-book 为 NGCF
    多物品行格式 `user item1 item2 ...`，解析逻辑兼容两者（遍历 `parts[1:]`）。
- 输出：`models/mf/data/processed/{dataset}/meta.pkl`
  - `num_users` / `num_items`（物品 ID 取 max+1，假设从 0 连续）
  - `train_pairs` / `test_pairs`：`(user, item)` 正样本对
  - `user_items`：`{user: set(items)}`（负采样过滤用）
- 验证结果（ml100k）：用户 608 / 物品 6298 / 训练交互 38614 / 测试交互 9965，
  与 LightGCN 预处理产物完全一致（数据源相同，格式相同）。

## ② 数据导入

- `MFDataset`：BPR 随机负采样，`__getitem__` 返回
  `(user, pos_item, neg_items[neg_ratio])`，batch 形状
  `(B,) / (B,) / (B, neg_ratio)`，dtype `int64`。
- `MFDataLoader`：实现 `DatasetProtocol` 五个方法；95/5 随机划分训练/验证
  （seed=42，与 LightGCN 一致）。
- 验证：真实 batch 形状断言通过（`256 / 256 / [256,1]`）。

## ③ 模型结构

| 层/参数 | 类型 | Shape | 初始化 |
|--------|------|-------|--------|
| embedding | nn.Embedding | (M+N, 64) | N(0, 0.1)（config `init_method: normal`）|

- 前向：`y_hat = e_u^T e_i`，无图卷积（与 LightGCN 的区别仅在无 LGC 传播）。
- 验证：embedding 统计 mean≈0、std≈0.1 与 N(0,0.1) 一致；
  `forward` 输出形状 `(256,)` 通过。

## ④ 模型评估

- 协议与 LightGCN 一致：all-ranking、过滤训练集已交互物品、Top-K 评估。
- 指标：`recall@20` / `ndcg@20`（`evaluation/metrics.py`）。
- 验证：训练过程每 epoch 全量评估写入 `outputs/eval_log.csv`。

## ⑤ 模型训练

- 损失：BPR + L2 正则（仅约束第 0 层嵌入，按 batch 归一化，与 LightGCN 一致）。
- 优化器：Adam，lr=1e-3，weight_decay=1e-4，epochs=100，batch=256。
- 验证：1 batch 训练后 loss 有限、参数确实更新；
  完整 100 epoch 训练 loss 单调下降（0.695 → 0.012）。

## ⑥ 结果展示

- 训练完成指标（ml100k，100 epoch）：recall@20 ≈ 0.214、ndcg@20 ≈ 0.198
  （best epoch 71 / best_epoch0071.pt；latest.pt 为 100 epoch 终点）。
- 产物：`outputs/history.json`（逐 epoch loss）、`outputs/eval_log.csv`（逐 epoch 指标）、
  `outputs/checkpoints/`（best + latest）。

## 关键决策记录

- 本实现未单独做 RMSE/显式评分评估（论文用显式评分），因本项目攻击框架（bandwagon/
  tpa）统一采用隐式反馈 BPR + HR@K/NDCG@K 评估协议；MF 作为受害模型与 LightGCN
  共享同一协议，保证 PGD 攻击对比公平。标注 [ai]。
- MF 与 LightGCN 共享 `embedding` 属性与
  `get_user_embeddings()/get_item_embeddings()` 接口，攻击流程（registry/
  warm-start/评估）无需区分模型类型。
