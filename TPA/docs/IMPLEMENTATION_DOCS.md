# LightGCN 实现文档

> 复现论文: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)
> 理解文档: g:/Idea/papers/lightgcn/2002.02126v4_understanding.md
> 项目路径: g:/Idea/TPA

---

## 步骤① 数据处理

### 方法
NGCF 格式 (user_id item1 item2 ...) → BPR 训练对 (user, item)。ID 已 remap，无需额外编码。

### 输入 → 输出
```
输入: data/raw/{dataset}/train.txt, test.txt
      data/raw/{dataset}/user_list.txt, item_list.txt

输出: models/lightgcn/data/processed/{dataset}/
      ├── train_pairs.txt    (user item 对，每行一对)
      ├── test_pairs.txt
      └── meta.pkl           (num_users, num_items, user_items 字典)
```

### 验证结果
| 数据集 | 用户 | 物品 | 训练交互 | 测试交互 | 总交互 | 论文值 | 偏差 |
|--------|------|------|---------|---------|--------|--------|------|
| Gowalla | 29,858 | 40,981 | 810,128 | 217,242 | 1,027,370 | 1,027,370 | 0% |
| Yelp2018 | 31,668 | 38,048 | 1,237,259 | 324,147 | 1,561,406 | 1,561,406 | 0% |
| Amazon-Book | 52,643 | 91,599 | 2,380,730 | 603,378 | 2,984,108 | 2,984,108 | 0% |

### 关键决策
- 数据格式: NGCF 格式，每行 `user_id item1 item2 ...`，ID 已从 0 开始连续 remap
- 无归一化/特征工程，LightGCN 仅使用 ID 嵌入

---

## 步骤② 数据导入

### 方法
`LightGCNDataset(Dataset)` + `LightGCNDataLoader`（实现 DatasetProtocol 五个方法）。

训练集 95/5 随机划分为 train/val（seed=42），BPR 随机负采样（每正样本采一个负样本）。

### Batch 格式
| 模式 | 字段 | Shape | dtype |
|------|------|-------|-------|
| train | (users, pos_items, neg_items) | (B,), (B,), (B,) | int64 |
| val | (users, pos_items, neg_items) | (B,), (B,), (B,) | int64 |
| test | (users, pos_items) | (B,), (B,) | int64 |

### 验证结果
```
Gowalla train batch: (1024,), (1024,), (1024,) — shape 通过
init_params: num_users=29858, num_items=40981 — 与预处理一致
```

### 配置键名常量
```python
KEY_NUM_USERS = "num_users"
KEY_NUM_ITEMS = "num_items"
KEY_DATASET = "dataset"
```

---

## 步骤③ 模型结构

### 逐层结构
| 层 | 类型 | 输入 shape | 输出 shape | 参数量 | 初始化 |
|----|------|-----------|-----------|--------|--------|
| Embedding | nn.Embedding | (M+N) IDs | (M+N, 64) | (M+N)×64 | Xavier uniform |
| LGC × K | 稀疏矩阵乘法 | (M+N, 64) | (M+N, 64) | 0 | - |
| 层组合 | mean(stack) | (K+1, M+N, 64) | (M+N, 64) | 0 | - |
| 预测 | 内积 | (B, 64)×(B, 64) | (B,) | 0 | - |

总参数量 = (M+N) × 64，与标准 MF 完全相同。

### 邻接矩阵
- 对称归一化: A_hat = D^(-1/2) A D^(-1/2)
- 格式: PyTorch sparse COO
- Gowalla: 70839×70839, 1,539,242 非零元

### 验证结果
```
forward (5,5): shape (5,) — 通过
Xavier init: std=0.005311 (范围 0.0005~0.02) — 通过
train_step ×3: loss 0.706→0.702→0.699 持续下降 — 通过
Embedding 在 CUDA 上，A_hat 在 CUDA 上 — 设备一致
```

---

## 步骤④ 模型评估

### 评估协议
- All-ranking: 所有未交互物品为候选
- 过滤训练集已交互物品
- 逐用户计算 recall@20 和 ndcg@20

### 公式
```
recall@K = |TopK ∩ test_pos| / |test_pos|
NDCG@K = DCG / IDCG
DCG = Σ_{rank=1..K} (hit ? 1/log2(rank+1) : 0)
```

### 验证结果
```
手工用例 (3 users, 5 items):
  recall@2=1.0000, ndcg@2=0.8770 — 与手算一致
```

---

## 步骤⑤ 模型训练

### 损失函数
```
L_BPR = -Σ ln σ(ŷ_ui - ŷ_uj) + λ‖E^(0)‖²
```
- BPR: `-F.logsigmoid(pos - neg).mean()`
- L2: `E.norm(p=2).pow(2) * weight_decay`（平方 Frobenius 范数，非未平方）
- λ = 1e-4（Gowalla/Amazon）, 1e-3（Yelp2018）

### 优化器
- Adam, lr=0.001
- Batch size: 1024 (Amazon-Book: 2048)

### 验证结果
```
1 epoch (Gowalla): 752 batches, avg loss=0.6932, 无 NaN
eval_step: {"val_loss": 0.6931}, 全部 float — 标量契约通过
```

---

## 步骤⑥ 结果展示

### 论文报告值 (Table 4, 3-layer LightGCN)
| 数据集 | recall@20 | ndcg@20 |
|--------|-----------|---------|
| Gowalla | 0.1823 | 0.1555 |
| Yelp2018 | 0.0639 | 0.0525 |
| Amazon-Book | 0.0410 | 0.0318 |

### 复现结果
| 数据集 | recall@20 | ndcg@20 | 偏差 | 状态 |
|--------|-----------|---------|------|------|

待正式训练后填写。

### 定期评估记录
每 10 epoch 记录一次 recall@20 / ndcg@20，格式：

```
epoch  recall@20  ndcg@20  val_loss
10    0.xxxx    0.xxxx   0.xxxx
20    0.xxxx    0.xxxx   0.xxxx
...
```
