# PyTorch API 常见陷阱：数学符号 → 代码对照表

复现论文时，数学公式和 PyTorch API 之间最常见的语义不匹配。在步骤⑤「公式→API 对照验证」时逐项检查。

## 1. 范数与正则化

| 论文写法 | 数学含义 | ❌ 错误 PyTorch | ✅ 正确 PyTorch | 说明 |
|---------|---------|----------------|----------------|------|
| `‖W‖²` 或 `‖W‖₂²` | Σw² (平方和) | `W.norm(p=2)` | `W.norm(p=2).pow(2)` | `.norm()` 返回的是 `√Σw²`，不是 `Σw²` |
| `‖W‖₁` | Σ|w| (绝对值和) | `W.norm(p=1)` ✅ | — | L1 范数没有平方问题，`norm(p=1)` = Σ|w| |
| `λ·‖W‖²` | L2 正则项 | `l2_reg * W.norm(p=2)` | `l2_reg * W.norm(p=2).pow(2)` | 本次复现踩的坑！正则项差 10-100 倍 |
| `‖W‖_F` | Frobenius 范数 | | `W.norm(p='fro')` | 对矩阵等价于 `norm(p=2)` |

## 2. 损失函数

| 论文写法 | 数学含义 | ❌ 容易写错 | ✅ 正确写法 | 说明 |
|---------|---------|-----------|-----------|------|
| `-ln σ(x)` | BPR loss | | `-F.logsigmoid(x)` | `F.logsigmoid` 数值比 `ln(sigmoid(x))` 更稳定 |
| `log(1+exp(x))` | softplus | | `F.softplus(x)` | 等价于 `-F.logsigmoid(-x)` |
| `σ(x)` | sigmoid | | `torch.sigmoid(x)` | `F.sigmoid` 已废弃 |
| Cross-entropy | `-Σ y·log(ŷ)` | 手动加 softmax 后再用 NLLLoss | `F.cross_entropy(logits, labels)` | PyTorch 内置了 log_softmax，不要重复 softmax |

## 3. 归一化层

| 操作 | 论文默认参数 | PyTorch 默认参数 | 是否一致 |
|------|------------|-----------------|---------|
| BatchNorm momentum | 写作 0.99 (TF 语义) | `momentum=0.1` | ⚠️ **语义相反**：TF `momentum=0.99` = PyTorch `momentum=0.01` |
| BatchNorm eps | 1e-3 (TF) | 1e-5 | 🔴 差 100 倍 |
| LayerNorm eps | 1e-3 (TF) | 1e-5 | 🔴 差 100 倍 |

## 4. Dropout

| 操作 | 关键点 | 正确用法 |
|------|-------|---------|
| `nn.Dropout(p)` | `p` 是丢弃概率 | 训练时自动启用，`model.eval()` 时自动关闭 |
| `F.dropout(x, p, training)` | 需要手动传 `training` 标志 | `F.dropout(x, p=0.5, training=self.training)` |
| Graph dropout | 稀疏张量的边丢弃 | 需手动实现 mask，保留边需 `/keep_prob` 缩放 |

## 5. 初始化

| 论文指定 | ❌ 容易误用 | ✅ 正确写法 |
|---------|-----------|-----------|
| Normal(0, 0.1) | `nn.init.xavier_uniform_()` | `nn.init.normal_(weight, std=0.1)` |
| Xavier uniform | `nn.init.normal_()` | `nn.init.xavier_uniform_(weight)` |
| Kaiming/He uniform | `nn.init.xavier_uniform_()` | `nn.init.kaiming_uniform_(weight, a=0)` (ReLU) / `a=1` (LeakyReLU) |
| 常数初始化 | `nn.init.zeros_()` + 忘了 bias | `nn.init.constant_(weight, val)` |

## 6. 优化器

| 参数 | PyTorch 默认 | 常见论文值 | 是否需改 |
|------|------------|----------|---------|
| Adam eps | 1e-8 | 1e-7 (TF 默认) | 🟡 可能影响小数据集 |
| Adam betas | (0.9, 0.999) | 论文通常不改 | ✅ 一致 |
| Adam weight_decay | 0.0 (旧版) | 需要显式设置 | 🔴 容易忘记设置 |

## 检查清单

在执行步骤⑤「公式→API 对照验证」时，按以下顺序检查：

- [ ] 所有 `.norm()` 调用确认是否需要 `.pow(2)`
- [ ] 损失函数中的 `log`/`exp`/`sigmoid` 项已确认数值稳定性写法
- [ ] BatchNorm / LayerNorm 的 eps 和 momentum 是否与论文/参考实现一致
- [ ] Dropout 的 `p` 值与论文一致，且 train/eval 模式切换正确
- [ ] 初始化分布和参数与理解文档 2.7 一致
- [ ] 优化器的 eps、betas、weight_decay 与论文一致
