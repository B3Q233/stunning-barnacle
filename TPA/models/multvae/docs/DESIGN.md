# MultVAE 设计说明

MultVAE 对用户多热交互向量做 L2 归一化，经编码器产生 `mu` 和 `logvar`，使用重参数化采样得到潜变量，再由解码器输出物品 logits。训练损失为多项式重构损失加 KL 散度，并通过 `kl_warmup` 逐步增加 KL 权重。
