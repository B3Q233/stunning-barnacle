# MultVAE 使用说明

## 功能

MultVAE 将用户多热交互向量归一化后编码为高斯潜变量，再解码为所有物品的 logits。损失由多项式重构损失和 KL 散度组成，并支持 KL warm-up。

## 模型接口

```python
model_cls(config, num_users, num_items, edge_index=None)
```

`edge_index` 为统一模型契约保留的可选参数，MultVAE 不使用图边。

## 运行

```powershell
cd G:\Idea\TPA
G:\Idea\.venv\Scripts\python.exe models/multvae/main.py --config models/multvae/config.yaml
```

默认数据来自 `models/wmf/data/processed/ml100k/meta.pkl`，产物写入 `models/multvae/outputs/{dataset}/{run_tag}/`。
