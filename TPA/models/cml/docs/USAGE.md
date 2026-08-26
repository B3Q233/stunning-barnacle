# CML 使用说明

## 功能

CML 在用户和物品 embedding 空间中最小化正样本距离、增大负样本距离，训练使用带 margin 的度量学习损失。

## 模型接口

```python
model_cls(config, num_users, num_items, edge_index=None)
```

`edge_index` 为统一模型契约保留的可选参数，CML 不使用图结构。

## 运行

```powershell
cd G:\Idea\TPA
G:\Idea\.venv\Scripts\python.exe models/cml/main.py --config models/cml/config.yaml
```

默认数据来自 `models/wmf/data/processed/ml100k/meta.pkl`，产物写入 `models/cml/outputs/{dataset}/{run_tag}/`。
