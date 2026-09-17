# NCF 使用说明

## 功能

NCF 使用用户和物品 embedding，并通过 MLP 对用户-物品对进行打分。训练采用 BPR pairwise ranking loss，负样本由训练辅助模块采样。

## 模型接口

```python
model_cls(config, num_users, num_items, edge_index=None)
```

`edge_index` 仅用于统一模型注册和攻击调用契约，NCF 不依赖图结构。

## 运行

```powershell
cd TPA
<repo>\.venv\Scripts\python.exe models/ncf/main.py --config models/ncf/config.yaml
```

默认数据来自 `models/wmf/data/processed/ml100k/meta.pkl`，实验产物写入 `models/ncf/outputs/{dataset}/{run_tag}/`。
