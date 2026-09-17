# ItemAE 使用说明

## 功能

ItemAE 将每个用户的 item 交互向量输入自编码器，通过重构交互向量学习用户表示和物品表示。训练损失使用带 logits 的二元交叉熵。

## 模型接口

`ItemAEModel` 遵循项目统一构造契约：

```python
model_cls(config, num_users, num_items, edge_index=None)
```

其中 `edge_index` 为兼容图模型和攻击模板而保留的可选参数，ItemAE 本身不使用它。

## 数据与运行

默认读取 `models/wmf/data/processed/ml100k/meta.pkl`，也可以通过 `data.processed_data_path` 指定数据位置。

```powershell
cd TPA
<repo>\.venv\Scripts\python.exe models/itemae/main.py --config models/itemae/config.yaml
```

产物保存到 `models/itemae/outputs/{dataset}/{run_tag}/`。
