# ItemCF 使用说明

## 功能

ItemCF 根据训练交互构造 item-item 余弦相似度矩阵，并用用户历史交互累积得到候选物品分数。该模型没有可训练参数，拟合过程在 `fit` 中完成。

## 数据

默认读取：

```text
models/wmf/data/processed/ml100k/meta.pkl
```

也可以在配置的 `data.processed_data_path` 中指定 `meta.pkl` 文件或其所在目录。

## 运行

```powershell
cd TPA
<repo>\.venv\Scripts\python.exe models/itemcf/main.py --config models/itemcf/config.yaml
```

实验结果保存到 `models/itemcf/outputs/{dataset}/{run_tag}/`，包括配置快照、指标、训练历史和最后模型状态。
