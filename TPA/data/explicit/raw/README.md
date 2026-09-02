# explicit/raw：显式评分原始数据

此目录存放显式评分（rating）数据集，如 MovieLens（1–5 星）、FilmTrust 等。

约定：

- 文件格式示例：`user_id item_id rating`（每行一条评分，tab 或空格分隔）；
- 0 = 未评分；评分上界由各数据集/配置的 `rating_scale` 声明；
- 接入新数据集时：① 文件放入 `{dataset}/`；② 在
  `TPA/training/paths.py` 的 `DATASET_INTERACTION` 登记为 `explicit`；
  ③ 按根 `AGENTS.md` §6.6 补齐显式评分配置与模型需求。
