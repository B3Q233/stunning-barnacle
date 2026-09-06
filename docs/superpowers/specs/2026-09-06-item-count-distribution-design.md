# 交互数分布图（x=交互数，y=物品数）——设计文档

> 日期：2026-09-06
> 状态：待人工审阅（brainstorming 已确认：新增可选输出，不改现有行为）

## 1. 目标

在 `TPA/visualization/item_freq/plot_item_freq.py` 增加一种可选输出：**交互数
分布图**——x 轴为物品的交互数（1..max，仅 c>0），y 轴为“拥有该交互数的物品
数量”，用于直接观察频次-频数（count-of-counts）分布。

## 2. 行为设计

- 新增 CLI 参数 `--count-dist`（默认关闭）：开启后，在现有 line/hist 输出
  基础上追加分布图与 CSV，不改变现有输出集。
- 单数据集输出：
  - `outputs/count_dist_{dataset}.png`（顶会风格曲线：双对数默认、Okabe-Ito
    配色、外置刻度、无网格、长尾自动断 0）；
  - `outputs/count_dist_{dataset}.csv`（表头 `interaction_count,item_count`）。
- 多数据集（len(panels)>1）另输出
  `outputs/count_dist_all_datasets.png`（1xN 组合图，共享 y 轴）。
- 数据口径：x 只包含交互数 > 0 的物品；y = Counter(counts.values()) 在
  [1, max] 上的频数（缺失交互数补 0，便于对齐坐标）。

## 3. 实现要点

- 新增 `build_count_distribution(counts: Counter) -> (np.ndarray, np.ndarray)`：
  返回 (x, y)，其中 x=arange(1, max_count+1)，y[i]=counts 中值为 i+1 的个数。
- 新增 `plot_count_distribution(dataset, x, y, out_path, split_label, ...)`
  与 `plot_count_distribution_all(panels, out_path, ...)`：复用
  `_apply_publication_style/_apply_scale/_plot_y`。
- `main()`：解析 `--count-dist`；在每个数据集统计完成后构造分布序列；
  写 CSV（与现有 item_freq csv 逻辑一致）并绘图；收集 panels 供组合图。
- 文档：更新 `visualization/item_freq/README.md`，说明参数与产物。

## 4. 测试

- 新增/扩展 `tests/test_item_freq.py`：
  - `build_count_distribution` 对已知 counts 返回正确 x/y；
  - 缺失交互数补 0、0 交互物品不进入 x；
  - 空 counts 抛 ValueError。

## 5. 明确不做

- 不改变现有 line/hist 图与默认输出行为；
- 不做分箱/对数分箱（已有 popularity histogram 承担）；
- 不改数据集路径与读取逻辑。

## 6. 风险

- 长尾交互数 max 很大时 x 长度 = max，绘图/CSV 仍可控（现有数据集 max
  交互数千级别）；无需额外抽样。
