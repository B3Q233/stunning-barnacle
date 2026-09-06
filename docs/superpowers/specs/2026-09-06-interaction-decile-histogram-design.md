# 交互数十等分直方图（十分位分档）——设计文档

> 日期：2026-09-06
> 状态：待人工审阅

## 1. 目标

在 `TPA/visualization/item_freq/plot_item_freq.py` 增加可选输出：把物品按
**交互数取值**的 10%/20%/…/100% 分位切成 10 档（0–10%、10–20%、…、90–100%），
每档直方图高度 = 落在该取值区间内的去重物品数。

## 2. 口径

- 只统计出现过的物品（c>0），与 `count_dist` 口径一致（不引入大量 0 交互
  冷物品）。
- 边界：对全部 c>0 的交互数求 `np.percentile(values, q)`，q=10,20,…,100，
  得到 10 个取值边界（可并列）。
- 分档规则（左开右闭、去重）：每件物品归入“第一个 ≥ c 的边界”所对应的档；
  c ≤ p10 → 0–10%；p10 < c ≤ p20 → 10–20%；…；p90 < c ≤ p100 → 90–100%。
  因并列边界可能产生空档（某档 0 个物品），保留空档并在 CSV 标 0。
- 直方图 x 轴为 10 个档名，y 轴为该档去重物品数。

## 3. 产物与接口

- 新增 `build_decile_item_counts(counts) -> (labels, boundaries, item_counts)`：
  labels=[“0-10%”,…,“90-100%”]；boundaries=10 个分位边界；item_counts 为
  每档物品数。空 counts 抛 ValueError。
- 新增 `plot_decile_histogram(dataset, labels, item_counts, out_path, ...)`
  与组合图 `plot_decile_histogram_all(...)`（沿用顶会风格，线性柱状图）。
- CLI：复用现有 `--count-dist` 同族，新增 `--decile-dist`（默认关闭），
  输出：
  - `outputs/decile_dist_{dataset}.csv`（列：bucket,boundary,item_count）
  - `outputs/decile_dist_{dataset}.png`
  - 多数据集时 `outputs/decile_dist_all_datasets.png`（1xN）

## 4. 测试

- 扩展 `tests/test_item_freq.py`：已知 counts 验证边界/分档计数；并列边界
  空档补 0；空输入抛 ValueError。

## 5. 明确不做

- 不改现有 line/hist/count-dist 输出；不做累计交互量 10% 切片（口径 2）。
