# 物品交互频次统计与曲线绘制

统计每个物品的交互数（物品在数据集中所有行出现的次数），并按交互数从大到小
**重映射**后绘制**顶会风格曲线**（默认双对数坐标）：交互数最多的物品新 id 为 1，之后依次为
2、3、…；x 轴取重映射后的新 id（1..N），y 轴取对应交互数。交互数相同的物品
按原始 id 升序排列，结果确定可复现；从未交互过的物品排在尾部（交互数为 0）。

## 数据格式

每行 = `用户id 物品id1 物品id2 ...`，例如 `1 3 4 5` 表示用户 1 与物品 3/4/5
各交互一次。物品 id 从 0 开始编号，因此 x 轴取 `物品 id + 1`。

## 数据集

默认统计 LightGCN 三个数据集（源文件：`TPA/data/raw/{dataset}/train.txt`）：

| 数据集 | 用户数 | 物品数 | train 交互数 |
| --- | ---: | ---: | ---: |
| Gowalla | 29,858 | 40,981 | 810,128 |
| Yelp2018 | 31,668 | 38,048 | 1,237,259 |
| Amazon-Book | 52,643 | 91,599 | 2,380,730 |

## 绘图要求（已实现）

- 默认按交互数降序重映射：x 轴 = 新物品 id（交互最多者 = 1），y 轴 = 对应交互数；
  `--sort id` 可恢复为按原始物品 id+1 为 x 轴；
- 默认 `--style line`：顶会（conference-paper）风格纯曲线——只保留左/下坐标轴、
  外置细刻度、无网格、无标记点、Okabe-Ito 色觉安全配色；
- 默认 `--xscale log --yscale log`：长尾分布头部不会被压扁折叠在 y 轴上；
  曲线四周留边距（xlim 左界 < 1），起点不与坐标轴贴合；交互数为 0 的物品
  在对数坐标下自动断开曲线；
- x 轴刻度固定为 10 的幂（1/10/100/1000/10000），三个面板刻度完全一致；
  需要线性坐标时用 `--xscale linear --yscale linear`；
- 三个数据集同时输出一张 1x3 组合图（面板标签 (a)(b)(c)，共享 y 轴）；
  `--style hist` 可选直方图（固定线性坐标，柱宽由 `--bar-width` 控制，
  默认 1.0 相邻无缝）；
- 配色采用 Okabe-Ito 色觉安全配色（Gowalla 蓝 / Yelp2018 橙 / Amazon-Book 绿）。

## 流行度直方图（Popularity Histogram）

另输出一张推荐视角的直方图：x 轴为**交互次数**（对数轴，分箱
[1] / [2] / [3–4] / [5–8] / [9–16] / [17–32] / ...，即 2 的幂边界），
y 轴为**物品占比 Proportion of items (%)**（各箱物品数 ÷ 总物品数），三个数据集
可直接比较长尾程度。

柱子颜色**来自排名划分，而不是交互次数阈值**：先按交互数降序给全部物品排名，
Top 5% 为 Hot、5%~40% 为 Medium-hot、40%~100% 为 Tail；每个交互数箱内
**并排**三根柱（非堆叠、非重叠），分别表示该箱内属于 Tail / Medium-hot / Hot
的物品占比，可直接比较三者高度。因此每个数据集的类别边界不同
（如 Gowalla Hot≥53、Yelp2018 Hot≥108、Amazon-Book Hot≥77），颜色表示
“物品属于哪一类”。图例顺序为 Tail → Medium-hot → Hot。
注意：这三个数据集经过 LightGCN 预处理，已过滤极低频物品，峰值通常落在
[9–16] 箱。

## 用法

```powershell
# 三个数据集（默认 train.txt）
G:\Idea\.venv\Scripts\python.exe TPA\visualization\item_freq\plot_item_freq.py

# 只画某个数据集（默认 --sort count：按交互数降序重映射）
G:\Idea\.venv\Scripts\python.exe TPA\visualization\item_freq\plot_item_freq.py --datasets gowalla

# 按原始物品 id+1 为 x 轴
G:\Idea\.venv\Scripts\python.exe TPA\visualization\item_freq\plot_item_freq.py --sort id

# 双线性坐标（不使用对数轴）
G:\Idea\.venv\Scripts\python.exe TPA\visualization\item_freq\plot_item_freq.py --xscale linear --yscale linear

# 合并 train + test 统计
G:\Idea\.venv\Scripts\python.exe TPA\visualization\item_freq\plot_item_freq.py --splits train test
```

## 输出

代码目录：`TPA/visualization/item_freq/`

- `plot_item_freq.py`：统计与绘图脚本
- `outputs/item_freq_{dataset}.png`：单数据集顶会风格曲线图（1560×1020，dpi=300）
- `outputs/item_freq_all_datasets.png`：三数据集 1x3 组合图（3750×1140，dpi=300）
- `outputs/popularity_hist_{dataset}.png`：单数据集流行度直方图
  （x=交互次数，y=Proportion of items (%)，按排名类别并排着色 + 图例）
- `outputs/popularity_hist_all_datasets.png`：三数据集 1x3 组合直方图
- `outputs/item_freq_{dataset}.csv`：统计明细（rank 模式：
  `new_item_id,original_item_id,interaction_count`；`--sort id` 模式：
  `item_id_plus_1,interaction_count`）

说明：`outputs/` 与 `*.png` 按仓库 `.gitignore` 规则不入库，脚本与 README 入库。

## 测试

```powershell
cd TPA
G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_item_freq -v
```
