"""统计每个物品的交互数，按交互数降序重映射后绘制顶会风格曲线。

数据格式（与 ``TPA/data/raw/{dataset}/train.txt`` 一致）：
    每行 = "用户id 物品id1 物品id2 ..."
例如 ``1 3 4 5`` 表示用户 1 与物品 3/4/5 各交互一次。

对每件物品，交互数 = 该物品在所有行中出现的总次数。
默认（``--sort count``）按交互数从大到小排序并重映射：交互数最多的物品
新 id 为 1，之后依次为 2、3、...；x 轴取重映射后的新 id（1..N），
y 轴取对应交互数。交互数相同的物品按原始 id 升序排列，保证结果确定可复现；
从未交互过的物品排在尾部（交互数为 0）。也可用 ``--sort id`` 恢复按原始
物品 id+1 为 x 轴绘制。

默认（``--style line``）绘制顶会（conference-paper）风格纯曲线：只保留
左/下坐标轴、外置细刻度、无网格、Okabe-Ito 色觉安全配色、无任何标记点。
交互频次是典型长尾分布，默认采用双对数坐标（``--xscale log --yscale log``）：
头部不会被压扁折叠在 y 轴上，各面板 x 轴刻度（1/10/100/1000/10000）完全一致；
曲线四周留边距，起点不与坐标轴贴合。交互数为 0 的物品在对数坐标下无法显示
（自动断开曲线）。三个数据集会额外输出一张 1x3 组合图（面板标签 (a)(b)(c)）。
``--style hist`` 可选直方图（固定线性坐标，柱宽由 --bar-width 控制）。
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_DATASETS = ["gowalla", "yelp2018", "amazon-book"]

# Okabe-Ito 通用配色：色觉安全、灰度打印可区分
DATASET_COLORS = {
    "gowalla": "#0072B2",
    "yelp2018": "#D55E00",
    "amazon-book": "#009E73",
}
DATASET_NAMES = {
    "gowalla": "Gowalla",
    "yelp2018": "Yelp2018",
    "amazon-book": "Amazon-Book",
}

# 流行度区域浅色：按排名划分（Top 5% / 5-40% / 40-100%），与攻击实验分层一致；
# 顺序即图例与堆叠顺序（自下而上）：Tail -> Medium-hot -> Hot
ZONE_COLORS = {
    "Hot": "#F2A9A1",
    "Medium-hot": "#F6CFA0",
    "Tail": "#A9DDB6",
}
ZONE_ORDER = ["Tail", "Medium-hot", "Hot"]

# 顶会风格中性墨色
PUBLICATION_INK = "#222222"


def count_interactions(lines: Iterable[str]) -> Counter:
    """统计每行（用户id 物品id...）中各物品的出现次数。"""
    counts: Counter = Counter()
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        for token in parts[1:]:
            counts[int(token)] += 1
    return counts


def build_series(
    counts: Counter,
    sort_by_count: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 (x, y, original_ids)。

    sort_by_count=False：x=物品id+1（1..max_id+1），保持原始顺序。
    sort_by_count=True：按交互数降序重映射，x=新 id（1..N），交互数最多者
    新 id=1；同数按原始 id 升序，零交互物品排在尾部。
    """
    if not counts:
        raise ValueError("counts is empty")
    n_items = max(counts) + 1
    if sort_by_count:
        items = sorted(
            range(n_items),
            key=lambda i: (-counts.get(i, 0), i),
        )
        x = np.arange(1, n_items + 1, dtype=np.int64)
        y = np.fromiter(
            (counts.get(i, 0) for i in items),
            dtype=np.int64,
            count=n_items,
        )
        return x, y, np.asarray(items, dtype=np.int64)
    x = np.arange(1, n_items + 1, dtype=np.int64)
    y = np.fromiter(
        (counts.get(i, 0) for i in range(n_items)),
        dtype=np.int64,
        count=n_items,
    )
    return x, y, np.arange(n_items, dtype=np.int64)


def build_popularity_histogram(
    counts: Counter,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按交互数构建论文版流行度直方图。

    分箱：[1], [2], [3–4], [5–8], [9–16], [17–32], ...（2 的幂边界）。
    返回 (centers, widths, ratios, edges)：
    - ratios：形状 (n_bins, 3) 的堆叠比例矩阵（%），列顺序 = ZONE_ORDER
      （Tail / Medium-hot / Hot）。颜色表示物品所属流行度类别，而不是交互次数
      阈值：先按交互数降序排名，Top 5% 物品为 Hot，5%~40% 为 Medium-hot，
      其余为 Tail；每根柱子的分段 = 该交互数箱内属于各类别的物品占比。
    三个数据集可直接比较；所有交互数 > 0 的物品参与统计。
    """
    if not counts:
        raise ValueError("counts is empty")
    max_count = max(counts.values())
    edges = [1.0, 2.0]
    k = 1
    while 2 ** k + 1 <= max_count:
        edges.append(float(2 ** k + 1))
        k += 1
    edges.append(float(2 ** k + 1))
    edges = np.asarray(edges)

    # 全部物品（含 0 交互物品，若存在）按交互数降序排名，切出三类
    n_total = max(counts) + 1
    items = sorted(
        range(n_total),
        key=lambda i: (-counts.get(i, 0), i),
    )
    hot_n = int(np.ceil(0.05 * n_total))
    cum_medium = int(np.ceil(0.40 * n_total))
    tier_idx = np.zeros(n_total, dtype=np.int64)
    for rank, item in enumerate(items):
        if rank < hot_n:
            tier_idx[item] = ZONE_ORDER.index("Hot")
        elif rank < cum_medium:
            tier_idx[item] = ZONE_ORDER.index("Medium-hot")
        else:
            tier_idx[item] = ZONE_ORDER.index("Tail")

    counts_arr = np.fromiter(
        (counts.get(i, 0) for i in range(n_total)),
        dtype=np.int64,
        count=n_total,
    )
    bin_idx = np.digitize(counts_arr, edges) - 1
    valid = counts_arr > 0
    matrix = np.zeros((len(edges) - 1, len(ZONE_ORDER)), dtype=np.int64)
    for i in np.nonzero(valid)[0]:
        matrix[bin_idx[i], tier_idx[i]] += 1
    ratios = matrix / n_total * 100.0

    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[:-1] * edges[1:])  # 几何中心
    return centers, widths, ratios, edges


def _apply_publication_style(ax) -> None:
    """顶会风格：只留左/下坐标轴、外置细刻度、无网格、深灰文字。"""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PUBLICATION_INK)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3.5,
        width=0.8,
        colors=PUBLICATION_INK,
        labelsize=9,
    )
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.grid(False)
    ax.set_facecolor("white")


def _apply_scale(ax, xscale: str, yscale: str) -> None:
    """设置坐标轴刻度（log/linear）。"""
    if xscale == "log":
        ax.set_xscale("log")
    if yscale == "log":
        ax.set_yscale("log")


def _apply_log_xticks(ax, x: np.ndarray) -> None:
    """固定对数 x 轴刻度为 10 的幂，保证各面板刻度一致且无越界杂刻度。"""
    if ax.get_xscale() == "log":
        max_x = float(x[-1])
        ax.set_xticks(
            [t for t in (1, 10, 100, 1000, 10000, 100000) if t <= max_x]
        )


def _apply_log2_xticks(ax, edges: np.ndarray) -> None:
    """直方图 log2 x 轴刻度：1/2/4/8/...（与 2 的幂分箱对齐）。"""
    if ax.get_xscale() == "log":
        k_max = int(round(np.log2(edges[-1] - 1)))
        ax.set_xticks([2.0 ** k for k in range(0, k_max + 1)])


def _add_zone_legend(ax) -> None:
    """Hot / Medium-hot / Tail 区域浅色图例。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=ZONE_COLORS[z], edgecolor="none", label=z)
        for z in ZONE_ORDER
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)


def _grouped_zone_bars(ax, edges: np.ndarray, ratios: np.ndarray) -> None:
    """每个交互数分箱内并排三根柱（Tail / Medium-hot / Hot），非堆叠。"""
    widths = edges[1:] - edges[:-1]
    for idx, zone in enumerate(ZONE_ORDER):
        # 组内 3 根柱的位置：1/6、3/6、5/6 处，柱宽取组宽的 30%（接近不重叠极限）
        x = edges[:-1] + widths * (2 * idx + 1) / 6
        w = widths * 0.30
        ax.bar(
            x, ratios[:, idx], width=w,
            color=ZONE_COLORS[zone], linewidth=0, zorder=5,
        )


def _plot_y(ax, x: np.ndarray, y: np.ndarray, color: str, yscale: str) -> None:
    """绘制纯曲线：log 坐标下将 0 值掩码掉（自动断线）。"""
    y_plot = np.ma.masked_where(y == 0, y) if yscale == "log" else y
    ax.plot(x, y_plot, color=color, linewidth=1.2, zorder=5)


def plot_series(
    dataset: str,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    split_label: str,
    x_label: str = "Item ID (1-based)",
    style: str = "line",
    bar_width: float = 1.0,
    xscale: str = "log",
    yscale: str = "log",
    figsize: Tuple[float, float] = (5.2, 3.4),
    dpi: int = 300,
) -> "matplotlib.figure.Figure":
    """绘制单个数据集的交互频次图：顶会风格纯曲线（默认）或直方图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = DATASET_COLORS.get(dataset, "#0072B2")
    name = DATASET_NAMES.get(dataset, dataset)
    fig, ax = plt.subplots(figsize=figsize)
    # 图形画在坐标轴之上：0 交互的物品落在 y=0，颜色覆盖坐标轴才可见
    ax.set_axisbelow(False)
    if style == "hist":
        # 直方图固定线性坐标，避免零柱在对数轴下失效
        xscale = yscale = "linear"
        ax.bar(x, y, width=bar_width, color=color, linewidth=0, zorder=5)
    else:
        _plot_y(ax, x, y, color, yscale)
    _apply_scale(ax, xscale, yscale)
    _apply_log_xticks(ax, x)
    _apply_publication_style(ax)
    ax.set_xlabel(x_label, fontsize=10.5, color=PUBLICATION_INK, labelpad=5)
    ax.set_ylabel(
        "Interaction count", fontsize=10.5, color=PUBLICATION_INK, labelpad=5
    )
    ax.set_title(
        f"{name} - Item interaction frequency ({split_label})",
        fontsize=11,
        color=PUBLICATION_INK,
        pad=8,
    )
    # 四周留边距：曲线起点不与坐标轴折叠贴合
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return fig


def plot_all_datasets(
    panels: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    x_label: str,
    split_label: str,
    xscale: str = "log",
    yscale: str = "log",
    figsize: Tuple[float, float] = (12.5, 3.8),
    dpi: int = 300,
) -> "matplotlib.figure.Figure":
    """顶会构图：1xN 组合图，面板 (a)(b)(c)... 对应各数据集。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    letters = ("a", "b", "c", "d", "e", "f")
    for ax, (dataset, x, y), letter in zip(axes, panels, letters):
        color = DATASET_COLORS.get(dataset, "#0072B2")
        name = DATASET_NAMES.get(dataset, dataset)
        ax.set_axisbelow(False)
        _plot_y(ax, x, y, color, yscale)
        _apply_scale(ax, xscale, yscale)
        _apply_log_xticks(ax, x)
        _apply_publication_style(ax)
        ax.set_title(
            f"({letter}) {name}", fontsize=11, color=PUBLICATION_INK, pad=8
        )
        ax.set_xlabel(x_label, fontsize=10.5, color=PUBLICATION_INK, labelpad=5)
        ax.margins(x=0.02, y=0.08)
    axes[0].set_ylabel(
        "Interaction count", fontsize=10.5, color=PUBLICATION_INK, labelpad=5
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return fig


def plot_popularity_hist(
    dataset: str,
    edges: np.ndarray,
    ratios: np.ndarray,
    out_path: Path,
    split_label: str,
    figsize: Tuple[float, float] = (5.2, 3.4),
    dpi: int = 300,
) -> "matplotlib.figure.Figure":
    """绘制单个数据集的流行度直方图：x=交互次数，y=物品占比（%），
    柱子按物品所属流行度类别（Tail/Medium-hot/Hot）堆叠着色。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = DATASET_NAMES.get(dataset, dataset)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(False)
    _grouped_zone_bars(ax, edges, ratios)
    ax.set_xscale("log", base=2)
    _apply_log2_xticks(ax, edges)
    _apply_publication_style(ax)
    ax.set_xlabel("Interaction count", fontsize=10.5, color=PUBLICATION_INK, labelpad=5)
    ax.set_ylabel("Proportion of items (%)", fontsize=10.5, color=PUBLICATION_INK, labelpad=5)
    ax.set_title(
        f"Popularity histogram ({name}, {split_label})",
        fontsize=11,
        color=PUBLICATION_INK,
        pad=8,
    )
    _add_zone_legend(ax)
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return fig


def plot_popularity_hist_all(
    panels: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    split_label: str,
    figsize: Tuple[float, float] = (12.5, 3.8),
    dpi: int = 300,
) -> "matplotlib.figure.Figure":
    """顶会构图：1xN 组合直方图，面板 (a)(b)(c)... 对应各数据集。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    letters = ("a", "b", "c", "d", "e", "f")
    for ax, (dataset, edges, ratios), letter in zip(axes, panels, letters):
        name = DATASET_NAMES.get(dataset, dataset)
        ax.set_axisbelow(False)
        _grouped_zone_bars(ax, edges, ratios)
        ax.set_xscale("log", base=2)
        _apply_log2_xticks(ax, edges)
        _apply_publication_style(ax)
        ax.set_title(
            f"({letter}) {name}", fontsize=11, color=PUBLICATION_INK, pad=8
        )
        ax.set_xlabel("Interaction count", fontsize=10.5, color=PUBLICATION_INK, labelpad=5)
        ax.margins(x=0.02, y=0.08)
    axes[0].set_ylabel(
        "Proportion of items (%)", fontsize=10.5, color=PUBLICATION_INK, labelpad=5
    )
    _add_zone_legend(axes[0])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="数据集名（默认：gowalla yelp2018 amazon-book）",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train"],
        help="数据分片（默认：train；可传 train test 合并统计）",
    )
    parser.add_argument(
        "--sort",
        choices=["count", "id"],
        default="count",
        help="count=按交互数降序重映射（默认，交互最多者新 id=1）；"
             "id=按原始物品 id+1 绘制",
    )
    parser.add_argument(
        "--style",
        choices=["hist", "line"],
        default="line",
        help="line=顶会风格纯曲线（默认）；hist=直方图（固定线性坐标，"
             "柱宽由 --bar-width 控制）",
    )
    parser.add_argument(
        "--bar-width",
        type=float,
        default=1.0,
        help="直方图柱宽（默认 1.0，相邻无缝；调小可留间隙）",
    )
    parser.add_argument(
        "--xscale",
        choices=["log", "linear"],
        default="log",
        help="x 轴刻度：log=对数（默认，头部展开、刻度均匀一致）；"
             "linear=线性",
    )
    parser.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="y 轴刻度：log=对数（默认，长尾分布更明显）；linear=线性",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="原始数据根目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="输出目录（默认：本目录 outputs/）",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sort_by_count = args.sort == "count"
    x_label = (
        "Item ID (sorted by interaction count)"
        if sort_by_count
        else "Item ID (1-based)"
    )
    split_label = "+".join(args.splits)
    panels: List[Tuple[str, np.ndarray, np.ndarray]] = []
    hist_panels: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for dataset in args.datasets:
        counts: Counter = Counter()
        for split in args.splits:
            path = args.raw_root / dataset / f"{split}.txt"
            if not path.exists():
                raise SystemExit(f"数据文件不存在: {path}")
            with open(path, encoding="utf-8") as f:
                counts.update(count_interactions(f))
        if not counts:
            raise SystemExit(f"数据集 {dataset} 没有统计到任何交互")

        x, y, original_ids = build_series(counts, sort_by_count=sort_by_count)
        panels.append((dataset, x, y))

        # 统计结果 CSV（与图一一对应，便于核对）
        csv_path = args.out_dir / f"item_freq_{dataset}.csv"
        if sort_by_count:
            data = np.column_stack([x, original_ids, y])
            header = "new_item_id,original_item_id,interaction_count"
        else:
            data = np.column_stack([x, y])
            header = "item_id_plus_1,interaction_count"
        np.savetxt(
            csv_path,
            data,
            fmt="%d",
            delimiter=",",
            header=header,
            comments="",
        )

        png_path = args.out_dir / f"item_freq_{dataset}.png"
        fig = plot_series(
            dataset,
            x,
            y,
            png_path,
            split_label,
            x_label,
            style=args.style,
            bar_width=args.bar_width,
            xscale=args.xscale,
            yscale=args.yscale,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(
            f"[OK] {dataset}: items={len(x)}, "
            f"interactions={int(y.sum())} -> {png_path}"
        )

        # 流行度直方图：x=交互次数（2 的幂分箱），y=物品占比（%），
        # 柱子按排名类别（Tail/Medium-hot/Hot）堆叠着色
        _, _, ratios, edges = build_popularity_histogram(counts)
        hist_png_path = args.out_dir / f"popularity_hist_{dataset}.png"
        fig = plot_popularity_hist(
            dataset, edges, ratios, hist_png_path, split_label
        )
        plt.close(fig)
        hist_panels.append((dataset, edges, ratios))
        print(f"[OK] {dataset}: popularity histogram -> {hist_png_path}")

    if args.style == "line" and len(panels) > 1:
        combined_path = args.out_dir / "item_freq_all_datasets.png"
        fig = plot_all_datasets(
            panels,
            combined_path,
            x_label,
            split_label,
            xscale=args.xscale,
            yscale=args.yscale,
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        print(f"[OK] combined 1x{len(panels)} -> {combined_path}")

    if len(hist_panels) > 1:
        combined_hist_path = args.out_dir / "popularity_hist_all_datasets.png"
        fig = plot_popularity_hist_all(hist_panels, combined_hist_path, split_label)
        plt.close(fig)
        print(
            f"[OK] combined popularity histogram 1x{len(hist_panels)} "
            f"-> {combined_hist_path}"
        )


if __name__ == "__main__":
    main()
