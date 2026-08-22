"""攻击分类公共逻辑：按训练集交互数（而非模型嵌入推荐频次）划分三档。

背景：原 classify 使用干净模型的 Top-K 推荐频次划分 popular / ordinary /
cold。模型本身带有流行度偏置（热门物品更容易被推荐），导致“流行物品”的
划分与真实交互热度循环依赖。现改为直接统计训练集每个物品的交互次数，按
交互数降序排名划分，与流行度直方图（Hot / Medium-hot / Tail）的分层一致：

  - popular  (Hot):        交互数前 popular_ratio（默认 5%）
  - ordinary (Medium-hot): popular 之后至 medium_ratio（默认 40%）
  - cold     (Tail):       其余 40% ~ 100%

交互数相同的物品按物品 id 升序排列，结果确定可复现。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Counter as CounterType, Dict, List, Tuple


def interaction_counts(train_pairs: List[Tuple[int, int]]) -> CounterType[int]:
    """统计每个物品在训练集中的交互次数。

    train_pairs: [(user_id, item_id), ...]。
    """
    return Counter(item for _, item in train_pairs)


def classify_by_interaction_counts(
    counts: CounterType[int],
    num_items: int,
    popular_ratio: float = 0.05,
    medium_ratio: float = 0.40,
) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """按交互数降序排名划分 popular / ordinary / cold 三档。"""
    if num_items <= 0:
        raise ValueError("num_items 必须为正")
    if not (0.0 < popular_ratio < medium_ratio <= 1.0):
        raise ValueError(
            f"需要 0 < popular_ratio < medium_ratio <= 1，"
            f"实际 popular_ratio={popular_ratio}, medium_ratio={medium_ratio}"
        )

    items = sorted(range(num_items), key=lambda i: (-counts.get(i, 0), i))
    n_popular = math.ceil(num_items * popular_ratio)
    n_ordinary = math.ceil(num_items * medium_ratio) - n_popular
    popular = items[:n_popular]
    ordinary = items[n_popular:n_popular + n_ordinary]
    cold = items[n_popular + n_ordinary:]

    summary = {
        "basis": "interaction_count",
        "num_items": num_items,
        "interacting_items": sum(
            1 for i in range(num_items) if counts.get(i, 0) > 0
        ),
        "popular_ratio": popular_ratio,
        "medium_ratio": medium_ratio,
        "popular_count": len(popular),
        "ordinary_count": len(ordinary),
        "cold_count": len(cold),
        "min_popular_count": counts.get(popular[-1], 0) if popular else None,
        "max_cold_count": counts.get(cold[0], 0) if cold else None,
        "top_interaction_items": [
            (i, counts.get(i, 0)) for i in popular[:10]
        ],
        "bottom_interaction_items": [
            (i, counts.get(i, 0)) for i in cold[-10:]
        ] if cold else [],
    }
    return {"popular": popular, "ordinary": ordinary, "cold": cold}, summary
