# -*- coding: utf-8 -*-
"""原生曝光（native exposure）：按"在原始嵌入上 HR/NDCG 最高"定义物品的受欢迎程度。

为什么需要它（目标选择口径）：
    生产计划原先按**交互数**取目标（top_popularity）。但"受欢迎"更准确的
    定义是"该物品在原始模型上被推荐给未交互用户的命中率"。两者在
    ml100k+LightGCN 上实测高度一致（Spearman ≈ 0.997，见 tmp 核查脚本），
    但在别的数据集/骨干上未必成立——所以 pre 同时支持两种策略。

口径（与仓库 compute_target_metrics 一致）：
    合格用户 = 测试用户中**训练集未与该物品交互过**的人；
    HR@K     = 合格用户里，该物品进入其 Top-K 的比例。
    分子分母都必须只用合格用户——否则热门物品会被自己的消费者刷高（可 > 1）。

实现：向量化。
    hit[u, j]  = j ∈ topk(u)；elig[u, j] = 用户 u 训练集里没有 j
    HR@K[j] = Σ_u (hit & elig) / Σ_u elig

用法示例：
    exp = native_exposure(model, meta, k=10)   # {item_id: {"hr":.., "n_elig":..}}
    top = select_targets_by_exposure(exp, counts, num_items=5, min_interactions=20)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch

from evaluation.attack_eval import ranking_scores  # noqa: E402


def native_exposure(model: Any, meta: Dict[str, Any], k: int = 10
                    ) -> Dict[int, Dict[str, float]]:
    """对全部物品算原生 HR@K（合格用户口径），返回 {item_id: {...}}。"""
    num_items = int(meta["num_items"])
    S, users, _ = ranking_scores(model, meta["test_pairs"])
    topk = torch.topk(S, k, dim=1).indices                    # (U, K)
    hit = torch.zeros(len(users), num_items, dtype=torch.bool)
    hit.scatter_(1, topk, True)
    elig = torch.ones(len(users), num_items, dtype=torch.bool)
    seg = defaultdict(set)
    for u, i in meta["train_pairs"]:
        seg[i].add(u)
    row_of = {u: r for r, u in enumerate(users)}
    for item, us in seg.items():
        for u in us:
            r = row_of.get(u)
            if r is not None:
                elig[r, item] = False
    n_elig = elig.sum(dim=0)
    n_hit = (hit & elig).sum(dim=0)
    out: Dict[int, Dict[str, float]] = {}
    for j in range(num_items):
        ne = int(n_elig[j].item())
        if ne == 0:
            continue
        out[j] = {"hr@k": float(n_hit[j].item()) / ne, "n_elig": ne,
                  "hits": int(n_hit[j].item())}
    return out


def select_targets_by_exposure(exposure: Dict[int, Dict[str, float]],
                               counts: Dict[int, int], num_items: int,
                               min_interactions: int
                               ) -> List[Tuple[int, int, float]]:
    """按原生 HR@K 降序取前 K 个（同分时按交互数、再按 id 稳定排序）。

    返回 [(item_id, interactions, hr@k), ...]，rank 由调用方按顺序给出。
    """
    cand = [(j, counts.get(j, 0), exposure.get(j, {}).get("hr@k", 0.0))
            for j in exposure if counts.get(j, 0) >= min_interactions]
    cand.sort(key=lambda t: (-t[2], -t[1], t[0]))
    if len(cand) < num_items:
        raise ValueError(
            f"满足 min_interactions={min_interactions} 且有合格用户的物品只有 "
            f"{len(cand)} 个，少于 num_items={num_items}")
    return cand[:num_items]
