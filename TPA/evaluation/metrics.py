"""推荐系统评估指标（共享层）
对齐论文 Section 4.1 的 all-ranking 协议：
- recall@K: 正样本在 Top-K 推荐中命中的比例
- ndcg@K: 考虑排名位置的归一化折损累积增益
- expected_percentile_rank: WMF 论文（Hu et al. 2008）Eq.(8) 主指标，
  测试观测在推荐列表中的期望百分位（越低越好，随机期望 50%）
- 候选集 = 所有物品（用户未交互过的）
"""
import math
import torch
import numpy as np
from typing import Dict, List, Tuple


def _topk_by_batch(scores: torch.Tensor, k: int,
                   train_user_items: Dict[int, set],
                   test_user_items: Dict[int, set]):
    """分批计算 Recall@K 和 NDCG@K（all-ranking 协议）。

    scores: (n_users, n_items) 预测分数矩阵
    train_user_items: {user_id: {item_ids}} 训练集交互（用于过滤）
    test_user_items: {user_id: {item_ids}} 测试集正样本（评估目标）
    """
    n_users = scores.shape[0]

    # 过滤训练集已交互物品（设为 -inf）
    for u in range(n_users):
        if u in train_user_items:
            for i in train_user_items[u]:
                scores[u, i] = float('-inf')

    _, topk_indices = torch.topk(scores, k, dim=1)

    recalls = []
    ndcgs = []
    for u in range(n_users):
        if u not in test_user_items:
            continue
        pos = test_user_items[u]
        if not pos:
            continue

        hits = 0
        dcg = 0.0
        for rank, item in enumerate(topk_indices[u].tolist(), start=1):
            if item in pos:
                hits += 1
                dcg += 1.0 / np.log2(rank + 1)

        ideal_hits = min(k, len(pos))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

        recalls.append(hits / len(pos))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def compute_metrics(scores: torch.Tensor, train_user_items: Dict[int, set],
                    test_user_items: Dict[int, set],
                    k: int = 20) -> Dict[str, float]:
    """计算 Recall@K 和 NDCG@K（all-ranking 协议）。

    Args:
        scores: (n_users, n_items) 预测分数
        train_user_items: 训练集交互 {user: {item, ...}}，用于过滤已交互物品
        test_user_items: 测试集正样本 {user: {item, ...}}，评估目标
        k: Top-K
    """
    recall, ndcg = _topk_by_batch(scores, k, train_user_items, test_user_items)
    return {f"recall@{k}": recall, f"ndcg@{k}": ndcg}


def rank_values(scores: torch.Tensor,
                train_user_items: Dict[int, set],
                test_user_items: Dict[int, set],
                test_weights: Dict[int, Dict[int, float]] = None,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """论文 Eq.(8) 的逐观测百分位秩（Rank CDF 数据源）。

    rank_ui 定义为用户 u 候选中预测分严格高于物品 i 的物品占比
    （0 = 最优先，1 = 最劣，随机预测期望 0.5）。训练集已交互物品
    过滤后不参与分母。

    Returns:
        (ranks, weights)：逐观测的 rank_ui 与其权重 r^t_ui
        （缺省全 1），长度为有效测试观测数；无有效观测时为空数组。
    """
    scores = scores.clone()
    n_items = scores.shape[1]
    if train_user_items:
        for u, items in train_user_items.items():
            if u < scores.shape[0]:
                for i in items:
                    if i < n_items:
                        scores[u, i] = float("-inf")

    ranks = []
    weights = []
    for u, pos_items in test_user_items.items():
        if u >= scores.shape[0]:
            continue
        row = scores[u]
        valid = int((row > float("-inf")).sum())
        if valid <= 0:
            continue
        for i in pos_items:
            if i >= n_items:
                continue
            s = float(row[i])
            if not math.isfinite(s):
                continue  # 测试正样本同时被训练集过滤
            ranks.append(int((row > s).sum()) / valid)
            w = 1.0
            if test_weights is not None:
                w = float(test_weights.get(u, {}).get(i, 1.0))
            weights.append(w)
    return (np.asarray(ranks, dtype=np.float64),
            np.asarray(weights, dtype=np.float64))


def expected_percentile_rank(scores: torch.Tensor,
                             train_user_items: Dict[int, set],
                             test_user_items: Dict[int, set],
                             test_weights: Dict[int, Dict[int, float]] = None,
                             ) -> float:
    """WMF 论文主指标：expected percentile rank（Eq.8，越低越好）。

    rank̄ = Σ_{u,i} r^t_ui · rank_ui / Σ_{u,i} r^t_ui
    复用 rank_values() 的逐观测秩；无有效测试观测时返回 NaN。
    """
    ranks, weights = rank_values(scores, train_user_items,
                                 test_user_items, test_weights)
    total_weight = float(np.sum(weights))

    if total_weight <= 0:
        return float("nan")
    return float(np.sum(weights * ranks) / total_weight)
