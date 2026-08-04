"""LightGCN 评估指标
对齐论文 Section 4.1 的 all-ranking 协议：
- recall@K: 正样本在 Top-K 推荐中命中的比例
- ndcg@K: 考虑排名位置的归一化折损累积增益
- 候选集 = 所有物品（用户未交互过的）
"""
import torch
import numpy as np
from typing import Dict, List


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
