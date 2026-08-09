"""Bandwagon 攻击效果评估

两类指标：
1. 模型效用（干净模型 vs 中毒模型在测试集上的 recall@K / ndcg@K）
2. 攻击效果（目标物品在 Top-K 中的曝光率 / 命中用户数 / NDCG / 平均排名）

评估协议与 LightGCN 一致：all-ranking、过滤训练集已交互物品。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from evaluation.metrics import compute_metrics


def ranking_scores(model, test_pairs: List[Tuple[int, int]]
                   ) -> Tuple[torch.Tensor, List[int], Dict[int, set]]:
    """对测试用户做全量排序评分。

    返回 (scores, test_user_ids, test_pos)。
    scores: (n_test_users, n_items)，行顺序与 test_user_ids 一致。
    """
    test_pos: Dict[int, set] = {}
    for u, i in test_pairs:
        test_pos.setdefault(u, set()).add(i)
    test_users = sorted(test_pos.keys())

    model.set_eval()
    with torch.no_grad():
        user_emb = model.get_user_embeddings()
        item_emb = model.get_item_embeddings()
        ids = torch.LongTensor(test_users).to(user_emb.device)
        scores = user_emb[ids] @ item_emb.T
    return scores, test_users, test_pos


def compute_target_metrics(scores: torch.Tensor, user_ids: List[int],
                           clean_user_items: Dict[int, set],
                           target_items: List[int], k: int) -> Dict[int, Dict[str, Any]]:
    """目标物品的攻击效果指标（HR@K / NDCG@K）。

    只统计"训练集中未交互过该目标物品"的合法用户（攻击的目标人群），
    用干净训练集过滤，保证 clean / poisoned 两次评估口径一致。

    返回 {target: {hr@k, ndcg@k, hit_users, mean_rank, mean_rank_all}}
    - hr@k: 目标物品进入 Top-K 的用户比例（= 经典 Hit Rate @K）
    - ndcg@k: 单目标 NDCG（命中用户按排名位置折损，IDCG=1）
    - mean_rank: 命中用户的平均排名（未命中记 None；越小攻击越强）
    - mean_rank_all: 全体合格用户的平均排名（不受 Top-K 截断，更灵敏）
    """ 
    topk = torch.topk(scores, k, dim=1).indices  # (n_users, k)
    ranks_all = torch.argsort(scores, dim=1, descending=True)  # (n_users, n_items)
    out: Dict[int, Dict[str, Any]] = {}
    for t in target_items:
        eligible = [
            r for r, uid in enumerate(user_ids)
            if t not in clean_user_items.get(uid, set())
        ]
        n_elig = len(eligible)
        if n_elig == 0:
            out[t] = {"hr@k": 0.0, "ndcg@k": 0.0, "hit_users": 0,
                      "mean_rank": None, "mean_rank_all": None}
            continue

        hits = 0
        dcg = 0.0
        ranks: List[int] = []
        ranks_all_list: List[int] = []
        for r in eligible:
            pos = (topk[r] == t).nonzero(as_tuple=False)
            if pos.numel():
                rank = int(pos.item()) + 1
                hits += 1
                dcg += 1.0 / np.log2(rank + 1)
                ranks.append(rank)
            pos_all = (ranks_all[r] == t).nonzero(as_tuple=False)
            ranks_all_list.append(int(pos_all.item()) + 1)

        out[t] = {
            "hr@k": hits / n_elig,
            "ndcg@k": dcg / n_elig,
            "hit_users": hits,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "mean_rank_all": float(np.mean(ranks_all_list)),
            # 旧字段别名，兼容已有 JSON 消费者
            "exposure": hits / n_elig,
            "ndcg": dcg / n_elig,
        }
    return out


def compare_models(clean_model, poisoned_model, clean_meta: Dict[str, Any],
                   poisoned_meta: Dict[str, Any], target_items: List[int], k: int,
                   report_utility: bool = True
                   ) -> Dict[str, Any]:
    """clean vs poisoned 全量对比。"""
    scores_c, users_c, test_pos_c = ranking_scores(clean_model, clean_meta["test_pairs"])
    scores_p, users_p, test_pos_p = ranking_scores(poisoned_model, poisoned_meta["test_pairs"])

    # 模型效用（各自训练集过滤；真实用户的训练交互在注入前后一致）
    # 用于衡量攻击代价：投毒后推荐质量不应显著下降
    if report_utility:
        clean_util = compute_metrics(scores_c, clean_meta["user_items"], test_pos_c, k)
        poisoned_util = compute_metrics(scores_p, poisoned_meta["user_items"], test_pos_p, k)
    else:
        clean_util = poisoned_util = None

    # 攻击效果（统一用干净训练集过滤目标人群）
    clean_att = compute_target_metrics(scores_c, users_c, clean_meta["user_items"],
                                       target_items, k)
    poisoned_att = compute_target_metrics(scores_p, users_p, clean_meta["user_items"],
                                          target_items, k)

    return {
        "k": k,
        "model_utility": {
            "clean": clean_util,
            "poisoned": poisoned_util,
        },
        "target_metrics": {
            "clean": clean_att,
            "poisoned": poisoned_att,
        },
    }


def format_report(report: Dict[str, Any]) -> str:
    """把对比结果格式化为 Markdown 报告。"""
    k = report["k"]
    lines = [
        f"# Bandwagon（从众）攻击对比报告（Top-{k}）",
        "",
    ]
    cu = report["model_utility"]["clean"]
    pu = report["model_utility"]["poisoned"]
    if cu is not None:
        lines += [
            "## 模型效用（测试集 all-ranking，投毒代价检查）",
            "",
            "| 指标 | Clean | Poisoned | Δ |",
            "|------|-------|----------|---|",
        ]
        for key in cu:
            delta = pu[key] - cu[key]
            lines.append(
                f"| {key} | {cu[key]:.4f} | {pu[key]:.4f} | {delta:+.4f} |"
            )

    lines += [
        "",
        f"## 目标物品攻击效果（HR@{k} / NDCG@{k}）",
        "",
        "合格用户 = 训练集未交互过该目标物品的用户（统一用干净训练集过滤）",
        "",
        f"| Target | Clean HR@{k} | Poisoned HR@{k} | Clean NDCG@{k} | Poisoned NDCG@{k} | "
        f"Clean 平均排名 | Poisoned 平均排名 |",
        f"|--------|------------|----------------|---------------|------------------|"
        f"---------------|------------------|",
    ]
    ca = report["target_metrics"]["clean"]
    pa = report["target_metrics"]["poisoned"]
    for t in ca:
        cr = ca[t]
        pr = pa[t]
        rank_c = f"{cr['mean_rank_all']:.1f}"
        rank_p = f"{pr['mean_rank_all']:.1f}"
        lines.append(
            f"| {t} | {cr['hr@k']:.4f} | {pr['hr@k']:.4f} | "
            f"{cr['ndcg@k']:.4f} | {pr['ndcg@k']:.4f} | {rank_c} | {rank_p} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_report(report: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "bandwagon_comparison.md"
    md_path.write_text(format_report(report), encoding="utf-8")
    (out_dir / "bandwagon_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return md_path
