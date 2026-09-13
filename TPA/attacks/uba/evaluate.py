"""UBA 攻击效果评估 —— 共享实现见 evaluation/attack_eval.py

两类指标：
1. 仓库统一口径（转出 attack_eval）：目标物品 HR@K/NDCG@K + 模型效用 recall/ndcg；
2. UBA 专属口径（本文件唯一新增逻辑）：**目标用户群** U_t 上的 HR@K / NDCG@K。

为什么需要第 2 类：论文 Table 1 报的是"被攻击目标用户里有多少人在 Top-K 收到目标
物品"（官方 utils/evaluator.py 就是逐目标用户累加 hr_10/20 再除以 50）；仓库统一的
target_hr@K 则是"所有未交互目标物品的合格用户"口径。两者都有意义，因此 UBA 把论文
口径作为报告附加段输出，而不改动共享评估层（否则会影响其它攻击的 BestTracker 契约）。

评估协议：全量排序 + 过滤该用户在**干净训练集**中的已交互物品（clean / poisoned
两次评估口径一致，避免假用户交互混进真实用户的过滤集合）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from evaluation.attack_eval import (  # noqa: F401 - 薄壳转出，供 fit.py 统一 import
    aggregate_target_metrics,
    build_attack_eval_metrics,
    compare_models,
    compute_target_metrics,
    format_report,
    ranking_scores,
    save_report,
)


def compute_target_user_metrics(scores: torch.Tensor,
                                user_ids: Sequence[int],
                                clean_user_items: Dict[int, set],
                                target_users: Sequence[int],
                                target_item: int, k: int,
                                device: Any = None) -> Dict[str, Any]:
    """目标用户群 U_t 上的 HR@K / NDCG@K（论文 Table 1 口径）。

    矩阵变换过程：
        scores            (n_eval_users, n_items)  —— 评估用户的全量排序分数
        sub = scores[rows]  (|U_t|', n_items)      —— 只看目标用户行
        sub.masked_fill(已交互, -inf)               —— 过滤该用户训练集已见物品
        topk(sub, k)     (|U_t|', k)              —— Top-K 物品 id
        hit = topk == i  (|U_t|', k) bool          —— 目标物品是否落在 Top-K
        rank_all = (sub > sub[:, i]).sum(1) + 1    —— 严格高于目标分的候选数 + 1

    为什么要 rank_all：HR@K 是截断指标（K 之外的信息全丢），论文式 rank_ui 定义能
    反映"攻击把目标物品往前推了多少"，与 evaluation/attack_eval.py 的口径保持一致。

    返回 {target_item, k, n_target_users, n_evaluated, hit_users, hr@k, ndcg@k,
    mean_rank, mean_rank_all}；无可用用户时指标为 0 / None。
    """
    target_item = int(target_item)
    rows: List[int] = []
    used_users: List[int] = []
    index = {int(u): r for r, u in enumerate(user_ids)}
    for u in target_users:
        u = int(u)
        if u not in index:
            continue
        if target_item in clean_user_items.get(u, set()):
            # 目标用户若已交互目标物品，就不属于"被攻击受益人群"，与选人策略一致
            continue
        rows.append(index[u])
        used_users.append(u)

    base = {
        "target_item": target_item,
        "k": int(k),
        "n_target_users": len(target_users),
        "n_evaluated": len(rows),
        "hit_users": 0,
        "hr@k": 0.0,
        "ndcg@k": 0.0,
        "mean_rank": None,
        "mean_rank_all": None,
        "_users": used_users,
    }
    if not rows:
        return base

    sub = scores[torch.tensor(rows, dtype=torch.long)].clone()
    mask = torch.zeros_like(sub, dtype=torch.bool)
    for r, u in enumerate(used_users):
        seen = clean_user_items.get(u, set())
        if seen:
            mask[r, torch.tensor(sorted(int(i) for i in seen),
                                 dtype=torch.long)] = True
    sub = sub.masked_fill(mask, float("-inf"))
    if device is not None and str(device).startswith("cuda"):
        sub = sub.to(device)

    top_k = min(int(k), int(sub.shape[1]))
    topk = torch.topk(sub, top_k, dim=1).indices.cpu()
    hit = topk == target_item
    hit_rows = hit.any(dim=1)
    hits = int(hit_rows.sum())
    n_eval = len(rows)
    position = hit.int().argmax(dim=1) + 1          # 命中行在 Top-K 中的 1-based 位次
    ranks_hit = position[hit_rows]
    dcg = float((1.0 / torch.log2(ranks_hit.double() + 1)).sum()) if hits else 0.0
    target_col = sub[:, target_item].unsqueeze(-1)
    rank_all = (sub > target_col).sum(dim=1) + 1    # 严格高于目标分 → 论文 rank_ui

    return {
        **base,
        "hit_users": hits,
        "hr@k": hits / n_eval,
        "ndcg@k": dcg / n_eval,
        "mean_rank": float(ranks_hit.double().mean()) if hits else None,
        "mean_rank_all": float(rank_all.double().mean()),
    }


def format_target_user_report(metrics: Dict[str, Any]) -> str:
    """把目标用户群指标格式化为 Markdown 段（追加在 attack_comparison.md 之后）。"""
    k = metrics["k"]
    lines = [
        f"# UBA 目标用户群指标（Top-{k}，论文 Table 1 口径）",
        "",
        "统计对象 = attack.uba.target_users 选出的目标用户（干净训练集未交互目标物品）；",
        "与 uba_comparison.md 的 target_hr@K（全合格用户口径）互为补充。",
        "",
        f"| 阶段 | 评估用户数 | 命中人数 | HR@{k} | NDCG@{k} | 平均排名(全体) |",
        "|---|---|---|---|---|---|",
    ]
    for phase in ("clean", "poisoned"):
        m = metrics.get(phase)
        if not m:
            continue
        rank_all = ("N/A" if m.get("mean_rank_all") is None
                    else f"{m['mean_rank_all']:.1f}")
        lines.append(
            f"| {phase} | {m.get('n_evaluated', 0)} | {m.get('hit_users', 0)} | "
            f"{m.get('hr@k', 0.0):.4f} | {m.get('ndcg@k', 0.0):.4f} | {rank_all} |"
        )
    clean = metrics.get("clean") or {}
    poisoned = metrics.get("poisoned") or {}
    if clean and poisoned:
        delta = poisoned.get("hr@k", 0.0) - clean.get("hr@k", 0.0)
        lines += [
            "",
            f"- 目标用户群 HR@{k}：{clean.get('hr@k', 0.0):.4f} → "
            f"{poisoned.get('hr@k', 0.0):.4f}（{delta:+.4f}）",
            f"- 命中人数：{clean.get('hit_users', 0)} → {poisoned.get('hit_users', 0)}"
            f"（共 {poisoned.get('n_evaluated', 0)} 个目标用户被评估）",
        ]
    return "\n".join(lines) + "\n"


def save_target_user_report(metrics: Dict[str, Any], out_dir: Path) -> Path:
    """写 target_user_metrics.md / .json，返回 md 路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "target_user_metrics.md"
    md_path.write_text(format_target_user_report(metrics), encoding="utf-8")
    (out_dir / "target_user_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path
