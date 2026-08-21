"""攻击评估共享层 —— 所有攻击模块（tpa/pgd/bandwagon/random）统一使用。

收敛自 attacks/{attack}/evaluate.py 的重复实现：
- ranking_scores / compute_target_metrics / compare_models：纯共用
- aggregate_target_metrics：多目标 hr@K / ndcg@K 等权均值（checkpoint 选优用）
- build_attack_eval_metrics：训练中单次评估 = 整体指标 + 目标指标合并
- format_report / save_report：报告标题与输出文件名由 name 参数控制

两类指标：
1. 模型效用（clean vs poisoned 在测试集上的 recall@K / ndcg@K）——投毒代价检查
2. 攻击效果（目标物品在 Top-K 中的 HR@K / NDCG@K / 命中用户数 / 平均排名）

评估协议与 LightGCN 一致：all-ranking、过滤训练集已交互物品。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from evaluation.metrics import compute_metrics
from training.metrics import match_metric_values


# 报告输出文件名与标题：按攻击模块名归一化，保持各攻击现有文件名不变
REPORT_NAMES = {
    "tpa": "attack",
    "random": "attack",
    "pgd": "pgd",
    "bandwagon": "bandwagon",
    "attack_imp_direct_poison": "attack",
}
REPORT_TITLES = {
    "pgd": "PGD（投影梯度上升投毒）攻击对比报告",
    "bandwagon": "Bandwagon（从众）攻击对比报告",
}


def ranking_scores(model, test_pairs: List[Tuple[int, int]],
                   batch_size: int = 1024
                   ) -> Tuple[torch.Tensor, List[int], Dict[int, set]]:
    """对测试用户做全量排序评分。

    返回 (scores, test_user_ids, test_pos)。
    scores: (n_test_users, n_items)，行顺序与 test_user_ids 一致。

    显存安全：按 batch_size 个用户分块在 GPU 上算分，每块立即 .cpu()，
    避免大矩阵（如 gowalla 29k×41k ≈ 4.6GB）一次性占满显存。
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
        chunks = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            chunks.append((user_emb[batch] @ item_emb.T).cpu())
        scores = torch.cat(chunks, dim=0)
    return scores, test_users, test_pos


def compute_target_metrics(scores: torch.Tensor, user_ids: List[int],
                           clean_user_items: Dict[int, set],
                           target_items: List[int], k: int,
                           chunk_size: int = 1024) -> Dict[int, Dict[str, Any]]:
    """目标物品的攻击效果指标（HR@K / NDCG@K）。

    只统计"训练集中未交互过该目标物品"的合法用户（攻击的目标人群），
    用干净训练集过滤，保证 clean / poisoned 两次评估口径一致。

    返回 {target: {hr@k, ndcg@k, hit_users, mean_rank, mean_rank_all, n_elig}}
    - hr@k: 目标物品进入 Top-K 的用户比例（= 经典 Hit Rate @K）
    - ndcg@k: 单目标 NDCG（命中用户按排名位置折损，IDCG=1）
    - mean_rank: 命中用户的平均排名（未命中记 None；越小攻击越强）
    - mean_rank_all: 全体合格用户的平均排名（不受 Top-K 截断，更灵敏）
    - n_elig: 合格用户数（聚合时跳过 n_elig == 0 的目标）
    """
    # 显存安全：分数先回 CPU，避免 GPU 上整矩阵 topk/argsort 分配数 GB
    if scores.is_cuda:
        scores = scores.cpu()
    topk = torch.topk(scores, k, dim=1).indices  # (n_users, k)
    out: Dict[int, Dict[str, Any]] = {}
    for t in target_items:
        eligible = [
            r for r, uid in enumerate(user_ids)
            if t not in clean_user_items.get(uid, set())
        ]
        n_elig = len(eligible)
        if n_elig == 0:
            out[t] = {"hr@k": 0.0, "ndcg@k": 0.0, "hit_users": 0,
                      "mean_rank": None, "mean_rank_all": None, "n_elig": 0,
                      "exposure": 0.0, "ndcg": 0.0}
            continue

        hits = 0
        dcg = 0.0
        ranks: List[int] = []
        ranks_all_list: List[int] = []
        # mean_rank_all 按块做 argsort，避免整矩阵一次性分配
        for start in range(0, n_elig, chunk_size):
            rows = eligible[start:start + chunk_size]
            order = torch.argsort(scores[rows], dim=1, descending=True)
            for j, r in enumerate(rows):
                pos = (topk[r] == t).nonzero(as_tuple=False)
                if pos.numel():
                    rank = int(pos.item()) + 1
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 1)
                    ranks.append(rank)
                pos_all = (order[j] == t).nonzero(as_tuple=False)
                ranks_all_list.append(int(pos_all.item()) + 1)

        out[t] = {
            "hr@k": hits / n_elig,
            "ndcg@k": dcg / n_elig,
            "hit_users": hits,
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "mean_rank_all": float(np.mean(ranks_all_list)),
            "n_elig": n_elig,
            # 旧字段别名，兼容已有 JSON 消费者
            "exposure": hits / n_elig,
            "ndcg": dcg / n_elig,
        }
    return out


def aggregate_target_metrics(target_metrics: Dict[int, Dict[str, Any]],
                             target_items: List[int], k: int) -> Dict[str, float]:
    """把每个目标物品的 hr@k / ndcg@k 聚合为 checkpoint 选优指标。

    等权均值：跳过 n_elig == 0 的目标（避免把 0 拉低均值）；
    所有目标均无合格用户时返回 0.0。

    返回 {"target_hr@k": float, "target_ndcg@k": float}
    """
    eligible = [target_metrics[t] for t in target_items
                if t in target_metrics and target_metrics[t].get("n_elig", 0) > 0]
    if not eligible:
        return {f"target_hr@{k}": 0.0, f"target_ndcg@{k}": 0.0}
    return {
        f"target_hr@{k}": float(np.mean([m["hr@k"] for m in eligible])),
        f"target_ndcg@{k}": float(np.mean([m["ndcg@k"] for m in eligible])),
    }


def build_attack_eval_metrics(scores: torch.Tensor, user_ids: List[int],
                              user_items: Dict[int, set],
                              test_pos: Dict[int, set],
                              clean_user_items: Dict[int, set],
                              targets: List[int], ks: List[int],
                              metric_names: List[str]
                              ) -> Tuple[Dict[str, float], Dict[int, Dict[str, Any]]]:
    """训练中单次评估：整体指标 + （可选）目标指标，与配置指标名对齐。

    - 先算目标指标（避免被 compute_metrics 的 -inf 过滤污染排名）；
    - 整体指标：compute_metrics(scores, user_items, test_pos, k=K)；
    - 若 metric_names 含 target_ 前缀指标，对每个 K 追加 target_hr@K /
      target_ndcg@K；
    - 返回 (res, target_details)：
      res 为扁平 {指标名: 值}（BestTracker.update 直接消费）；
      target_details 为最大 K 下的 {target: {...}} 明细（写入 history）。
    """
    target_by_k: Dict[int, Dict[int, Dict[str, Any]]] = {}
    if any(name.startswith("target_") for name in metric_names):
        for K in ks:
            target_by_k[K] = compute_target_metrics(
                scores, user_ids, clean_user_items, targets, K)

    res_by_k: Dict[int, Dict[str, float]] = {
        K: compute_metrics(scores, user_items, test_pos, k=K) for K in ks
    }
    for K in ks:
        if K in target_by_k:
            res_by_k[K].update(aggregate_target_metrics(target_by_k[K], targets, K))

    target_details = target_by_k.get(max(ks), {}) if ks else {}
    return match_metric_values(metric_names, res_by_k), target_details


def compare_models(clean_model, poisoned_model, clean_meta: Dict[str, Any],
                   poisoned_meta: Dict[str, Any], target_items: List[int], k: int,
                   report_utility: bool = True) -> Dict[str, Any]:
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


def format_report(report: Dict[str, Any], title: str = "投毒攻击对比报告") -> str:
    """把对比结果格式化为 Markdown 报告；标题由参数控制。"""
    k = report["k"]
    lines = [
        f"# {title}（Top-{k}）",
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
            lines.append(f"| {key} | {cu[key]:.4f} | {pu[key]:.4f} | {delta:+.4f} |")

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

    # 结论段：攻击增益 + 投毒代价（通用文案，不再写攻击名专属措辞）
    lines += ["", "## 结论", ""]
    for t in ca:
        cr = ca[t]
        pr = pa[t]
        hr_delta = pr["hr@k"] - cr["hr@k"]
        exposure_trend = "投毒后目标物品曝光提升" if hr_delta > 0 else "投毒后目标物品曝光未提升"
        lines.append(
            f"- 目标物品 {t}：HR@{k} {cr['hr@k']:.4f} → {pr['hr@k']:.4f} "
            f"（{hr_delta:+.4f}），平均排名 {cr['mean_rank_all']:.1f} → "
            f"{pr['mean_rank_all']:.1f}；{exposure_trend}。"
        )
    if cu is not None:
        rec_keys = [key for key in cu if key.startswith("recall@")]
        if rec_keys:
            rec_key = rec_keys[0]
            util_delta = pu[rec_key] - cu[rec_key]
            trend = ("未下降（投毒代价可接受）" if util_delta >= -0.01
                     else "显著下降（投毒代价过大，需调低假用户比例）")
            lines.append(
                f"- 模型效用 {rec_key}：Clean {cu[rec_key]:.4f} → Poisoned "
                f"{pu[rec_key]:.4f}（{util_delta:+.4f}），{trend}。"
            )
    lines.append("")
    return "\n".join(lines)


def save_report(report: Dict[str, Any], out_dir: Path, name: str = "attack",
                title: str | None = None) -> Path:
    """写 {name}_comparison.md 与 {name}_comparison.json，返回 md 路径。

    name 按 REPORT_NAMES 归一化（保持各攻击现有文件名）；title 缺省时按
    REPORT_TITLES 或通用标题。
    """
    file_name = REPORT_NAMES.get(name, "attack")
    if title is None:
        title = REPORT_TITLES.get(name, "投毒攻击对比报告")
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{file_name}_comparison.md"
    md_path.write_text(format_report(report, title=title), encoding="utf-8")
    (out_dir / f"{file_name}_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return md_path
