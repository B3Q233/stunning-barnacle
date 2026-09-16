# -*- coding: utf-8 -*-
"""Phase 1：目标物品选择（一次确定，全实验复用）。

为什么必须先选后跑（设计动机）：
    实验矩阵是 模型 × 攻击 × 删除比例 × 目标物品。如果目标集合由每个模型各自挑选，
    就会出现"LightGCN 选了一批、MF 选了另一批"，跨模型比较直接失效。因此这里把
    目标集合固化成 targets.json，后续所有阶段只读不选。

两种策略（v2 新增第二种）：
    top_popularity：按**干净训练集**交互数取前 K 个。
        优点：不需要任何模型，纯数据可得。
        实测（ml100k + LightGCN）它与 top_exposure 的排名 Spearman ≈ 0.997，
        前 5 名完全一致——但这是数据/骨干相关的事实，不能当普遍规律。
    top_exposure：按**原始模型上的原生 HR@K**取前 K 个（"受欢迎 = 在原始嵌入上
        最容易被推荐给未交互用户"）。需要一个参考模型（pre.targets.exposure_source_model），
        因此 Phase 1 会先训练/复用一份原始模型。这是更贴合"受欢迎"本义的定义。

用法示例：
    python pre/run.py --mode targets        # 按 config 的 strategy 选择并落盘
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pre.runners.common import (item_counts, now_iso, read_json, save_json)


def select_targets_popularity(meta: Dict[str, Any], num_items: int,
                              min_interactions: int, seed: int) -> List[Tuple[int, int, float]]:
    """按交互数降序取前 K 个，返回 [(item_id, interactions, hr_or_nan), ...]。"""
    counts = item_counts(meta["train_pairs"])
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = [(i, c, float("nan")) for i, c in ranked if c >= min_interactions]
    if len(picked) < num_items:
        raise ValueError(
            f"满足 min_interactions={min_interactions} 的物品只有 {len(picked)} 个，"
            f"少于 num_items={num_items}；请放宽门槛或减少目标数")
    return picked[:num_items]


def build_targets(picked: List[Tuple[int, int, float]], *, strategy: str,
                  dataset: str, seed: int, min_interactions: int,
                  k: int, source_model: Optional[str] = None,
                  exposure_metric: Optional[str] = None) -> Dict[str, Any]:
    """把 [(item_id, interactions, hr), ...] 组装成可落盘的 targets 结构。"""
    items = []
    for rank, (item_id, cnt, hr) in enumerate(picked, start=1):
        entry: Dict[str, Any] = {"item_id": int(item_id),
                                 "original_interactions": int(cnt),
                                 "popularity_rank": rank}
        if strategy == "top_exposure":
            entry["native_hr@k"] = float(hr)
            entry["exposure_rank"] = rank
        items.append(entry)
    return {
        "dataset": dataset,
        "selection_seed": seed,
        "strategy": strategy,
        "min_interactions": min_interactions,
        "k": k,
        "exposure_source_model": source_model if strategy == "top_exposure" else None,
        "exposure_metric": exposure_metric if strategy == "top_exposure" else None,
        "num_targets": len(items),
        "created_at": now_iso(),
        "items": items,
    }


def load_or_build_targets(cfg: Dict[str, Any], exp_dir: Path,
                          clean_meta: Dict[str, Any],
                          exposure: Optional[Dict[int, Dict[str, float]]] = None
                          ) -> Dict[str, Any]:
    """读缓存 targets.json；不存在则生成。

    strategy=top_exposure 时调用方必须先给出 exposure（由 pipeline 训练/复用
    参考模型后算出），否则直接报错而不是静默退化成按交互数选。
    """
    path = exp_dir / "targets.json"
    if path.exists():
        return read_json(path)
    p = cfg.get("pre", {}).get("targets", {})
    strategy = str(p.get("strategy", "top_popularity"))
    num_items = int(p.get("num_items", 20))
    min_int = int(p.get("min_interactions", 20))
    seed = int(p.get("seed", cfg.get("experiment", {}).get("seed", 42)))
    k = int(cfg.get("k", 10))

    if strategy == "top_popularity":
        picked = select_targets_popularity(clean_meta, num_items, min_int, seed)
        src = None
        metric = None
    elif strategy == "top_exposure":
        if exposure is None:
            raise ValueError(
                "strategy=top_exposure 需要先提供 exposure（由 pipeline 训练"
                "参考模型后计算）；请通过 phase_targets 调用，不要直接调用本函数")
        from pre.analysis.exposure import select_targets_by_exposure
        counts = item_counts(clean_meta["train_pairs"])
        picked = select_targets_by_exposure(exposure, counts, num_items, min_int)
        src = str(p.get("exposure_source_model", "lightgcn"))
        metric = f"hr@{k}"
    else:
        raise ValueError(f"未知 targets.strategy={strategy!r}，"
                         f"可选 top_popularity | top_exposure")

    t = build_targets(picked, strategy=strategy, dataset=str(cfg.get("dataset")),
                      seed=seed, min_interactions=min_int, k=k,
                      source_model=src, exposure_metric=metric)
    save_json(t, path)
    return t
