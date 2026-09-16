# -*- coding: utf-8 -*-
"""Phase 12：汇总所有条件，计算距离与恢复率，输出标准化 CSV/JSON。

v2 修正（重要）：
    ① **主口径改为 L2（绝对距离）**。余弦忽略模长，而退化主要表现为行范数缩小
       （LightGCN 上 3.8 → 0.7），攻击注入交互会把范数拉回来——于是绝对距离改善、
       方向未必改善。实测两者结论相反（L2 正恢复 91/600 vs 余弦 26/600），
       所以两个都报，且用 pre.embedding.distance 指定主口径。
    ② 新增 **范数恢复率** norm_recovery = (‖z_att‖ − ‖z_deg‖) / (‖z0‖ − ‖z_deg‖)，
       把"绝对距离改善"拆成"范数补回来多少"与"方向对不对"两部分。
    ③ 原始模型改为共享缓存 original/<model>/clean/（全量物品矩阵），
       参考系不再逐物品重复训练。

输出（<exp>/results/）：
    embedding_distances.csv  逐 (model,item,ratio,attack,alignment)
    recovery_metrics.csv     按 (model,attack,ratio,alignment) 对目标物品求均值
    summary.json
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pre.runners.common import load_meta, read_json, save_json
from pre.analysis.align import (load_item_matrix, pairwise_report, procrustes,
                                support_mask)
from pre.analysis.distances import distances, recovery_rate
from pre.runners.degrade import ratio_dirname
from pre.runners.pipeline import clean_meta_path


def _cond_dir(exp_dir: Path, model_name: str, group: str, item_id: int,
              label: str) -> Path:
    return exp_dir / group / model_name / f"item_{item_id:05d}" / label


def analyze(cfg: Dict[str, Any], exp_dir: Path, targets: Dict[str, Any],
            attacks: Optional[List[str]] = None) -> Dict[str, Any]:
    """遍历缓存，产出距离/恢复率表。"""
    pre = cfg.get("pre", {})
    ratios = [float(r) for r in pre.get("deletion_ratios", [])]
    min_support = int(pre.get("embedding", {}).get("min_support", 5))
    primary = str(pre.get("embedding", {}).get("distance", "l2"))
    attacks = attacks or list(pre.get("attacks", []))
    rows: List[Dict[str, Any]] = []

    for model_name in pre.get("models", []):
        clean = load_meta(clean_meta_path(model_name, cfg["dataset"]))
        num_items = int(clean["num_items"])
        mask = support_mask(clean, num_items, min_support)
        ref_dir = exp_dir / "original" / model_name / "clean"
        if not (ref_dir / "model.pt").exists():
            continue
        # 参考系：共享原始模型直接读全量物品矩阵（无需重建）
        ref = torch.load(ref_dir / "embeddings.pt",
                         map_location="cpu", weights_only=False)["item_factors"]
        for t in targets["items"]:
            item = int(t["item_id"])
            z0 = ref[item]
            for ratio in ratios:
                label = ratio_dirname(ratio)
                d_dir = _cond_dir(exp_dir, model_name, "degraded", item, label)
                if not (d_dir / "model.pt").exists():
                    continue
                drec = read_json(d_dir / "metadata.json")
                dmeta_path = drec.get("meta_path")
                if not dmeta_path:
                    continue
                B = load_item_matrix(model_name, cfg, Path(dmeta_path),
                                     d_dir / "model.pt")
                R = procrustes(ref, B, mask)
                zb_raw, zb_al = B[item], (B @ R)[item]
                d_deg_raw, d_deg_al = distances(zb_raw, z0), distances(zb_al, z0)
                diag = pairwise_report(ref, B, mask, item, R)
                for attack_name in ["__none__"] + list(attacks):
                    if attack_name == "__none__":
                        z_att_raw, z_att_al = zb_raw, zb_al
                        att_meta_path, fake = None, 0
                    else:
                        a_dir = (exp_dir / "attacks" / model_name /
                                 f"item_{item:05d}" / f"{attack_name}_{label}")
                        rec_path = a_dir / "attack.json"
                        if not rec_path.exists():
                            continue
                        arec = read_json(rec_path)
                        if arec.get("status") != "ok":
                            continue
                        ameta_json = a_dir / "metadata.json"
                        if not (a_dir / "model.pt").exists() or not ameta_json.exists():
                            continue
                        amrec = read_json(ameta_json)
                        C = load_item_matrix(model_name, cfg,
                                             Path(arec["poisoned_meta"]),
                                             a_dir / "model.pt")
                        Rc = procrustes(ref, C, mask)
                        z_att_raw, z_att_al = C[item], (C @ Rc)[item]
                        att_meta_path = arec["poisoned_meta"]
                        fake = int(amrec.get("fake_users_added", 0))
                    d_att_raw, d_att_al = distances(z_att_raw, z0), distances(z_att_al, z0)
                    for alignment, d_deg, d_att, z_a in (
                            ("raw", d_deg_raw, d_att_raw, z_att_raw),
                            ("procrustes", d_deg_al, d_att_al, z_att_al)):
                        n0, nd, na = float(z0.norm()), float(zb_raw.norm()), float(z_a.norm())
                        gap = n0 - nd
                        rows.append({
                            # ── 主口径（由 pre.embedding.distance 指定）排在最前 ──
                            "primary_metric": primary,
                            "recovery_rate": recovery_rate(d_deg[primary], d_att[primary]),
                            "degraded_distance": d_deg[primary],
                            "attacked_distance": d_att[primary],
                            # ── 两种距离都保留 ──
                            "recovery_rate_l2": recovery_rate(d_deg["l2"], d_att["l2"]),
                            "recovery_rate_cos": recovery_rate(d_deg["cosine"],
                                                               d_att["cosine"]),
                            "degraded_distance_l2": d_deg["l2"],
                            "attacked_distance_l2": d_att["l2"],
                            "degraded_distance_cos": d_deg["cosine"],
                            "attacked_distance_cos": d_att["cosine"],
                            # ── 范数分解：绝对距离的改善有多少来自"模长补回来" ──
                            "norm_recovery": ((na - nd) / gap) if abs(gap) > 1e-9 else 0.0,
                            "norm_original": n0, "norm_degraded": nd, "norm_attacked": na,
                            # ── 标识与审计 ──
                            "model": model_name, "attack": attack_name,
                            "item_id": item, "deletion_ratio": ratio,
                            "alignment": alignment,
                            "fake_users_added": fake,
                            "original_interactions": drec.get("original_interaction_count"),
                            "remaining_interactions": drec.get("remaining_interaction_count"),
                            "deletion_scheme": drec.get("deletion_scheme"),
                            "mean_row_cos_raw": diag["mean_row_cos_raw"],
                            "mean_row_cos_aligned": diag["mean_row_cos_aligned"],
                            "procrustes_residual": diag["procrustes_residual"],
                            "poisoned_meta": att_meta_path,
                        })
    return _write_outputs(exp_dir, rows, cfg, targets)


def _write_outputs(exp_dir: Path, rows: List[Dict[str, Any]],
                   cfg: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = exp_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["model", "attack", "item_id"]
    csv_path = out_dir / "embedding_distances.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    agg: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["model"], r["attack"], r["deletion_ratio"], r["alignment"])
        agg.setdefault(key, []).append(r)
    agg_rows = []
    for (model, attack, ratio, alignment), rs in sorted(agg.items()):
        n = len(rs)
        m = lambda k: sum(float(r[k]) for r in rs) / n          # noqa: E731
        agg_rows.append({
            "model": model, "attack": attack, "deletion_ratio": ratio,
            "alignment": alignment, "num_items": n,
            "recovery_rate": m("recovery_rate"),
            "degraded_distance": m("degraded_distance"),
            "attacked_distance": m("attacked_distance"),
            "recovery_rate_l2": m("recovery_rate_l2"),
            "recovery_rate_cos": m("recovery_rate_cos"),
            "norm_recovery": m("norm_recovery"),
            "norm_original": m("norm_original"),
            "norm_degraded": m("norm_degraded"),
            "norm_attacked": m("norm_attacked"),
        })
    agg_path = out_dir / "recovery_metrics.csv"
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows
                           else ["model", "attack", "deletion_ratio"])
        w.writeheader()
        w.writerows(agg_rows)

    pos = [r for r in rows if r["attack"] != "__none__" and
           float(r["recovery_rate"]) > 0]
    summary = {
        "num_conditions": len(rows),
        "primary_metric": cfg.get("pre", {}).get("embedding", {}).get("distance", "l2"),
        "deletion_scheme": cfg.get("pre", {}).get("deletion_scheme", "nested"),
        "target_strategy": targets.get("strategy"),
        "positive_recovery_conditions": len(pos),
        "attack_conditions": sum(1 for r in rows if r["attack"] != "__none__"),
        "models": cfg.get("pre", {}).get("models"),
        "attacks": cfg.get("pre", {}).get("attacks"),
        "deletion_ratios": cfg.get("pre", {}).get("deletion_ratios"),
        "num_targets": targets.get("num_targets"),
        "files": {"embedding_distances": str(csv_path),
                  "recovery_metrics": str(agg_path)},
    }
    save_json(summary, out_dir / "summary.json")
    return {"rows": rows, "aggregated": agg_rows, **summary}
