# -*- coding: utf-8 -*-
"""Phase 11：统一 embedding 空间映射 P_m(·)。

为什么必须映射（问题定义）：
    每一次训练都会产生自己的坐标帧。实测：**同随机初始化**重训的物品行平均余弦
    0.956~0.975（可比），**不同初始化**只有 −0.002（不可比）。所以跨条件比较嵌入
    之前，必须把各条件映射到同一个参考帧。

映射方式（本实现用正交 Procrustes）：
    给定参考物品矩阵 A（原始模型）与条件物品矩阵 B，
        R = argmin_R ‖A − B·R‖_F ,  s.t. RᵀR = I
    闭式解：M = BᵀA = UΣVᵀ  →  R = U·Vᵀ。
    只用**干净训练集交互数 ≥ min_support** 的物品行拟合（弱行是噪声，会把 R 拉偏）。

两种口径都会输出（生产计划要求可复核）：
    raw        ：直接用条件模型自己的嵌入行（同初始化时帧已近似对齐）
    procrustes ：对齐后的嵌入行

注意（重要，避免误用）：
    只对**物品侧**做正交变换、不动用户侧，会破坏 U·Vᵀ 的自洽性，因此
    **对齐后的嵌入只能用于"嵌入距离"比较，不能拿去算曝光/排名**。
    本模块只服务距离分析，不提供曝光评估。

用法示例：
    A = load_item_matrix("lightgcn", cfg, clean_meta_path, ref_state_path, 42)
    B = load_item_matrix("lightgcn", cfg, cond_meta_path, cond_state_path, 42)
    R = procrustes(A, B, mask)
    z = B[item] @ R
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from pre.runners.common import build_model, build_training_config, load_meta


def load_item_matrix(model_name: str, cfg: Dict[str, Any], meta_path: Path,
                     state_path: Path) -> torch.Tensor:
    """重建条件模型并返回它的物品嵌入矩阵 (num_items, d)。

    为什么需要 meta：LightGCN 的邻接矩阵 A_hat 由 edge_index 现构，**不在
    state_dict 里**（实测 state_dict 只有 embedding.weight）。所以必须用当时
    那份 meta（干净/退化/中毒）重建图，才能得到与权重匹配的模型。
    """
    meta = load_meta(Path(meta_path))
    tcfg = build_training_config(cfg, model_name)
    edge_index = torch.LongTensor([[u, i] for u, i in meta["train_pairs"]]).T
    model = build_model(model_name, tcfg, meta["num_users"], meta["num_items"],
                        edge_index)
    model.load_state_dict(torch.load(state_path, map_location="cpu",
                                     weights_only=False))
    model.set_eval()
    return model.get_item_embeddings().detach().cpu().float()


def support_mask(clean_meta: Dict[str, Any], num_items: int,
                 min_support: int) -> torch.Tensor:
    """参与 Procrustes 拟合的物品掩码（按干净训练集交互数）。"""
    counts = torch.zeros(num_items, dtype=torch.long)
    for _, i in clean_meta["train_pairs"]:
        counts[i] += 1
    return counts >= int(min_support)


def procrustes(ref: torch.Tensor, cond: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """求正交矩阵 R，使 cond @ R 尽量接近 ref（只在 mask 行上拟合）。"""
    a = ref[mask]
    b = cond[mask]
    m = b.T @ a
    u, _, vt = torch.linalg.svd(m)
    return u @ vt


def pairwise_report(ref: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor,
                    item: int, R: torch.Tensor
                    ) -> Dict[str, float]:
    """记录对齐质量：平均行余弦（原始/对齐后）与拟合残差。"""
    def mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        an = a / (a.norm(dim=1, keepdim=True) + 1e-12)
        bn = b / (b.norm(dim=1, keepdim=True) + 1e-12)
        return float((an * bn).sum(dim=1).mean())

    cond_aligned = cond @ R
    resid = float((ref[mask] - cond_aligned[mask]).norm() /
                  (ref[mask].norm() + 1e-12))
    return {
        "mean_row_cos_raw": mean_cos(ref[mask], cond[mask]),
        "mean_row_cos_aligned": mean_cos(ref[mask], cond_aligned[mask]),
        "procrustes_residual": resid,
        "item_row_cos_raw": float(torch.dot(ref[item], cond[item]) /
                                  (ref[item].norm() * cond[item].norm() + 1e-12)),
        "item_row_cos_aligned": float(
            torch.dot(ref[item], cond_aligned[item]) /
            (ref[item].norm() * cond_aligned[item].norm() + 1e-12)),
    }
