# -*- coding: utf-8 -*-
"""距离与恢复率定义（Phase 11/12 的核心量）。

定义（与生产计划一致）：
    z^0     原始模型目标物品嵌入（参考）
    z^{r}   删掉比例 r 的交互后、未攻击的嵌入
    z^{A,r} 删掉 r 后、被攻击 A 推到的嵌入

    d_deg   = d(z^{r},   z^0)      退化造成的距离
    d_att   = d(z^{A,r}, z^0)      攻击后的距离
    RR      = (d_deg − d_att) / d_deg       恢复率

    RR > 0  ：攻击把物品推回原始表示（攻击"修复"了退化）
    RR ≈ 0  ：攻击没有改变退化状态
    RR < 0  ：攻击把物品推得更远（攻击方向与原始表示相反）

距离度量：cosine = 1 − cos(u, v)；l2 = ‖u − v‖₂。两者都会写入 CSV，
主口径由配置 pre.embedding.distance 决定（默认 cosine）。
"""
from __future__ import annotations

from typing import Dict

import torch


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm())


def distances(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    return {"cosine": cosine_distance(a, b), "l2": l2_distance(a, b)}


def recovery_rate(d_deg: float, d_att: float) -> float:
    """恢复率；d_deg 为 0（未删除任何交互）时定义为 0.0。"""
    if d_deg <= 1e-12:
        return 0.0
    return float((d_deg - d_att) / d_deg)
