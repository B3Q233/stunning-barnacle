# -*- coding: utf-8 -*-
"""Phase 3：受控交互退化（删除目标物品自身的真实交互，并缓存删除结果）。

定义（严格）：
    对目标物品 i 与删除比例 r，删除 floor(r × |I_i|) 条**该物品的**真实交互，
    其余物品与用户完全不动。不是删除整个数据集的比例。

**删除必须嵌套（v2 修正）**：
    早期实现对每个比例用独立种子抽样，导致 r=0.9 删掉的集合**不包含** r=0.1 删掉的
    集合，"删除比例"这条轴失去阶梯含义（同一物品在不同比例下看到的退化数据互不相干）。
    现在改为：对每个物品抽**一次**随机排列（种子只由 (seed, item) 决定），
    比例 r 删除该排列的前 floor(r·|I_i|) 个元素。于是
        D(0.1) ⊆ D(0.2) ⊆ D(0.3) ⊆ D(0.5) ⊆ D(0.8) ⊆ D(0.9)
    删除集合天然嵌套，比例轴单调可解释。

为什么必须缓存删除结果：
    删除是随机的。如果每次跑都重新随机，"同一条件下不同模型/攻击看到的数据不同"，
    跨模型比较立刻失去意义。这里把被删掉的交互对落盘到 deleted_interactions.json，
    并用确定性种子（stable_seed 基于 md5，跨进程稳定），任何人任何时候都能复现。

产物（每个 (model, item, ratio) 一份）：
    degraded/<model>/item_<id>/ratio_<pp>/deleted_interactions.json
    degraded/<model>/item_<id>/ratio_<pp>/train_data/meta.pkl

用法示例：
    meta_r, info = build_degraded_meta(clean_meta, item_id=402, ratio=0.5, seed=42)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pre.runners.common import (item_interactions, read_json, save_json,
                                save_meta, stable_seed, user_items_of)


def ratio_dirname(ratio: float) -> str:
    """比例目录名：0.1 -> 'ratio_10'（与生产计划的目录约定一致）。"""
    return f"ratio_{int(round(ratio * 100)):02d}"


def item_deletion_order(clean_meta: Dict[str, Any], item_id: int,
                        seed: int) -> List[Tuple[int, int]]:
    """该物品交互的一次随机排列（种子只由 seed 与 item 决定）。

    嵌套删除的基础：任何比例都取这个排列的前缀，所以小比例集合必然是大比例集合的子集。
    """
    own = item_interactions(clean_meta["train_pairs"], item_id)
    rng = random.Random(stable_seed(seed, item_id))
    idx = list(range(len(own)))
    rng.shuffle(idx)
    return [own[k] for k in idx]


def build_degraded_meta(clean_meta: Dict[str, Any], item_id: int, ratio: float,
                        seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """构造退化 meta 与删除记录（纯函数，确定性，且跨比例嵌套）。

    返回 (degraded_meta, info)：
        degraded_meta：num_users/num_items/test_pairs/user_items 与干净数据一致，
                       train_pairs 去掉了被删的 n_del 条目标物品交互
        info：item_id / ratio / original/remaining/deleted 计数 / 被删交互明细
    """
    pairs = list(clean_meta["train_pairs"])
    order = item_deletion_order(clean_meta, item_id, seed)
    n_original = len(order)
    n_delete = int(n_original * ratio)          # floor，按定义向下取整
    deleted: List[Tuple[int, int]] = order[:n_delete] if n_delete > 0 else []
    deleted_set = set(deleted)
    kept_pairs = [p for p in pairs if p not in deleted_set]

    new_meta = dict(clean_meta)
    new_meta["train_pairs"] = kept_pairs
    new_meta["user_items"] = user_items_of(kept_pairs)
    info = {
        "item_id": int(item_id),
        "deletion_ratio": float(ratio),
        "original_interaction_count": int(n_original),
        "deleted_interaction_count": int(len(deleted)),
        "remaining_interaction_count": int(n_original - len(deleted)),
        "deleted_interactions": [[int(u), int(i)] for u, i in deleted],
        "deletion_scheme": "nested",
        "delete_seed": stable_seed(seed, item_id),
    }
    return new_meta, info


def build_and_cache_degraded(cfg: Dict[str, Any], exp_dir: Path,
                             model_name: str, clean_meta: Dict[str, Any],
                             item_id: int, ratio: float, seed: int,
                             reuse: bool = True) -> Dict[str, Any]:
    """构造并缓存退化数据；缓存命中时直接读取（不重新随机删除）。"""
    base = (exp_dir / "degraded" / model_name / f"item_{item_id:05d}"
            / ratio_dirname(ratio))
    meta_path = base / "train_data" / "meta.pkl"
    info_path = base / "deleted_interactions.json"
    if reuse and meta_path.exists() and info_path.exists():
        info = read_json(info_path)
        info["meta_path"] = str(meta_path)
        return info
    degraded_meta, info = build_degraded_meta(clean_meta, item_id, ratio, seed)
    save_meta(degraded_meta, meta_path)
    info["meta_path"] = str(meta_path)
    save_json(info, info_path)
    return info
