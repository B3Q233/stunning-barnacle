# -*- coding: utf-8 -*-
"""Phase 2/4/5：训练受害模型并缓存权重、嵌入与元数据。

两类缓存（v2 修正后）：
    1) **共享原始模型**（每个受害模型只训一次）
       路径：original/<model>/clean/
       内容：model.pt / checkpoint.pt / embeddings.pt（全量物品矩阵）/ metadata.json
       为什么改：原始模型与目标物品无关，早期实现按 (model, item) 各训一次，
       K 个目标就白训 K-1 次（K=20 时浪费 38 次训练）。现在一个模型只有一份。

    2) **条件模型**（degraded / attacked）
       路径：<group>/<model>/item_<id>/<label>/
       内容：model.pt / checkpoint.pt / embeddings.pt（只存目标物品行）/ metadata.json
       条件之间只差数据，所以必须逐条件训练。

统一口径：训练入口是 common.train_victim（固定种子 + 同初始化 + 固定轮数），
三处调用只在传入的 meta 上不同。
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from pre.runners.common import (git_commit, item_embedding, item_matrix, now_iso,
                                read_json, save_json, train_victim)


def _cond_dir(exp_dir: Path, model_name: str, group: str, item_id: int,
              label: str) -> Path:
    """条件目录：<group>/<model>/item_<id>/<label>/。"""
    return exp_dir / group / model_name / f"item_{item_id:05d}" / label


def _ckpt_payload(cfg: Dict[str, Any], model) -> Dict[str, Any]:
    """仓库格式 checkpoint（攻击模块的 checkpoint.clean 需要这个结构）。"""
    return {"epoch": int(cfg.get("pre", {}).get("training", {}).get("epochs", 30)),
            "model_state_dict": model.state_dict()}


def train_original(cfg: Dict[str, Any], exp_dir: Path, model_name: str,
                   meta: Dict[str, Any], seed: int, reuse: bool = True
                   ) -> Dict[str, Any]:
    """训练（或复用）**共享原始模型**：完整训练集，得到参考嵌入 z^0。

    embeddings.pt 存**全量物品矩阵**——它同时是 Procrustes 的参考系，
    分析阶段不用再重建模型。
    """
    base = exp_dir / "original" / model_name / "clean"
    base.mkdir(parents=True, exist_ok=True)
    meta_json = base / "metadata.json"
    emb_pt = base / "embeddings.pt"
    if reuse and meta_json.exists() and emb_pt.exists():
        rec = read_json(meta_json)
        rec["cached"] = True
        return rec

    t0 = time.time()
    model, history = train_victim(model_name, cfg, meta, seed)
    V = item_matrix(model)
    U = model.get_user_embeddings().detach().cpu().float()
    torch.save(model.state_dict(), base / "model.pt")
    torch.save(_ckpt_payload(cfg, model), base / "checkpoint.pt")
    torch.save({"item_factors": V, "user_factors": U}, emb_pt)
    rec = {
        "cached": False, "kind": "original_shared", "model_name": model_name,
        "item_id": None, "seed": int(seed),
        "num_users": int(meta["num_users"]), "num_items": int(meta["num_items"]),
        "train_pairs": int(len(meta["train_pairs"])),
        "model_config": copy.deepcopy(cfg.get("pre", {}).get("training", {})),
        "model_path": str(base / "model.pt"),
        "checkpoint_path": str(base / "checkpoint.pt"),
        "embedding_path": str(emb_pt),
        "history": history, "seconds": round(time.time() - t0, 1),
        "git_commit": git_commit(), "created_at": now_iso(),
    }
    save_json(rec, meta_json)
    return rec


def train_and_cache(cfg: Dict[str, Any], exp_dir: Path, model_name: str,
                    meta: Dict[str, Any], item_id: int, group: str,
                    label: str, seed: int, reuse: bool = True,
                    extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """训练（或复用）一个条件模型（degraded / attacked）。

    group: "degraded" | "attacks"
    label: 如 "ratio_50"、"pgd_ratio_50"
    """
    base = _cond_dir(exp_dir, model_name, group, item_id, label)
    base.mkdir(parents=True, exist_ok=True)
    meta_json = base / "metadata.json"
    emb_pt = base / "embeddings.pt"
    if reuse and meta_json.exists() and emb_pt.exists():
        rec = read_json(meta_json)
        rec["cached"] = True
        return rec

    t0 = time.time()
    model, history = train_victim(model_name, cfg, meta, seed)
    e = item_embedding(model, item_id)
    torch.save(model.state_dict(), base / "model.pt")
    torch.save(_ckpt_payload(cfg, model), base / "checkpoint.pt")
    torch.save({f"item_{item_id}": e}, emb_pt)

    rec: Dict[str, Any] = {
        "cached": False, "kind": "condition", "group": group, "label": label,
        "model_name": model_name, "item_id": int(item_id), "seed": int(seed),
        "num_users": int(meta["num_users"]), "num_items": int(meta["num_items"]),
        "train_pairs": int(len(meta["train_pairs"])),
        "model_config": copy.deepcopy(cfg.get("pre", {}).get("training", {})),
        "model_path": str(base / "model.pt"),
        "checkpoint_path": str(base / "checkpoint.pt"),
        "embedding_path": str(emb_pt),
        "embedding_norm": float(e.norm()),
        "history": history, "seconds": round(time.time() - t0, 1),
        "git_commit": git_commit(), "created_at": now_iso(),
    }
    if extra_meta:
        rec.update(extra_meta)
    save_json(rec, meta_json)
    return rec


def load_original_record(exp_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
    """读共享原始模型记录；不存在返回 None（调用方负责给可执行的提示）。"""
    p = exp_dir / "original" / model_name / "clean" / "metadata.json"
    return read_json(p) if p.exists() else None


def original_target_embedding(exp_dir: Path, model_name: str, item_id: int
                              ) -> Optional[torch.Tensor]:
    """从共享原始模型的全量矩阵里取某个目标物品的 z^0。"""
    p = exp_dir / "original" / model_name / "clean" / "embeddings.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["item_factors"][item_id]
