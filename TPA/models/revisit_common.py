"""AdvInject victim model 共用的数据、排序与训练辅助。"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """正样本 pair 数据集，训练模型负责负采样。"""

    def __init__(self, pairs: Sequence[tuple[int, int]]):
        self.users = torch.tensor([u for u, _ in pairs], dtype=torch.long)
        self.items = torch.tensor([i for _, i in pairs], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.users.numel())

    def __getitem__(self, index: int):
        return self.users[index], self.items[index]


def validate_pairs(pairs: Iterable[tuple[int, int]], num_users: int,
                   num_items: int) -> list[tuple[int, int]]:
    result = []
    for user, item in pairs:
        user = int(user)
        item = int(item)
        if not 0 <= user < num_users:
            raise ValueError(f"用户 ID 越界: {user}, num_users={num_users}")
        if not 0 <= item < num_items:
            raise ValueError(f"物品 ID 越界: {item}, num_items={num_items}")
        result.append((user, item))
    return result


def build_user_items(pairs: Iterable[tuple[int, int]], num_users: int) -> list[set[int]]:
    result = [set() for _ in range(num_users)]
    for user, item in pairs:
        if not 0 <= int(user) < num_users:
            raise ValueError(f"用户 ID 越界: {user}, num_users={num_users}")
        result[int(user)].add(int(item))
    return result


def build_item_users(pairs: Iterable[tuple[int, int]], num_items: int) -> list[set[int]]:
    result = [set() for _ in range(num_items)]
    for user, item in pairs:
        if not 0 <= int(item) < num_items:
            raise ValueError(f"物品 ID 越界: {item}, num_items={num_items}")
        result[int(item)].add(int(user))
    return result


def build_train_mask(user_items: Sequence[set[int]], user_ids: Sequence[int],
                     num_items: int) -> np.ndarray:
    mask = np.zeros((len(user_ids), num_items), dtype=bool)
    for row, user in enumerate(user_ids):
        if not 0 <= int(user) < len(user_items):
            raise ValueError(f"用户 ID 越界: {user}")
        items = [i for i in user_items[int(user)] if 0 <= i < num_items]
        mask[row, items] = True
    return mask


def compute_ranking_metrics(scores: np.ndarray,
                            test_items: Sequence[set[int]],
                            k: int) -> dict[str, float]:
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != len(test_items):
        raise ValueError("scores 与 test_items 的用户维度不一致")
    k = max(1, min(int(k), scores.shape[1]))
    recalls = []
    ndcgs = []
    for row, targets in enumerate(test_items):
        order = np.argsort(-scores[row], kind="stable")[:k]
        hits = [idx for idx, item in enumerate(order) if int(item) in targets]
        recalls.append(len(hits) / max(1, len(targets)))
        dcg = sum(1.0 / math.log2(pos + 2) for pos in hits)
        ideal_hits = min(len(targets), k)
        idcg = sum(1.0 / math.log2(pos + 2) for pos in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {f"recall@{k}": float(np.mean(recalls)),
            f"ndcg@{k}": float(np.mean(ndcgs))}


def load_meta(config: dict, model_name: str) -> dict:
    configured = config.get("data", {}).get("processed_data_path")
    if configured:
        path = Path(configured)
        if path.is_dir():
            path = path / "meta.pkl"
    else:
        data_cfg = config.get("data", {})
        dataset = config.get("dataset") or data_cfg.get("dataset", "ml100k")
        source_model = data_cfg.get("source_model", model_name)
        path = (Path(__file__).resolve().parent / source_model / "data" /
                "processed" / dataset / "meta.pkl")
    if not path.exists():
        raise FileNotFoundError(f"找不到处理后数据: {path}")
    with path.open("rb") as handle:
        meta = pickle.load(handle)
    required = {"num_users", "num_items", "train_pairs", "test_pairs"}
    missing = required.difference(meta)
    if missing:
        raise KeyError(f"meta.pkl 缺少字段: {sorted(missing)}")
    meta["train_pairs"] = validate_pairs(meta["train_pairs"], meta["num_users"], meta["num_items"])
    meta["test_pairs"] = validate_pairs(meta["test_pairs"], meta["num_users"], meta["num_items"])
    meta.setdefault("user_items", build_user_items(meta["train_pairs"], meta["num_users"]))
    return meta

def ensure_training_config(config):
    """? YAML ???????????? TrainingConfig?"""
    from training.framework import TrainingConfig
    if isinstance(config, TrainingConfig):
        return config
    flat = {}
    flat.update(config.get("model", {}))
    flat.update(config.get("training", {}))
    flat.update(config.get("evaluation", {}))
    return TrainingConfig(overrides=flat)
