"""WMF 数据导入模块（步骤②）

实现 DatasetProtocol 五个方法：
- train_loader：返回【整个训练矩阵】的单批 batch（论文 ALS 是全量闭式优化，
  一个 epoch = 一轮完整交替最小二乘，不存在 mini-batch；batch 内附带预构建的
  user_obs / item_obs 观测分组，避免每轮重建）
- val_loader / test_loader：单批全量（仅用于损失监控，无需分组）
- get_init_params：num_users / num_items（模型初始化用）
- get_dataset：按 split 返回 WMFDataset

依据理解文档「模块一·1.4」：p_ui = 1 if r_ui > 0 else 0；置信度
c_ui = 1 + α·r_ui（minimal，论文 Eq.2/3）或 1 + α·log(1 + r_ui/ε)
（log-scaling）。本地 ml100k 为成对格式、无评分值，r_ui 恒为 1。
"""
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset

from models.wmf.config_keys import (
    KEY_ALPHA,
    KEY_CONFIDENCE_SCHEME,
    KEY_DATASET,
    KEY_EPSILON,
    KEY_NUM_ITEMS,
    KEY_NUM_USERS,
    KEY_PROCESSED_DATA_PATH,
    KEY_VAL_RATIO,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA 项目根
DEFAULT_VAL_RATIO = 0.05
SPLIT_SEED = 42


def group_observations(users: List[int], items: List[int]):
    """按用户/物品分组观测下标，返回 (user_obs, item_obs) 两个 dict。

    user_obs[u] = [观测 j, ...]（用户 u 的所有交互在训练矩阵中的位置）。
    分组只依赖 (users, items)，训练过程中固定不变，因此只在数据导入时
    构建一次，ALS 每轮直接引用，不做重复扫描。
    """
    user_obs = defaultdict(list)
    item_obs = defaultdict(list)
    for j, (u, i) in enumerate(zip(users, items)):
        user_obs[u].append(j)
        item_obs[i].append(j)
    return user_obs, item_obs


def _confidence(alpha: float, epsilon: float, scheme: str, r: float) -> float:
    """论文置信度公式：c_ui = 1 + α·r（minimal）或 1 + α·log(1 + r/ε)。"""
    if scheme == "minimal":
        return 1.0 + alpha * r
    if scheme == "log-scaling":
        return 1.0 + alpha * np.log1p(r / epsilon)
    raise ValueError(
        f"不支持的 confidence_scheme='{scheme}'，可选 minimal | log-scaling"
    )


class WMFDataset(Dataset):
    """WMF 观测数据集：每行一条 (user, item, confidence, p)。

    论文 Eq.(2)：p_ui = 1 if r_ui > 0 else 0（本地数据恒为 1）；
    Eq.(3)：c_ui 为置信度（见 _confidence）。
    """

    def __init__(self, pairs, alpha: float = 40.0, epsilon: float = 1e-8,
                 scheme: str = "minimal"):
        self.pairs = list(pairs)
        users = [u for u, _ in self.pairs]
        items = [i for _, i in self.pairs]
        r = 1.0  # 本地成对数据无评分值，r_ui 恒为 1
        c = _confidence(alpha, epsilon, scheme, r)

        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.conf = torch.full((len(self.pairs),), float(c),
                               dtype=torch.float32)
        self.p = torch.ones(len(self.pairs), dtype=torch.float32)
        # 预构建观测分组：ALS 每轮直接引用，不重复扫描（审阅②）
        self.user_obs, self.item_obs = group_observations(users, items)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.conf[idx], self.p[idx]


class WMFDataLoader:
    """WMF 数据加载器：实现 DatasetProtocol 五个方法。"""

    def __init__(self, config):
        self.config = config
        dataset_name = config.get(KEY_DATASET, "ml100k")
        processed_cfg = config.get(KEY_PROCESSED_DATA_PATH)
        if processed_cfg:
            meta_path = Path(processed_cfg)
            if not meta_path.is_absolute():
                meta_path = PROJECT_ROOT / meta_path
            meta_path = meta_path / "meta.pkl"
        else:
            meta_path = (
                PROJECT_ROOT / "models" / "wmf" / "data" / "processed"
                / dataset_name / "meta.pkl"
            )
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.num_users = meta["num_users"]
        self.num_items = meta["num_items"]
        self.train_pairs = meta["train_pairs"]
        self.test_pairs = meta["test_pairs"]
        self.user_items = meta["user_items"]
        self.test_user_items = {}
        for u, i in self.test_pairs:
            self.test_user_items.setdefault(u, set()).add(i)

        self.alpha = float(config.get(KEY_ALPHA, 40.0))
        self.epsilon = float(config.get(KEY_EPSILON, 1e-8))
        self.scheme = config.get(KEY_CONFIDENCE_SCHEME, "minimal")
        val_ratio = float(config.get(KEY_VAL_RATIO, DEFAULT_VAL_RATIO))

        random.seed(SPLIT_SEED)
        shuffled = self.train_pairs.copy()
        random.shuffle(shuffled)
        split = int(len(shuffled) * (1.0 - val_ratio))
        self._train_pairs = shuffled[:split]
        self._val_pairs = shuffled[split:]

        print(f"[WMFDataLoader] {dataset_name}: users={self.num_users}, "
              f"items={self.num_items}, train={len(self._train_pairs)}, "
              f"val={len(self._val_pairs)}, test={len(self.test_pairs)}, "
              f"scheme={self.scheme}, alpha={self.alpha}")

    def _dataset(self, pairs) -> WMFDataset:
        return WMFDataset(pairs, alpha=self.alpha, epsilon=self.epsilon,
                          scheme=self.scheme)

    def _full_loader(self, dataset: WMFDataset) -> TorchDataLoader:
        return TorchDataLoader(
            dataset, batch_size=len(dataset), shuffle=False,
            num_workers=0, pin_memory=False,
        )

    def train_loader(self) -> TorchDataLoader:
        """返回单批【全量训练矩阵】batch。

        batch = (users, items, conf, p, user_obs, item_obs)：
        - users/items/conf/p：全部训练观测（论文 Eq.3 的 Cᵘ、p(u) 输入）
        - user_obs/item_obs：预构建的观测分组（审阅②，避免每轮重建）
        """
        ds = self._dataset(self._train_pairs)
        return TorchDataLoader(
            ds, batch_size=len(ds), shuffle=False, num_workers=0,
            pin_memory=False,
            collate_fn=lambda b: (
                ds.users, ds.items, ds.conf, ds.p,
                ds.user_obs, ds.item_obs,
            ),
        )

    def val_loader(self) -> TorchDataLoader:
        return self._full_loader(self._dataset(self._val_pairs))

    def test_loader(self) -> TorchDataLoader:
        return self._full_loader(self._dataset(self.test_pairs))

    def get_init_params(self) -> Dict[str, Any]:
        return {KEY_NUM_USERS: self.num_users, KEY_NUM_ITEMS: self.num_items}

    def get_dataset(self, split: str) -> WMFDataset:
        if split == "train":
            return self._dataset(self._train_pairs)
        if split == "val":
            return self._dataset(self._val_pairs)
        return self._dataset(self.test_pairs)
