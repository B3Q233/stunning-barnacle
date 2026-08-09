"""MF 数据导入模块

与 LightGCN 的 dataset.py 完全同构（BPR 负采样，batch 格式
(users, pos_items, neg_items)），仅数据路径指向 models/mf/data/processed。
"""
import pickle
import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset


KEY_NUM_USERS = "num_users"
KEY_NUM_ITEMS = "num_items"
KEY_DATASET = "dataset"


class MFDataset(Dataset):
    """MF BPR 训练/验证数据集（与 LightGCNDataset 同构）。"""

    def __init__(self, pairs: List[Tuple[int, int]], num_items: int,
                 user_items: Dict[int, set], num_users: int, mode: str = "train",
                 neg_ratio: int = 1):
        self.pairs = pairs
        self.num_items = num_items
        self.num_users = num_users
        self.user_items = user_items
        self.mode = mode
        self.neg_ratio = neg_ratio
        self.users = list(user_items.keys())

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        user, pos_item = self.pairs[idx]
        if self.mode == "train":
            neg_items = []
            user_interacted = self.user_items.get(user, set())
            while len(neg_items) < self.neg_ratio:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in user_interacted and neg_item not in neg_items:
                    neg_items.append(neg_item)
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_items, dtype=torch.long),
            )
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
        )


class MFDataLoader:
    """MF 数据加载器，实现 DatasetProtocol 五个方法。"""

    def __init__(self, config):
        self.config = config
        dataset_name = config.get(KEY_DATASET, "ml100k")
        meta_path = f"g:/Idea/TPA/models/mf/data/processed/{dataset_name}/meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.num_users = meta["num_users"]
        self.num_items = meta["num_items"]
        self.train_pairs = meta["train_pairs"]
        self.test_pairs = meta["test_pairs"]
        self.user_items = meta["user_items"]
        self.neg_ratio = config.get("neg_ratio", 1)

        random.seed(42)
        shuffled = self.train_pairs.copy()
        random.shuffle(shuffled)
        split = int(len(shuffled) * 0.95)
        self._train_pairs = shuffled[:split]
        self._val_pairs = shuffled[split:]
        self.all_train_pairs = self.train_pairs

        print(f"[MFDataLoader] {dataset_name}: "
              f"users={self.num_users}, items={self.num_items}, "
              f"train={len(self._train_pairs)}, val={len(self._val_pairs)}, "
              f"test={len(self.test_pairs)}")

    def train_loader(self) -> TorchDataLoader:
        dataset = MFDataset(
            self._train_pairs, self.num_items, self.user_items,
            self.num_users, mode="train", neg_ratio=self.neg_ratio
        )
        return TorchDataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=0, pin_memory=True,
        )

    def val_loader(self) -> TorchDataLoader:
        dataset = MFDataset(
            self._val_pairs, self.num_items, self.user_items,
            self.num_users, mode="train", neg_ratio=self.neg_ratio
        )
        return TorchDataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def test_loader(self) -> TorchDataLoader:
        dataset = MFDataset(
            self.test_pairs, self.num_items, self.user_items,
            self.num_users, mode="test"
        )
        return TorchDataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def get_init_params(self) -> Dict[str, Any]:
        return {KEY_NUM_USERS: self.num_users, KEY_NUM_ITEMS: self.num_items}

    def get_dataset(self, split: str):
        if split == "train":
            pairs = self._train_pairs
        elif split == "val":
            pairs = self._val_pairs
        else:
            pairs = self.test_pairs
        return MFDataset(
            pairs, self.num_items, self.user_items,
            self.num_users, mode="train" if split != "test" else "test",
            neg_ratio=self.neg_ratio
        )
