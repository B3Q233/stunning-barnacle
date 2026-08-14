"""LightGCN 数据导入模块
实现 LightGCNDataset 和 LightGCNDataLoader，遵循 DatasetProtocol 协议。
训练: BPR 随机负采样，batch 格式 (users, pos_items, neg_items)
评估: 返回 BPR loss 标量；全量排序指标通过 predict_full_ranking() 单独计算
"""
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset


# ── 配置键名常量（统一来源，禁止各文件硬编码） ──
KEY_NUM_USERS = "num_users"
KEY_NUM_ITEMS = "num_items"
KEY_DATASET = "dataset"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA 项目根


class LightGCNDataset(Dataset):
    """LightGCN BPR 训练/验证数据集"""

    def __init__(self, pairs: List[Tuple[int, int]], num_items: int,
                 user_items: Dict[int, set], num_users: int, mode: str = "train",
                 neg_ratio: int = 1):
        self.pairs = pairs
        self.num_items = num_items
        self.num_users = num_users
        self.user_items = user_items
        self.mode = mode
        self.neg_ratio = neg_ratio

        # 构建用户列表，供训练时随机采样用户用于负采样
        self.users = list(user_items.keys())

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        user, pos_item = self.pairs[idx]

        if self.mode == "train":
            # 随机负采样：为每个正样本采样 neg_ratio 个用户未交互过的物品
            neg_items = []
            user_interacted = self.user_items.get(user, set())
            while len(neg_items) < self.neg_ratio:
                neg_item = random.randint(0, self.num_items - 1)
                if neg_item not in user_interacted and neg_item not in neg_items:
                    neg_items.append(neg_item)

            # neg_items: 1D 张量 [neg_ratio]；collate 后成为 [batch_size, neg_ratio]
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
                torch.tensor(neg_items, dtype=torch.long),
            )
        else:
            # 验证/测试模式：返回用户和正样本物品
            return (
                torch.tensor(user, dtype=torch.long),
                torch.tensor(pos_item, dtype=torch.long),
            )


class LightGCNDataLoader:
    """LightGCN 数据载入器，实现 DatasetProtocol 五个方法"""

    def __init__(self, config):
        self.config = config
        dataset_name = config.get(KEY_DATASET, "gowalla")

        # 加载预处理后的数据
        meta_path = (
            PROJECT_ROOT / "models" / "lightgcn" / "data" / "processed"
            / dataset_name / "meta.pkl"
        )
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.num_users = meta["num_users"]
        self.num_items = meta["num_items"]
        self.train_pairs = meta["train_pairs"]
        self.test_pairs = meta["test_pairs"]
        self.user_items = meta["user_items"]
        self.neg_ratio = config.get("neg_ratio", 1)

        # 验证集：从训练集中划出最后 5%
        random.seed(42)
        shuffled = self.train_pairs.copy()
        random.shuffle(shuffled)
        split = int(len(shuffled) * 0.95)
        self._train_pairs = shuffled[:split]
        self._val_pairs = shuffled[split:]

        # 完整训练集（用于构建邻接矩阵等不需要验证划分的场景）
        self.all_train_pairs = self.train_pairs

        print(f"[LightGCNDataLoader] {dataset_name}: "
              f"users={self.num_users}, items={self.num_items}, "
              f"train={len(self._train_pairs)}, val={len(self._val_pairs)}, "
              f"test={len(self.test_pairs)}")

    def train_loader(self) -> TorchDataLoader:
        dataset = LightGCNDataset(
            self._train_pairs, self.num_items, self.user_items,
            self.num_users, mode="train", neg_ratio=self.neg_ratio
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_loader(self) -> TorchDataLoader:
        """验证集：BPR 损失计算用"""
        dataset = LightGCNDataset(
            self._val_pairs, self.num_items, self.user_items,
            self.num_users, mode="train", neg_ratio=self.neg_ratio  # 使用负采样
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_loader(self) -> TorchDataLoader:
        """测试集"""
        dataset = LightGCNDataset(
            self.test_pairs, self.num_items, self.user_items,
            self.num_users, mode="test"
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def get_init_params(self) -> Dict[str, Any]:
        return {
            KEY_NUM_USERS: self.num_users,
            KEY_NUM_ITEMS: self.num_items,
        }

    def get_dataset(self, split: str):
        if split == "train":
            pairs = self._train_pairs
        elif split == "val":
            pairs = self._val_pairs
        else:
            pairs = self.test_pairs
        return LightGCNDataset(
            pairs, self.num_items, self.user_items,
            self.num_users, mode="train" if split != "test" else "test",
            neg_ratio=self.neg_ratio
        )
