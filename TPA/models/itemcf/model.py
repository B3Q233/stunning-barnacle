"""ItemCF victim model。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from training.framework import TrainableModel
from models.revisit_common import ensure_training_config

from models.revisit_common import build_user_items


class ItemCFModel(TrainableModel):
    def __init__(self, config, num_users: int, num_items: int,
                 edge_index: Optional[torch.Tensor] = None):
        config = ensure_training_config(config)
        super().__init__(config)
        self.config = config
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.register_buffer("item_similarity", torch.eye(self.num_items))
        self.register_buffer("user_history", torch.zeros(self.num_users, self.num_items))
        del edge_index

    def fit(self, pairs):
        history = torch.zeros(self.num_users, self.num_items, dtype=torch.float32)
        for user, item in pairs:
            if not 0 <= user < self.num_users or not 0 <= item < self.num_items:
                raise ValueError("交互 ID 越界")
            history[user, item] = 1.0
        cooc = history.T @ history
        norms = torch.sqrt(torch.diag(cooc).clamp_min(1e-12))
        sim = cooc / (norms[:, None] * norms[None, :])
        sim.fill_diagonal_(0.0)
        self.user_history.copy_(history)
        self.item_similarity.copy_(sim)
        return {"loss": 0.0}

    def score_users(self, user_ids: torch.Tensor, interactions: torch.Tensor) -> torch.Tensor:
        if user_ids.numel() and (int(user_ids.min()) < 0 or int(user_ids.max()) >= self.num_users):
            raise IndexError("用户 ID 越界")
        if interactions.shape != (len(user_ids), self.num_items):
            raise ValueError("interactions shape 与模型不一致")
        return interactions.to(self.item_similarity) @ self.item_similarity

    def predict_full_ranking(self, user_ids: torch.Tensor, item_ids=None, batch_size: int = 1024):
        if user_ids.numel() and (int(user_ids.min()) < 0 or int(user_ids.max()) >= self.num_users):
            raise IndexError("用户 ID 越界")
        scores = self.user_history[user_ids.cpu()] @ self.item_similarity.cpu()
        if item_ids is not None:
            scores = scores[:, item_ids.cpu()]
        return scores

    def get_user_embeddings(self):
        return self.user_history

    def get_item_embeddings(self):
        return self.item_similarity

    def train_step(self, batch):
        pairs = list(zip(batch[0].tolist(), batch[1].tolist()))
        return self.fit(pairs)

    def eval_step(self, batch):
        return {"val_loss": 0.0}
    def build_dataloader(self, config):
        from models.itemcf.dataset import ItemCFDataLoader
        return ItemCFDataLoader(config)
