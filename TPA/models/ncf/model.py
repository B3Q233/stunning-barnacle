"""Neural collaborative filtering victim model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.revisit_common import ensure_training_config
from training.framework import TrainableModel

class NCFModel(TrainableModel):
    def __init__(self, config, num_users, num_items, edge_index=None):
        del edge_index
        config = ensure_training_config(config)
        super().__init__(config)
        self.config = config
        self.num_users, self.num_items = int(num_users), int(num_items)
        emb_dim = int(config.get("emb_dim", 64))
        hidden_dim = int(config.get("hidden_dim", 128))
        self.user_embedding = nn.Embedding(self.num_users, emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, emb_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def _validate_ids(self, user_ids, item_ids):
        if user_ids.numel() and (user_ids.min() < 0 or user_ids.max() >= self.num_users or item_ids.min() < 0 or item_ids.max() >= self.num_items):
            raise IndexError("user or item id out of range")

    def forward(self, user_ids, item_ids):
        self._validate_ids(user_ids, item_ids)
        features = torch.cat([self.user_embedding(user_ids), self.item_embedding(item_ids)], dim=-1)
        return self.mlp(features).squeeze(-1)

    def bpr_loss(self, user_ids, positive_ids, negative_ids):
        return -F.logsigmoid(self.forward(user_ids, positive_ids) - self.forward(user_ids, negative_ids)).mean()

    def predict_full_ranking(self, user_ids, item_ids=None, batch_size=1024):
        item_ids = torch.arange(self.num_items, device=self._device) if item_ids is None else item_ids.to(self._device)
        rows = []
        for user_batch in user_ids.to(self._device).split(batch_size):
            users = user_batch[:, None].expand(-1, item_ids.numel()).reshape(-1)
            items = item_ids[None, :].expand(user_batch.numel(), -1).reshape(-1)
            rows.append(self.forward(users, items).view(user_batch.numel(), -1).detach().cpu())
        return torch.cat(rows, dim=0) if rows else torch.empty((0, item_ids.numel()))

    def get_user_embeddings(self):
        return self.user_embedding.weight

    def get_item_embeddings(self):
        return self.item_embedding.weight

    def train_step(self, batch):
        users, positives = batch[:2]
        users, positives = users.to(self._device), positives.to(self._device)
        negatives = torch.randint(self.num_items, positives.shape, device=self._device)
        if not hasattr(self, "_optimizer"):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config.get("lr", 1e-3)))
        self._optimizer.zero_grad(set_to_none=True)
        loss = self.bpr_loss(users, positives, negatives)
        loss.backward()
        self._optimizer.step()
        return {"loss": float(loss.item())}

    def eval_step(self, batch):
        users, positives = batch[:2]
        users, positives = users.to(self._device), positives.to(self._device)
        negatives = torch.randint(self.num_items, positives.shape, device=self._device)
        with torch.no_grad():
            loss = self.bpr_loss(users, positives, negatives)
        return {"val_loss": float(loss.item())}

    def build_dataloader(self, config):
        from models.ncf.dataset import NCFModelDataLoader
        return NCFModelDataLoader(config)
