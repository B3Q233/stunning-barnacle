"""Collaborative metric learning victim model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.revisit_common import ensure_training_config
from training.framework import TrainableModel

class CMLModel(TrainableModel):
    def __init__(self, config, num_users, num_items, edge_index=None):
        del edge_index
        config = ensure_training_config(config)
        super().__init__(config)
        self.config = config
        self.num_users, self.num_items = int(num_users), int(num_items)
        emb_dim = int(config.get("emb_dim", 64))
        self.user_embedding = nn.Embedding(self.num_users, emb_dim, max_norm=1.0)
        self.item_embedding = nn.Embedding(self.num_items, emb_dim, max_norm=1.0)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def distance(self, user_ids, item_ids):
        return (self.user_embedding(user_ids) - self.item_embedding(item_ids)).pow(2).sum(dim=1)

    def forward(self, user_ids, item_ids):
        return -self.distance(user_ids, item_ids)

    def margin_loss(self, user_ids, positive_ids, negative_ids):
        margin = float(self.config.get("margin", 1.0))
        positive = self.distance(user_ids, positive_ids)
        negative = self.distance(user_ids, negative_ids)
        loss = F.relu(margin + positive - negative).mean()
        reg = float(self.config.get("weight_decay", 0.0)) * (self.user_embedding.weight.pow(2).mean() + self.item_embedding.weight.pow(2).mean())
        return loss + reg

    def predict_full_ranking(self, user_ids, item_ids=None, batch_size=1024):
        del batch_size
        user_ids = user_ids.to(self._device)
        item_ids = torch.arange(self.num_items, device=self._device) if item_ids is None else item_ids.to(self._device)
        scores = -((self.user_embedding(user_ids)[:, None, :] - self.item_embedding(item_ids)[None, :, :]) ** 2).sum(dim=-1)
        return scores.detach().cpu()

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
        loss = self.margin_loss(users, positives, negatives)
        loss.backward()
        self._optimizer.step()
        return {"loss": float(loss.item())}

    def eval_step(self, batch):
        users, positives = batch[:2]
        users, positives = users.to(self._device), positives.to(self._device)
        negatives = torch.randint(self.num_items, positives.shape, device=self._device)
        with torch.no_grad():
            loss = self.margin_loss(users, positives, negatives)
        return {"val_loss": float(loss.item())}

    def build_dataloader(self, config):
        from models.cml.dataset import CMLModelDataLoader
        return CMLModelDataLoader(config)
