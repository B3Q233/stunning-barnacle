"""Item auto-encoder victim model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.revisit_common import ensure_training_config
from training.framework import TrainableModel

class ItemAEModel(TrainableModel):
    def __init__(self, config, num_users, num_items, edge_index=None):
        del edge_index
        config = ensure_training_config(config)
        super().__init__(config)
        self.config = config
        self.num_users, self.num_items = int(num_users), int(num_items)
        hidden_dim = int(config.get("hidden_dim", config.get("emb_dim", 128)))
        self.encoder = nn.Linear(self.num_items, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, self.num_items)
        self.register_buffer("user_history", torch.zeros(self.num_users, self.num_items))

    def forward(self, interactions):
        return self.decoder(torch.tanh(self.encoder(interactions.float())))

    def reconstruction_loss(self, interactions):
        return F.binary_cross_entropy_with_logits(self.forward(interactions), interactions.float())

    def set_history(self, history):
        if tuple(history.shape) != (self.num_users, self.num_items):
            raise ValueError("history shape does not match model dimensions")
        self.user_history.copy_(history.float().to(self.user_history))
        return self

    def fit(self, matrix, epochs=10, lr=1e-3):
        self.set_history(matrix)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        interactions = matrix.to(self._device).float()
        for _ in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            loss = self.reconstruction_loss(interactions)
            loss.backward()
            optimizer.step()
        return {"loss": float(loss.item())}

    def predict_full_ranking(self, user_ids, item_ids=None, batch_size=1024):
        del batch_size
        scores = self.forward(self.user_history[user_ids.to(self._device)]).detach().cpu()
        return scores if item_ids is None else scores[:, item_ids.detach().cpu()]

    def get_user_embeddings(self):
        return self.user_history

    def get_item_embeddings(self):
        return self.decoder.weight.detach().T

    def train_step(self, batch):
        interactions = batch[0].to(self._device).float()
        if not hasattr(self, "_optimizer"):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config.get("lr", 1e-3)))
        self._optimizer.zero_grad(set_to_none=True)
        loss = self.reconstruction_loss(interactions)
        loss.backward()
        self._optimizer.step()
        return {"loss": float(loss.item())}

    def eval_step(self, batch):
        with torch.no_grad():
            loss = self.reconstruction_loss(batch[0].to(self._device).float())
        return {"val_loss": float(loss.item())}

    def build_dataloader(self, config):
        from models.itemae.dataset import ItemAEModelDataLoader
        return ItemAEModelDataLoader(config)
