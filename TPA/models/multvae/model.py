"""MultVAE victim model."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.revisit_common import ensure_training_config
from training.framework import TrainableModel

class MultVAEModel(TrainableModel):
    def __init__(self, config, num_users, num_items, edge_index=None):
        del edge_index
        config = ensure_training_config(config)
        super().__init__(config)
        self.config = config
        self.num_users, self.num_items = int(num_users), int(num_items)
        hidden_dim = int(config.get("hidden_dim", 128))
        latent_dim = int(config.get("latent_dim", config.get("emb_dim", 64)))
        self.encoder = nn.Sequential(nn.Linear(self.num_items, hidden_dim), nn.Tanh())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, self.num_items)
        self.register_buffer("user_history", torch.zeros(self.num_users, self.num_items))

    @staticmethod
    def normalize_input(interactions):
        interactions = interactions.float()
        return interactions / interactions.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

    def encode(self, interactions):
        hidden = self.encoder(self.normalize_input(interactions))
        return self.mu(hidden), self.logvar(hidden)

    def forward(self, interactions):
        mu, logvar = self.encode(interactions)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.decoder(latent), mu, logvar

    def loss(self, interactions, logits, mu, logvar, anneal=None):
        reconstruction = -(F.log_softmax(logits, dim=1) * interactions.float()).sum(dim=1).mean()
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
        if anneal is None:
            anneal = float(self.config.get("kl_anneal", 1.0))
        return reconstruction + float(anneal) * kl

    def set_history(self, history):
        if tuple(history.shape) != (self.num_users, self.num_items):
            raise ValueError("history shape does not match model dimensions")
        self.user_history.copy_(history.float().to(self.user_history))
        return self

    def fit(self, matrix, epochs=10, lr=1e-3):
        self.set_history(matrix)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        interactions = matrix.to(self._device).float()
        for epoch in range(int(epochs)):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            logits, mu, logvar = self.forward(interactions)
            anneal = min(1.0, (epoch + 1) / max(1, int(self.config.get("kl_warmup", 10))))
            loss = self.loss(interactions, logits, mu, logvar, anneal)
            loss.backward()
            optimizer.step()
        return {"loss": float(loss.item())}

    def predict_full_ranking(self, user_ids, item_ids=None, batch_size=1024):
        rows = []
        self.eval()
        with torch.no_grad():
            for batch in user_ids.to(self._device).split(batch_size):
                scores, _, _ = self.forward(self.user_history[batch])
                rows.append(scores.cpu())
        scores = torch.cat(rows, dim=0) if rows else torch.empty((0, self.num_items))
        return scores if item_ids is None else scores[:, item_ids.cpu()]

    def get_user_embeddings(self):
        return self.user_history

    def get_item_embeddings(self):
        return self.decoder.weight.detach().T

    def train_step(self, batch):
        interactions = batch[0].to(self._device).float()
        if not hasattr(self, "_optimizer"):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config.get("lr", 1e-3)))
        self._optimizer.zero_grad(set_to_none=True)
        logits, mu, logvar = self.forward(interactions)
        loss = self.loss(interactions, logits, mu, logvar)
        loss.backward()
        self._optimizer.step()
        return {"loss": float(loss.item())}

    def eval_step(self, batch):
        interactions = batch[0].to(self._device).float()
        with torch.no_grad():
            logits, mu, logvar = self.forward(interactions)
            loss = self.loss(interactions, logits, mu, logvar)
        return {"val_loss": float(loss.item())}

    def build_dataloader(self, config):
        from models.multvae.dataset import MultVAEModelDataLoader
        return MultVAEModelDataLoader(config)
