"""Matrix Factorization (MF) 模型结构

与 LightGCN 共享同一套 TrainableModel 接口（train_step / eval_step /
build_dataloader / get_user_embeddings / get_item_embeddings / embedding），
区别仅在于：
- 不做图卷积传播：最终嵌入 = 第 0 层嵌入（E = E^(0)）
- 预测：y_hat = e_u^T e_i（与 LightGCN 相同的 BPR 训练目标）

这样 attacks/pgd 的攻击流程可以对 MF / LightGCN 使用完全相同的代码路径，
只是通过 registry 注册的模型类不同。
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from training.framework import TrainableModel, TrainingConfig


class MatrixFactorization(TrainableModel):
    """经典 MF：用户/物品共享嵌入表，BPR 成对损失训练（隐式反馈）。

    论文依据：PGD 论文 Eq. (2) 的矩阵分解形式 M ≈ U V^T；
    训练目标采用本项目框架统一的 BPR（与 LightGCN 一致）。
    """

    def __init__(self, config: TrainingConfig, num_users: int, num_items: int,
                 edge_index: torch.Tensor = None):
        super().__init__(config)
        self.config = config

        self.num_users = num_users
        self.num_items = num_items
        self.n_nodes = num_users + num_items
        self.emb_dim = config.get("emb_dim", 64)

        # 唯一可训练参数：嵌入表（用户行 + 物品行，布局与 LightGCN 一致）
        self.embedding = nn.Embedding(self.n_nodes, self.emb_dim)
        self.embedding.to(self._device)
        init_method = config.get("init_method", "xavier_uniform")
        self._init_embedding(init_method)
        print(f"[MatrixFactorization] n_users={num_users}, n_items={num_items}, "
              f"emb_dim={self.emb_dim}, init={init_method}")

        self._step_count = 0

    def _init_embedding(self, method: str):
        init_map = {
            "normal":           lambda w: nn.init.normal_(w, mean=0.0, std=0.1),
            "xavier_uniform":   nn.init.xavier_uniform_,
            "xavier_normal":    nn.init.xavier_normal_,
            "kaiming_uniform":  nn.init.kaiming_uniform_,
            "kaiming_normal":   nn.init.kaiming_normal_,
        }
        if method in init_map:
            init_map[method](self.embedding.weight)
        elif method == "uniform":
            bound = (3.0 / self.emb_dim) ** 0.5
            nn.init.uniform_(self.embedding.weight, a=-bound, b=bound)
        else:
            raise ValueError(
                f"不支持的 init_method='{method}'，可选 "
                f"normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform"
            )

    def forward(self, users: torch.Tensor, items: torch.Tensor = None):
        """前向传播：直接查嵌入表计算预测分数。
        返回:
        - items 为 None: 返回全部节点嵌入（用户行 + 物品行）
        - items 给定: 返回指定 (user, item) 对的预测分数
        """
        if items is not None:
            user_emb = self.embedding.weight[users]
            item_emb = self.embedding.weight[self.num_users + items]
            return (user_emb * item_emb).sum(dim=1)
        return self.embedding.weight

    def get_user_embeddings(self):
        return self.embedding.weight[:self.num_users]

    def get_item_embeddings(self):
        return self.embedding.weight[self.num_users:]

    def _bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users, pos_items)
        batch_size, neg_ratio = users.size(0), neg_items.size(1)
        users_expanded = users.unsqueeze(1).expand(-1, neg_ratio).reshape(-1)
        neg_scores = self.forward(users_expanded, neg_items.reshape(-1))
        neg_scores = neg_scores.view(batch_size, neg_ratio)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores))

        user_ego = self.embedding.weight[users]
        pos_ego = self.embedding.weight[self.num_users + pos_items]
        neg_ego = self.embedding.weight[self.num_users + neg_items]
        reg_loss = (0.5) * (user_ego.norm(p=2).pow(2) +
                            pos_ego.norm(p=2).pow(2) +
                            neg_ego.norm(p=2).pow(2)) / users.size(0)
        reg_loss = reg_loss * self.config.get("weight_decay", 1e-4)
        return bpr_loss, reg_loss

    def train_step(self, batch: Any) -> Dict[str, float]:
        users, pos_items, neg_items = batch
        users = users.to(self._device)
        pos_items = pos_items.to(self._device)
        neg_items = neg_items.to(self._device)

        bpr_loss, reg_loss = self._bpr_loss(users, pos_items, neg_items)
        loss = bpr_loss + reg_loss

        if not hasattr(self, "_optimizer"):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        self._optimizer.zero_grad()
        loss.backward()

        if self._step_count == 0 and self.embedding.weight.grad is not None:
            g_norm = self.embedding.weight.grad.norm().item()
            print(f"  [diag-step0] first_batch_grad={g_norm:.4f} "
                  f"bpr={bpr_loss.item():.4f} reg={reg_loss.item():.4f}")

        self._optimizer.step()
        self._step_count += 1
        return {"loss": loss.item(), "bpr": bpr_loss.item(), "reg": reg_loss.item()}

    def eval_step(self, batch: Any) -> Dict[str, float]:
        users, pos_items, neg_items = batch
        users = users.to(self._device)
        pos_items = pos_items.to(self._device)
        neg_items = neg_items.to(self._device)

        with torch.no_grad():
            bpr_loss, reg_loss = self._bpr_loss(users, pos_items, neg_items)
            loss = bpr_loss + reg_loss
        return {"val_loss": loss.item()}

    def build_dataloader(self, config: TrainingConfig):
        from models.mf.dataset import MFDataLoader
        return MFDataLoader(config)

    def predict_full_ranking(self, user_ids: torch.Tensor,
                             item_ids: torch.Tensor, batch_size: int = 1024):
        """全量排序评分（独立方法，不通过 eval_step）。"""
        self.eval()
        user_emb = self.get_user_embeddings()
        item_emb = self.get_item_embeddings()
        scores_list = []
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size].to(self._device)
            batch_scores = user_emb[batch_users] @ item_emb.T
            scores_list.append(batch_scores.cpu())
        return torch.cat(scores_list, dim=0)
