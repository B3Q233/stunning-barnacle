"""LightGCN 模型结构
遵循理解文档模块二·2.3：
- 唯一可训练参数: E^(0) ∈ R^(M+N)×64（Xavier 初始化）
- LGC: E^(k+1) = A_hat @ E^(k)（无 W, 无 σ, 无自连接）
- 层组合: E = Σ α_k E^(k), α_k = 1/(K+1)
- 预测: y_hat = e_u^T e_i
"""
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from typing import Any, Dict

from training.framework import TrainableModel, TrainingConfig


class LightGCN(TrainableModel):
    """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"""

    def __init__(self, config: TrainingConfig, num_users: int, num_items: int,
                 edge_index: torch.Tensor = None):
        super().__init__(config)
        self.config = config  # 保存配置引用

        self.num_users = num_users
        self.num_items = num_items
        self.n_nodes = num_users + num_items
        self.emb_dim = config.get("emb_dim", 64)
        self.n_layers = config.get("n_layers", 3)

        # 唯一可训练参数：第 0 层嵌入
        self.embedding = nn.Embedding(self.n_nodes, self.emb_dim)
        self.embedding.to(self._device)  # 手动移到设备（super().__init__ 的 .to() 时尚未创建）
        nn.init.xavier_uniform_(self.embedding.weight)
        print(f"[LightGCN] n_users={num_users}, n_items={num_items}, "
              f"emb_dim={self.emb_dim}, n_layers={self.n_layers}")

        # 构造归一化邻接矩阵 A_hat = D^(-1/2) A D^(-1/2)
        if edge_index is not None:
            self._build_adj(edge_index)
        else:
            self.A_hat = None

    def _build_adj(self, edge_index: torch.Tensor):
        """从边列表构造对称归一化邻接矩阵（稀疏 CSR 格式）
        edge_index: (2, num_edges), 行 0=user id, 行 1=item id
        用户 id 保持原样 [0, M), 物品 id 偏移 M 到 [M, M+N)
        """
        users = edge_index[0].numpy()
        items = edge_index[1].numpy() + self.num_users  # 偏移到物品节点空间
        values = np.ones(len(users))

        # 构建稀疏邻接矩阵（对称）
        adj = sp.coo_matrix(
            (values, (users, items)),
            shape=(self.n_nodes, self.n_nodes)
        )
        adj = adj + adj.T  # 对称化

        # 计算度矩阵并归一化: D^(-1/2) A D^(-1/2)
        rowsum = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat

        # 转为稀疏 COO → PyTorch sparse tensor（必须在构造时就指定正确设备）
        coo = norm_adj.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col])).to(self._device)
        values = torch.FloatTensor(coo.data).to(self._device)
        self.A_hat = torch.sparse_coo_tensor(indices, values,
                                              torch.Size([self.n_nodes, self.n_nodes]),
                                              device=self._device)
        print(f"[LightGCN] 邻接矩阵: {self.n_nodes}x{self.n_nodes}, "
              f"edges={len(coo.data)}, device={self.A_hat.device}")

    def forward(self, users: torch.Tensor, items: torch.Tensor = None):
        """前向传播：使用缓存的最终嵌入计算预测分数。

        返回:
        - 若 items 为 None: 返回所有用户/物品的最终嵌入
        - 若 items 给定: 返回指定 (user, item) 对的预测分数
        """
        final_emb = self._compute_final_emb()

        if items is not None:
            user_emb = final_emb[users]
            item_emb = final_emb[self.num_users + items]
            return (user_emb * item_emb).sum(dim=1)
        else:
            return final_emb

    def get_user_embeddings(self):
        """获取用户最终嵌入"""
        return self._compute_final_emb()[:self.num_users]

    def get_item_embeddings(self):
        """获取物品最终嵌入"""
        return self._compute_final_emb()[self.num_users:]

    def _compute_final_emb(self):
        """计算最终嵌入（内部使用）"""
        all_emb = [self.embedding.weight]
        for _ in range(self.n_layers):
            all_emb.append(torch.sparse.mm(self.A_hat, all_emb[-1]))
        return torch.stack(all_emb, dim=0).mean(dim=0)

    def train_step(self, batch: Any) -> Dict[str, float]:
        """BPR 训练步骤"""
        users, pos_items, neg_items = batch
        users = users.to(self._device)
        pos_items = pos_items.to(self._device)
        neg_items = neg_items.to(self._device)

        # 计算预测分数
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)

        # BPR 损失 + L2 正则
        bpr_loss = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.embedding.weight.norm(p=2).pow(2) * self.config.get("weight_decay", 1e-4)

        loss = bpr_loss + reg_loss

        # 优化
        if not hasattr(self, '_optimizer'):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return {"loss": loss.item(), "bpr": bpr_loss.item(), "reg": reg_loss.item()}

    def eval_step(self, batch: Any) -> Dict[str, float]:
        """评估步骤：计算 BPR 验证损失（标量）"""
        users, pos_items, neg_items = batch
        users = users.to(self._device)
        pos_items = pos_items.to(self._device)
        neg_items = neg_items.to(self._device)

        with torch.no_grad():
            pos_scores = self.forward(users, pos_items)
            neg_scores = self.forward(users, neg_items)
            bpr_loss = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))
            reg_loss = self.embedding.weight.norm(p=2).pow(2) * self.config.get("weight_decay", 1e-4)
            loss = bpr_loss + reg_loss

        return {"val_loss": loss.item()}

    def build_dataloader(self, config: TrainingConfig):
        """构造与本模型匹配的数据载入器"""
        from models.lightgcn.dataset import LightGCNDataLoader
        return LightGCNDataLoader(config)

    def predict_full_ranking(self, user_ids: torch.Tensor,
                             item_ids: torch.Tensor, batch_size: int = 1024):
        """全量排序评分（独立方法，不通过 eval_step）"""
        self.eval()
        user_emb = self.get_user_embeddings()
        item_emb = self.get_item_embeddings()

        scores_list = []
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size].to(self._device)
            batch_scores = user_emb[batch_users] @ item_emb.T
            scores_list.append(batch_scores.cpu())
        return torch.cat(scores_list, dim=0)
