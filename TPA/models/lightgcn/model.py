"""LightGCN 模型结构
遵循理解文档模块二·2.3：
- 唯一可训练参数: E^(0) ∈ R^(M+N)×64（Xavier 初始化）
- LGC: E^(k+1) = A_hat @ E^(k)（无 W, 无 σ, 无自连接）
- 层组合: E = Σ α_k E^(k), α_k = 1/(K+1)
- 预测: y_hat = e_u^T e_i
"""
import os
import sys

# 确保项目根目录 G:\Idea\TPA 在 sys.path 中（支持直接 import/运行本模块）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
        init_method = config.get("init_method", "xavier_uniform")
        self._init_embedding(init_method)
        emb_norm_after = self.embedding.weight.norm(p=2).item()
        print(f"[LightGCN] n_users={num_users}, n_items={num_items}, "
              f"emb_dim={self.emb_dim}, n_layers={self.n_layers}, "
              f"init={init_method}, emb_norm={emb_norm_after:.1f}")

        # 步数计数器（用于诊断第一个 batch 的梯度）
        self._step_count = 0

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
        with np.errstate(divide='ignore', invalid='ignore'):
            d_inv_sqrt = np.where(rowsum > 0, np.power(rowsum, -0.5), 0.0)
        isolated = (rowsum == 0).sum()
        if isolated > 0:
            print(f"[LightGCN] [!] 发现 {isolated} 个孤立节点（度数为 0），其嵌入仅靠 L2 正则更新")
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

    def _init_embedding(self, method: str):
        """根据配置初始化嵌入权重。

        支持的方法:
        - normal  (N(0, 0.1)) — 原始 LightGCN 论文推荐，N(0,1) 在无激活函数时更优
        - xavier_uniform / xavier_normal (Glorot)
        - kaiming_uniform / kaiming_normal (He)
        - uniform  (U(-√(3/d), √(3/d))) — 按 emb_dim 自适应
        """
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
                f"不支持的 init_method='{method}'，可选: "
                f"normal, xavier_uniform, xavier_normal, kaiming_uniform, kaiming_normal, uniform"
            )

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
        neg_items = neg_items.to(self._device)              # [batch_size, neg_ratio]

        # 计算预测分数
        pos_scores = self.forward(users, pos_items)          # [batch_size]

        # neg_items 为 2D [batch_size, neg_ratio]，展开后传入 forward
        batch_size, neg_ratio = users.size(0), neg_items.size(1)
        users_expanded = users.unsqueeze(1).expand(-1, neg_ratio).reshape(-1)
        neg_scores = self.forward(users_expanded, neg_items.reshape(-1))
        neg_scores = neg_scores.view(batch_size, neg_ratio)  # [batch_size, neg_ratio]

        # BPR 损失
        bpr_loss = -torch.mean(nn.functional.logsigmoid(
            pos_scores.unsqueeze(1) - neg_scores))            # 广播: [B,1] - [B,R]

        # L2 正则：仅约束第 0 层嵌入，按 batch 归一化（对齐原始论文）
        user_ego = self.embedding.weight[users]                             # [B, D]
        pos_ego = self.embedding.weight[self.num_users + pos_items]         # [B, D]
        neg_ego = self.embedding.weight[self.num_users + neg_items]         # [B, R, D]
        reg_loss = (0.5) * (user_ego.norm(p=2).pow(2) +
                            pos_ego.norm(p=2).pow(2) +
                            neg_ego.norm(p=2).pow(2)) / users.size(0)
        reg_loss = reg_loss * self.config.get("weight_decay", 1e-4)

        loss = bpr_loss + reg_loss

        # 优化
        if not hasattr(self, '_optimizer'):
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        self._optimizer.zero_grad()
        loss.backward()

        # 诊断：第一个 batch 后打印梯度信息
        if self._step_count == 0 and self.embedding.weight.grad is not None:
            g_norm = self.embedding.weight.grad.norm().item()
            pos_mean = pos_scores.detach().mean().item()
            neg_mean = neg_scores.detach().mean().item()
            print(f"  [diag-step0] first_batch_grad={g_norm:.4f} "
                  f"pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} "
                  f"bpr={bpr_loss.item():.4f} reg={reg_loss.item():.4f}")

        self._optimizer.step()
        self._step_count += 1

        return {"loss": loss.item(), "bpr": bpr_loss.item(), "reg": reg_loss.item()}

    def eval_step(self, batch: Any) -> Dict[str, float]:
        """评估步骤：计算 BPR 验证损失（标量）"""
        users, pos_items, neg_items = batch
        users = users.to(self._device)
        pos_items = pos_items.to(self._device)
        neg_items = neg_items.to(self._device)              # [batch_size, neg_ratio]

        with torch.no_grad():
            pos_scores = self.forward(users, pos_items)      # [batch_size]

            # neg_items 为 2D [batch_size, neg_ratio]，展开后传入 forward
            batch_size, neg_ratio = users.size(0), neg_items.size(1)
            users_expanded = users.unsqueeze(1).expand(-1, neg_ratio).reshape(-1)
            neg_scores = self.forward(users_expanded, neg_items.reshape(-1))
            neg_scores = neg_scores.view(batch_size, neg_ratio)  # [batch_size, neg_ratio]

            bpr_loss = -torch.mean(nn.functional.logsigmoid(
                pos_scores.unsqueeze(1) - neg_scores))       # 广播: [B,1] - [B,R]

            # L2 正则：仅约束第 0 层嵌入，按 batch 归一化（对齐原始论文）
            user_ego = self.embedding.weight[users]                         # [B, D]
            pos_ego = self.embedding.weight[self.num_users + pos_items]     # [B, D]
            neg_ego = self.embedding.weight[self.num_users + neg_items]     # [B, R, D]
            reg_loss = (0.5) * (user_ego.norm(p=2).pow(2) +
                                pos_ego.norm(p=2).pow(2) +
                                neg_ego.norm(p=2).pow(2)) / users.size(0)
            reg_loss = reg_loss * self.config.get("weight_decay", 1e-4)
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
