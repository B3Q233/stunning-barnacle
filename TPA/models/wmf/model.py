"""WMF 模型结构（步骤③）

依据理解文档「模块二」：
- 双因子模型：user_factors X ∈ R^(m×f)、item_factors Y ∈ R^(n×f)；
  预测 p̂_ui = x_uᵀ y_i
- 无偏置/无 Dropout/无神经网络层（论文明确：纯线性双因子 + 置信加权）
- 初始化：论文未提及，复现默认高斯 N(0, 0.01)，可配置

训练优化为 ALS 闭式交替最小二乘（Eq.4/5），实现于 train_step / eval_step：
- 每轮先固定 Y 解 X，再固定 X 解 Y（论文约 10 轮收敛）
- 加速：A_u = YᵀY + Yᵀ(Cᵘ−I)Y + λI，只对观测对累加（Eq.6 区域）
- 一个 epoch = 一轮完整 sweep：train_loader 返回全量训练矩阵 + 预构建的
  user_obs / item_obs（不产生 mini-batch，审阅①）
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.wmf.config_keys import (
    KEY_FACTORS,
    KEY_INIT_METHOD,
    KEY_INIT_STD,
    KEY_LAMBDA_REG,
)
from training.framework import TrainableModel, TrainingConfig


def _als_sweep(X, Y, users, items, conf, p, user_obs, item_obs, lam):
    """一轮交替最小二乘（论文 Eq.4/5 + 加速技巧）。

    Args:
        X: (m, f) 用户因子
        Y: (n, f) 物品因子
        users/items/conf/p: 观测对数组（p_ui 与 c_ui）
        user_obs/item_obs: 预构建的观测分组（见 dataset.group_observations），
            全量 sweep 直接引用，不重复扫描
        lam: L2 正则系数
    Returns:
        (X_new, Y_new)（float64）
    """
    f = X.shape[1]
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    users_t = torch.as_tensor(users, dtype=torch.long)
    items_t = torch.as_tensor(items, dtype=torch.long)
    conf_t = torch.as_tensor(conf, dtype=torch.float64)
    p_t = torch.as_tensor(p, dtype=torch.float64)
    lam = float(lam)
    eye = torch.eye(f, dtype=torch.float64, device=Y.device)

    # 固定 Y，逐用户解 x_u = (YᵀCᵘY + λI)⁻¹ YᵀCᵘp(u)
    # 加速：YᵀCᵘY = YᵀY + Yᵀ(Cᵘ−I)Y，仅 n_u 个非零项
    YtY = Y.T @ Y
    X_new = X.clone()
    for u, pos in user_obs.items():
        idx = items_t[pos]
        c = conf_t[pos]
        pv = p_t[pos]
        Yu = Y[idx]
        A = YtY + (Yu.T * (c - 1.0)) @ Yu + lam * eye
        b = Yu.T @ (c * pv)
        X_new[u] = torch.linalg.solve(A, b)
    X = X_new

    # 固定 X，逐物品对称求解 y_i = (XᵀCⁱX + λI)⁻¹ XᵀCⁱp(i)
    XtX = X.T @ X
    item_ids = sorted(item_obs.keys())
    Y_new = Y.clone()
    chunk_size = 128  # 控制内存：块内观测外积张量 ~60MB（float64）
    for start in range(0, len(item_ids), chunk_size):
        ids = item_ids[start:start + chunk_size]
        # 汇总本块物品的全部观测，构建 local 下标（块内位置）
        obs_idx = []
        local_idx = []
        for k, i in enumerate(ids):
            for j in item_obs[i]:
                obs_idx.append(j)
                local_idx.append(k)
        obs_idx = torch.as_tensor(obs_idx, dtype=torch.long)
        local_idx = torch.as_tensor(local_idx, dtype=torch.long)

        Xu = X[users_t[obs_idx]]              # (N_C, f)
        c = conf_t[obs_idx]
        pv = p_t[obs_idx]
        # A_i = XᵀX + Σ_{u∈U_i}(c_ui−1)x_u x_uᵀ + λI（批量 scatter-add）
        outer = Xu[:, :, None] * Xu[:, None, :] * (c - 1.0)[:, None, None]
        A_chunk = XtX.expand(len(ids), f, f).clone()
        A_chunk.index_add_(0, local_idx, outer)
        A_chunk += lam * eye
        # b_i = Σ_{u∈U_i} c_ui p_ui x_u
        b_chunk = torch.zeros(len(ids), f, dtype=torch.float64)
        b_chunk.index_add_(0, local_idx, Xu * (c * pv)[:, None])
        Y_new[torch.as_tensor(ids, dtype=torch.long)] = torch.linalg.solve(
            A_chunk, b_chunk.unsqueeze(-1)
        ).squeeze(-1)
    return X.numpy(), Y_new.numpy()


def _wmf_loss(X, Y, users, items, conf, p, lam):
    """论文 Eq.(3) 全量损失（含未观测项 c=1、p=0）。

    full = Σ_ui c_ui(p_ui − s_ui)² + λ(Σ‖x‖² + Σ‖y‖²)
         = [Σ_all s² + Σ_obs (c(p−s)² − s²)] + λ(Σ‖x‖² + Σ‖y‖²)

    Σ_all s² = trace(X YᵀY Xᵀ)，O(m f²) 而不是 O(m n f)。

    Returns:
        (full, obs, reg)：full=含正则全量损失，obs=观测项修正值，
        reg=正则项 λ(‖X‖²+‖Y‖²)
    """
    YtY = Y.T @ Y
    s_all = float(np.sum((X @ YtY) * X))  # trace(X YᵀY Xᵀ) = Σ_all s²
    Xu = X[users]
    Yi = Y[items]
    s = np.sum(Xu * Yi, axis=1)
    obs = float(np.sum(conf * (p - s) ** 2 - s ** 2))
    reg = float(lam * (np.sum(X * X) + np.sum(Y * Y)))
    return s_all + obs + reg, obs, reg


class WMFModel(TrainableModel):
    """WMF（Hu, Koren & Volinsky 2008）双因子隐式反馈模型。"""

    def __init__(self, config: TrainingConfig, num_users: int, num_items: int,
                 edge_index: Optional[torch.Tensor] = None):
        super().__init__(config)
        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.factors = int(config.get(KEY_FACTORS, 100))
        # edge_index 仅用于兼容攻击模板的统一构造签名
        # model_cls(cfg, num_users, num_items, edge_index)；WMF 为全量 ALS，
        # 不消费图结构，参数被忽略。
        del edge_index

        self.user_factors = nn.Parameter(
            torch.empty(num_users, self.factors, dtype=torch.float32)
        )
        self.item_factors = nn.Parameter(
            torch.empty(num_items, self.factors, dtype=torch.float32)
        )
        self._init_factors(config.get(KEY_INIT_METHOD, "normal"))
        self.to(self._device)
        print(f"[WMFModel] users={num_users}, items={num_items}, "
              f"factors={self.factors}, device={self._device}")

    def _init_factors(self, method: str):
        """初始化因子表（理解文档 2.7：论文未说明，默认 N(0, 0.01)）。"""
        init_std = float(self.config.get(KEY_INIT_STD, 0.01))
        if method == "normal":
            nn.init.normal_(self.user_factors, mean=0.0, std=init_std)
            nn.init.normal_(self.item_factors, mean=0.0, std=init_std)
        elif method == "uniform":
            bound = init_std * (3.0 ** 0.5)  # 保持与 normal 同标准差
            nn.init.uniform_(self.user_factors, a=-bound, b=bound)
            nn.init.uniform_(self.item_factors, a=-bound, b=bound)
        else:
            raise ValueError(
                f"不支持的 init_method='{method}'，可选 normal | uniform"
            )
        print(f"[WMFModel] init={method}(std={init_std})")

    def forward(self, users: torch.Tensor, items: torch.Tensor):
        """预测分数 p̂_ui = x_uᵀ y_i。

        始终是“预测”语义（审阅④）：返回 (n,) 预测分数；
        因子表通过 get_user_embeddings / get_item_embeddings 获取。
        """
        x_u = self.user_factors[users]
        y_i = self.item_factors[items]
        return (x_u * y_i).sum(dim=1)

    def get_user_embeddings(self):
        """全量用户因子 (m, f)，攻击流程与全量评估共用。"""
        return self.user_factors

    def get_item_embeddings(self):
        """全量物品因子 (n, f)。"""
        return self.item_factors

    def predict_full_ranking(self, user_ids: torch.Tensor,
                             item_ids: Optional[torch.Tensor] = None,
                             batch_size: int = 1024):
        """全量排序评分（独立方法，不通过 eval_step）。

        Args:
            user_ids: (n_users,) 用户 id
            item_ids: 可选；给定则只对这批物品评分，列序与 item_ids 一致
        Returns:
            CPU tensor (n_users, n_items 或 len(item_ids))
        """
        self.eval()
        user_ids = user_ids.to(self._device)
        x = self.user_factors[user_ids]
        y = self.item_factors
        if item_ids is not None:
            y = y[item_ids.to(self._device)]
        scores_list = []
        for i in range(0, len(user_ids), batch_size):
            scores_list.append((x[i:i + batch_size] @ y.T).cpu())
        return torch.cat(scores_list, dim=0)

    def _numpy_state(self):
        X = self.user_factors.detach().cpu().numpy()
        Y = self.item_factors.detach().cpu().numpy()
        return X, Y

    def train_step(self, batch):
        """一个训练 step = 一轮 ALS 交替最小二乘（全量闭式更新，无优化器）。

        batch 来自 train_loader 的全量训练矩阵：
        (users, items, conf, p, user_obs, item_obs)。
        论文 ALS 是全量矩阵优化（Eq.4/5），不存在 mini-batch——每个 epoch
        只调用一次本方法，内部遍历全部用户与物品做闭式求解。

        返回 {"loss", "obs_loss", "reg"}，均为标量。
        """
        users, items, conf, p, user_obs, item_obs = batch
        users_np = users.cpu().numpy()
        items_np = items.cpu().numpy()
        conf_np = conf.cpu().numpy()
        p_np = p.cpu().numpy()
        lam = float(self.config.get(KEY_LAMBDA_REG, 0.01))

        with torch.no_grad():
            X, Y = self._numpy_state()
            X, Y = _als_sweep(X, Y, users_np, items_np, conf_np, p_np,
                              user_obs, item_obs, lam)
            self.user_factors.data.copy_(torch.from_numpy(X))
            self.item_factors.data.copy_(torch.from_numpy(Y))
            loss, obs, reg = _wmf_loss(X, Y, users_np, items_np,
                                       conf_np, p_np, lam)
        return {"loss": float(loss), "obs_loss": float(obs),
                "reg": float(reg)}

    def eval_step(self, batch):
        """验证损失：Eq.(3) 全量损失限制在本批用户上 + 全物品正则。

        S_val = Σ_{u∈batch} x_uᵀ(YᵀY)x_u + Σ_obs(c(p−s)² − s²)
                + λ(Σ_{u∈batch}‖x‖² + Σ_i‖y‖²)
        """
        users, items, conf, p = batch
        users_np = users.cpu().numpy()
        items_np = items.cpu().numpy()
        conf_np = conf.cpu().numpy()
        p_np = p.cpu().numpy()
        lam = float(self.config.get(KEY_LAMBDA_REG, 0.01))

        with torch.no_grad():
            X, Y = self._numpy_state()
            YtY = Y.T @ Y
            X_val = np.zeros_like(X)
            X_val[users_np] = X[users_np]
            s_all = float(np.sum((X_val @ YtY) * X_val))
            Xu = X[users_np]
            Yi = Y[items_np]
            s = np.sum(Xu * Yi, axis=1)
            obs = float(np.sum(conf_np * (p_np - s) ** 2 - s ** 2))
            reg = float(lam * (np.sum(X_val * X_val) + np.sum(Y * Y)))
        return {"val_loss": float(s_all + obs + reg)}

    def build_dataloader(self, config: TrainingConfig):
        from models.wmf.dataset import WMFDataLoader
        return WMFDataLoader(config)
