"""WMF 模型训练（步骤⑤）单元测试：ALS 正确性与标量契约"""
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from models.wmf.config_keys import KEY_FACTORS, KEY_INIT_STD, KEY_LAMBDA_REG
from models.wmf.dataset import WMFDataset, group_observations
from models.wmf.model import WMFModel, _als_sweep, _wmf_loss
from training.framework import TrainingConfig


PAIRS = [(0, 1), (0, 2), (1, 3), (1, 0), (2, 1), (2, 2)]


def make_model(factors=4, lam=0.01, num_users=3, num_items=4):
    config = TrainingConfig(overrides={
        KEY_FACTORS: factors,
        KEY_LAMBDA_REG: lam,
        KEY_INIT_STD: 0.01,
        "device": "cpu",
    })
    return WMFModel(config, num_users=num_users, num_items=num_items)


def make_batch(pairs=PAIRS):
    """模拟 train_loader 的全量训练矩阵 batch（6 元组）。"""
    ds = WMFDataset(pairs)
    return (ds.users, ds.items, ds.conf, ds.p, ds.user_obs, ds.item_obs)


class ALSSolveTest(unittest.TestCase):
    """ALS 闭式解验证：解必须满足论文 Eq.(4) 的正常方程。"""

    def test_solution_satisfies_normal_equations(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        users = np.array([u for u, _ in PAIRS])
        items = np.array([i for _, i in PAIRS])
        conf = np.full(len(PAIRS), 41.0)
        p = np.ones(len(PAIRS))
        X0 = rng.normal(0, 0.01, (3, 4))
        Y0 = rng.normal(0, 0.01, (4, 4))
        lam = 0.01
        user_obs, item_obs = group_observations(users.tolist(),
                                                items.tolist())

        X, Y = _als_sweep(X0, Y0, users, items, conf, p,
                          user_obs, item_obs, lam)

        # 用户侧：A_u x_u == b_u（用求解时的旧 Y0 验证）
        YtY = Y0.T @ Y0
        for u in range(3):
            mask = users == u
            Yu = Y0[items[mask]]
            A = YtY + (Yu.T * (conf[mask] - 1.0)) @ Yu \
                + lam * np.eye(4)
            b = Yu.T @ (conf[mask] * p[mask])
            np.testing.assert_allclose(A @ X[u], b, atol=1e-6)

        # 物品侧对称验证
        XtX = X.T @ X
        for i in range(4):
            mask = items == i
            Xu = X[users[mask]]
            A = XtX + (Xu.T * (conf[mask] - 1.0)) @ Xu \
                + lam * np.eye(4)
            b = Xu.T @ (conf[mask] * p[mask])
            np.testing.assert_allclose(A @ Y[i], b, atol=1e-6)

    def test_full_loss_matches_brute_force(self):
        """Eq.(3) 加速式（含未观测项）与 O(m·n) 全量展开一致。"""
        rng = np.random.default_rng(1)
        X = rng.normal(0, 0.1, (3, 4))
        Y = rng.normal(0, 0.1, (4, 4))
        users = np.array([u for u, _ in PAIRS])
        items = np.array([i for _, i in PAIRS])
        conf = np.full(len(PAIRS), 41.0)
        p = np.ones(len(PAIRS))
        lam = 0.01
        user_obs, item_obs = group_observations(users.tolist(),
                                                items.tolist())

        s = X @ Y.T
        c_mat = np.ones((3, 4))
        p_mat = np.zeros((3, 4))
        for u, i, c, pv in zip(users, items, conf, p):
            c_mat[u, i] = c
            p_mat[u, i] = pv
        brute = float(
            np.sum(c_mat * (p_mat - s) ** 2)
            + lam * (np.sum(X ** 2) + np.sum(Y ** 2))
        )

        full, obs, reg = _wmf_loss(X, Y, users, items, conf, p, lam)
        self.assertAlmostEqual(full, brute, places=6)
        self.assertAlmostEqual(full, obs + np.sum(s ** 2) + reg, places=6)


class WMFModelTrainStepTest(unittest.TestCase):
    """train_step / eval_step：1 batch 1 epoch 最小验证 + 标量契约。"""

    def setUp(self):
        torch.manual_seed(0)
        self.model = make_model()
        self.batch = make_batch()

    def _initial_loss(self):
        X = self.model.user_factors.detach().cpu().numpy()
        Y = self.model.item_factors.detach().cpu().numpy()
        users = self.batch[0].numpy()
        items = self.batch[1].numpy()
        conf = self.batch[2].numpy()
        p = self.batch[3].numpy()
        return _wmf_loss(X, Y, users, items, conf, p,
                         float(self.model.config.get(KEY_LAMBDA_REG, 0.01)))[0]

    def test_train_step_loss_finite_and_decreases(self):
        before = self._initial_loss()
        X_before = self.model.user_factors.detach().clone()

        metrics = self.model.train_step(self.batch)

        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(metrics["loss"] < 1e4, f"loss 异常偏大: {metrics['loss']}")
        self.assertLess(metrics["loss"], before,
                        "ALS 一轮后全量损失应下降")
        self.assertFalse(
            torch.equal(X_before, self.model.user_factors.detach()),
            "参数未更新",
        )

    def test_train_step_scalar_contract(self):
        metrics = self.model.train_step(self.batch)
        for k, v in metrics.items():
            self.assertIsInstance(v, float, f"train_step 指标 {k} 不是标量")

    def test_eval_step_scalar_contract(self):
        ds = WMFDataset(PAIRS[:3])
        val_batch = (ds.users, ds.items, ds.conf, ds.p)
        self.model.eval()
        with torch.no_grad():
            out = self.model.eval_step(val_batch)
        self.assertIn("val_loss", out)
        for k, v in out.items():
            self.assertIsInstance(v, float, f"eval_step 指标 {k} 不是标量")
        self.assertTrue(np.isfinite(out["val_loss"]))


if __name__ == "__main__":
    unittest.main()
