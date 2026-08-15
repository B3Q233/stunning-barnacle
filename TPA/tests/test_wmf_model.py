"""WMF 模型结构（步骤③）单元测试（unittest）"""
import unittest

import torch

from models.wmf.config_keys import KEY_FACTORS, KEY_INIT_METHOD, KEY_INIT_STD
from models.wmf.model import WMFModel
from training.framework import TrainingConfig


def make_config(factors=4, init_std=0.01):
    return TrainingConfig(overrides={
        KEY_FACTORS: factors,
        KEY_INIT_METHOD: "normal",
        KEY_INIT_STD: init_std,
        "device": "cpu",
    })


class WMFModelStructureTest(unittest.TestCase):
    """结构：双因子表、forward 输出形状、初始化分布。"""

    def setUp(self):
        self.model = WMFModel(make_config(), num_users=3, num_items=4)

    def test_forward_shape(self):
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 1, 3])
        out = self.model.forward(users, items)
        self.assertEqual(out.shape, (3,))
        self.assertEqual(out.dtype, torch.float32)

    def test_embedding_shapes(self):
        self.assertEqual(self.model.get_user_embeddings().shape, (3, 4))
        self.assertEqual(self.model.get_item_embeddings().shape, (4, 4))

    def test_predict_full_ranking_shape(self):
        scores = self.model.predict_full_ranking(torch.tensor([0, 2]))
        self.assertEqual(scores.shape, (2, 4))
        selected = self.model.predict_full_ranking(
            torch.tensor([0, 2]), item_ids=torch.tensor([1, 3])
        )
        self.assertEqual(selected.shape, (2, 2))

    def test_state_dict_keys(self):
        keys = set(self.model.state_dict().keys())
        self.assertIn("user_factors", keys)
        self.assertIn("item_factors", keys)

    def test_constructor_accepts_edge_index(self):
        """攻击模板构造签名：model_cls(cfg, num_users, num_items, edge_index)。"""
        model = WMFModel(make_config(), num_users=3, num_items=4,
                         edge_index=torch.zeros((2, 5), dtype=torch.long))
        self.assertEqual(model.num_users, 3)
        self.assertEqual(model.item_factors.shape, (4, 4))

    def test_init_distribution_normal(self):
        """初始化自检：N(0, 0.01)，均值≈0、std≈0.01（大样本）。"""
        model = WMFModel(make_config(factors=32, init_std=0.01),
                         num_users=200, num_items=200)
        X = model.user_factors.detach()
        self.assertAlmostEqual(X.mean().item(), 0.0, places=2)
        self.assertAlmostEqual(X.std().item(), 0.01, places=2)
        Y = model.item_factors.detach()
        self.assertAlmostEqual(Y.std().item(), 0.01, places=2)

    def test_init_method_uniform_bound(self):
        config = make_config(factors=8, init_std=0.1)
        config[KEY_INIT_METHOD] = "uniform"
        model = WMFModel(config,
                         num_users=20, num_items=20)
        X = model.user_factors.detach()
        # 均匀分布标准差 = bound / sqrt(3) -> bound = std * sqrt(3)
        self.assertGreaterEqual(X.min().item(), -0.2)
        self.assertLessEqual(X.max().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
