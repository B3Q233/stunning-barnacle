"""WMF 数据导入（步骤②）单元测试（unittest）"""
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from models.wmf.config_keys import (
    KEY_ALPHA,
    KEY_CONFIDENCE_SCHEME,
    KEY_DATASET,
    KEY_NUM_ITEMS,
    KEY_NUM_USERS,
    KEY_PROCESSED_DATA_PATH,
    KEY_VAL_RATIO,
)
from models.wmf.dataset import WMFDataLoader, WMFDataset
from models.wmf.scripts.preprocess import build_meta
from training.framework import TrainingConfig


TRAIN = [(0, 1), (0, 2), (1, 3), (1, 0), (2, 1), (2, 2)]
TEST = [(0, 1), (1, 2)]


def write_meta(meta_dir: Path):
    meta = build_meta(TRAIN, TEST)
    with open(meta_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta


class WMFDatasetTest(unittest.TestCase):
    """置信度公式（论文 Eq.2/3）与 batch 字段格式。"""

    def test_minimal_confidence(self):
        ds = WMFDataset(TRAIN, alpha=40.0, scheme="minimal")
        self.assertEqual(len(ds), 6)
        users, items, conf, p = ds[0]
        self.assertEqual(int(users), 0)
        self.assertEqual(int(items), 1)
        self.assertAlmostEqual(float(conf), 41.0)   # 1 + 40*1
        self.assertEqual(float(p), 1.0)             # r_ui > 0 -> p_ui = 1

    def test_log_scaling_confidence(self):
        ds = WMFDataset(TRAIN, alpha=40.0, epsilon=1e-8,
                        scheme="log-scaling")
        expected = 1.0 + 40.0 * math.log1p(1.0 / 1e-8)
        self.assertAlmostEqual(float(ds.conf[0]), expected, places=3)

    def test_prebuilt_observation_groups(self):
        """user_obs / item_obs 预构建一次，ALS 每轮直接引用（审阅②）。"""
        ds = WMFDataset(TRAIN)
        # TRAIN = [(0,1),(0,2),(1,3),(1,0),(2,1),(2,2)]
        self.assertEqual(ds.user_obs[0], [0, 1])
        self.assertEqual(ds.user_obs[1], [2, 3])
        self.assertEqual(ds.user_obs[2], [4, 5])
        self.assertEqual(ds.item_obs[1], [0, 4])
        self.assertEqual(ds.item_obs[2], [1, 5])
        self.assertEqual(ds.item_obs[3], [2])

    def test_invalid_scheme(self):
        with self.assertRaises(ValueError):
            WMFDataset(TRAIN, scheme="unknown")


class WMFDataLoaderTest(unittest.TestCase):
    """五方法协议与 batch 格式（对照理解文档 2.1）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.meta_dir = Path(self._tmp.name) / "ml100k"
        self.meta_dir.mkdir()
        write_meta(self.meta_dir)

    def tearDown(self):
        self._tmp.cleanup()

    def make_loader(self, val_ratio=0.5):
        config = TrainingConfig(overrides={
            KEY_DATASET: "ml100k",
            KEY_PROCESSED_DATA_PATH: str(self.meta_dir),
            KEY_VAL_RATIO: val_ratio,
            KEY_ALPHA: 40.0,
            KEY_CONFIDENCE_SCHEME: "minimal",
        })
        return WMFDataLoader(config)

    def test_init_params_and_split(self):
        loader = self.make_loader()
        self.assertEqual(loader.num_users, 3)
        self.assertEqual(loader.num_items, 4)
        params = loader.get_init_params()
        self.assertEqual(params[KEY_NUM_USERS], 3)
        self.assertEqual(params[KEY_NUM_ITEMS], 4)
        # 6 条训练对，val_ratio=0.5 -> 3/3
        self.assertEqual(len(loader._train_pairs), 3)
        self.assertEqual(len(loader._val_pairs), 3)
        self.assertEqual(len(loader.test_pairs), 2)

    def test_train_batch_shape_dtype(self):
        loader = self.make_loader()
        batch = next(iter(loader.train_loader()))
        # train_loader 返回全量训练矩阵 6 元组（审阅①：无 mini-batch）
        self.assertEqual(len(batch), 6)
        users, items, conf, p, user_obs, item_obs = batch
        self.assertEqual(users.shape, (3,))
        self.assertEqual(users.dtype, torch.long)
        self.assertEqual(items.shape, (3,))
        self.assertEqual(items.dtype, torch.long)
        self.assertEqual(conf.shape, (3,))
        self.assertEqual(conf.dtype, torch.float32)
        self.assertEqual(p.shape, (3,))
        self.assertTrue(torch.all(conf == 41.0))
        self.assertTrue(torch.all(p == 1.0))
        self.assertIn(0, user_obs)
        self.assertIn(3, item_obs)

    def test_split_datasets(self):
        loader = self.make_loader()
        train_ds = loader.get_dataset("train")
        val_ds = loader.get_dataset("val")
        test_ds = loader.get_dataset("test")
        self.assertEqual(len(train_ds), 3)
        self.assertEqual(len(val_ds), 3)
        self.assertEqual(len(test_ds), 2)

    def test_protocol_methods_present(self):
        loader = self.make_loader()
        for name in ("train_loader", "val_loader", "test_loader",
                     "get_init_params", "get_dataset"):
            self.assertTrue(callable(getattr(loader, name)),
                            f"缺少 DatasetProtocol 方法 {name}")


if __name__ == "__main__":
    unittest.main()
