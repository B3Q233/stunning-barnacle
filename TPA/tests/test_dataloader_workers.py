"""DataLoader num_workers / persistent_workers 配置化单测。"""
import unittest

from models.lightgcn.dataset import LightGCNDataLoader
from models.mf.dataset import MFDataLoader
from models.wmf.dataset import WMFDataLoader
from training.framework import TrainingConfig


class DataLoaderWorkerConfigTest(unittest.TestCase):

    def test_lightgcn_reads_config(self):
        cfg = TrainingConfig(overrides={
            "dataset": "ml100k", "num_workers": 2, "persistent_workers": True,
        })
        loader = LightGCNDataLoader(cfg).train_loader()
        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.persistent_workers)

    def test_mf_reads_config(self):
        cfg = TrainingConfig(overrides={
            "dataset": "ml100k", "num_workers": 2, "persistent_workers": True,
        })
        loader = MFDataLoader(cfg).train_loader()
        self.assertEqual(loader.num_workers, 2)
        self.assertTrue(loader.persistent_workers)

    def test_wmf_reads_config(self):
        cfg = TrainingConfig(overrides={
            "dataset": "ml100k", "num_workers": 0, "persistent_workers": False,
        })
        loader = WMFDataLoader(cfg).train_loader()
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)

    def test_default_fallback(self):
        cfg = TrainingConfig(overrides={"dataset": "ml100k"})
        loader = LightGCNDataLoader(cfg).train_loader()
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)
