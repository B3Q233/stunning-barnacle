"""FullRankingCallback 快速评估路径接线单测（spy 校验新参数已传入）。"""
import tempfile
import unittest

import torch

import models.lightgcn.train as lightgcn_train
import models.mf.train as mf_train
from models.lightgcn.dataset import LightGCNDataLoader
from models.lightgcn.model import LightGCN
from models.mf.dataset import MFDataLoader
from models.mf.model import MatrixFactorization
from training.framework import TrainingConfig


def _cfg():
    return TrainingConfig(overrides={
        "dataset": "ml100k", "emb_dim": 8, "n_layers": 2,
        "batch_size": 64, "lr": 0.001, "weight_decay": 1e-4,
        "device": "cpu", "k": 5, "eval_every": 1,
        "metrics": [{"recall@5": "upper"}], "checkpoint_mode": "per_metric",
        "num_workers": 0,
    })


def _prime_optimizer(model):
    users = torch.LongTensor([0, 1])
    pos = torch.LongTensor([0, 1])
    neg = torch.LongTensor([[1], [2]])
    model.train_step((users, pos, neg))


class LightGCNCallbackFastPathTest(unittest.TestCase):

    def test_callback_passes_mask_indices_and_topk_device(self):
        cfg = _cfg()
        loader = LightGCNDataLoader(cfg)
        edge_index = torch.LongTensor(
            [[u, i] for u, i in loader.all_train_pairs]).T
        model = LightGCN(cfg, loader.num_users, loader.num_items, edge_index)
        _prime_optimizer(model)
        original = lightgcn_train.compute_metrics
        seen = {}

        def spy(scores, train_user_items, test_user_items, k=20, **kwargs):
            seen.update(kwargs)
            return original(scores, train_user_items, test_user_items, k=k)

        lightgcn_train.compute_metrics = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cb = lightgcn_train.FullRankingCallback(loader, model, cfg, tmp)
                result = cb.on_epoch_end(1, {})
        finally:
            lightgcn_train.compute_metrics = original
        self.assertIn("mask_indices", seen)
        self.assertIn("topk_device", seen)
        self.assertIn("recall@5", result)


class MFCallbackFastPathTest(unittest.TestCase):

    def test_callback_passes_mask_indices_and_topk_device(self):
        cfg = _cfg()
        loader = MFDataLoader(cfg)
        model = MatrixFactorization(cfg, loader.num_users, loader.num_items)
        _prime_optimizer(model)
        original = mf_train.compute_metrics
        seen = {}

        def spy(scores, train_user_items, test_user_items, k=20, **kwargs):
            seen.update(kwargs)
            return original(scores, train_user_items, test_user_items, k=k)

        mf_train.compute_metrics = spy
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cb = mf_train.FullRankingCallback(loader, model, cfg, tmp)
                result = cb.on_epoch_end(1, {})
        finally:
            mf_train.compute_metrics = original
        self.assertIn("mask_indices", seen)
        self.assertIn("topk_device", seen)
        self.assertIn("recall@5", result)
