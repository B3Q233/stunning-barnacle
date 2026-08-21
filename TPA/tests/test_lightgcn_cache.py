"""LightGCN 传播缓存与 CSR 存储单测（CPU 小图，不依赖 CUDA）。"""
import unittest

import torch
import torch.nn.functional as F

from models.lightgcn.model import LightGCN
from training.framework import TrainingConfig


def _make_model(seed=0):
    torch.manual_seed(seed)
    cfg = TrainingConfig(overrides={
        "device": "cpu", "emb_dim": 8, "n_layers": 2,
        "lr": 0.001, "weight_decay": 1e-4,
    })
    edge_index = torch.LongTensor([[0, 1, 2, 0], [0, 1, 2, 3]])
    return LightGCN(cfg, 3, 4, edge_index)


class CsrPropagationTest(unittest.TestCase):

    def test_csr_matches_coo(self):
        model = _make_model()
        emb = torch.randn(7, 8)
        coo = model.A_hat.to_sparse_coo()
        out_csr = torch.sparse.mm(model.A_hat, emb)
        out_coo = torch.sparse.mm(coo, emb)
        self.assertTrue(torch.allclose(out_csr, out_coo, atol=1e-6))


class TrainStepCacheTest(unittest.TestCase):

    def test_cached_step_matches_reference(self):
        model = _make_model()
        weights0 = model.embedding.weight.detach().clone()
        users = torch.LongTensor([0, 1, 2])
        pos_items = torch.LongTensor([0, 1, 2])
        neg_items = torch.LongTensor([[1], [0], [3]])
        batch = (users, pos_items, neg_items)

        # 参考实现：旧版语义——两次独立 forward，各自重算 final_emb
        ref = _make_model(seed=1)
        ref.embedding.weight.data.copy_(weights0)
        batch_size, neg_ratio = users.size(0), neg_items.size(1)
        pos_scores = ref.forward(users, pos_items)
        users_expanded = users.unsqueeze(1).expand(-1, neg_ratio).reshape(-1)
        neg_scores = ref.forward(users_expanded, neg_items.reshape(-1))
        neg_scores = neg_scores.view(batch_size, neg_ratio)
        bpr = -torch.mean(F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores))
        user_ego = ref.embedding.weight[users]
        pos_ego = ref.embedding.weight[ref.num_users + pos_items]
        neg_ego = ref.embedding.weight[ref.num_users + neg_items]
        reg = (0.5) * (user_ego.norm(p=2).pow(2) + pos_ego.norm(p=2).pow(2)
                       + neg_ego.norm(p=2).pow(2)) / users.size(0)
        reg = reg * ref.config.get("weight_decay", 1e-4)
        ref_loss = (bpr + reg).item()
        (bpr + reg).backward()
        ref_grad = ref.embedding.weight.grad.clone()

        # 新实现：train_step 内单次传播
        model.embedding.weight.data.copy_(weights0)
        out = model.train_step(batch)
        new_grad = model.embedding.weight.grad.clone()

        self.assertAlmostEqual(out["loss"], ref_loss, places=6)
        self.assertTrue(torch.allclose(new_grad, ref_grad, atol=1e-6))


class ForwardFinalEmbTest(unittest.TestCase):

    def test_forward_final_emb_reuse(self):
        model = _make_model()
        users = torch.LongTensor([0, 1])
        items = torch.LongTensor([0, 1])
        final = model._compute_final_emb()
        with_cache = model.forward(users, items, final_emb=final)
        without = model.forward(users, items)
        self.assertTrue(torch.allclose(with_cache, without, atol=1e-6))
