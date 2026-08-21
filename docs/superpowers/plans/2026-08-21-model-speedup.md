# 模型训练与评估提速（方案 A）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 本会话按 multi_agent_mode 不启用子代理，采用 inline 执行（executing-plans），每个任务结束跑测试并提交。

**Goal:** 在保持训练/评估结果逐位一致的前提下，将 LightGCN/MF 的训练与评估提速（gowalla 单 epoch 训练 ~10.9 分钟 → ~2 分钟量级，评估单次 ~11.4 s → ~2 s 量级）。

**Architecture:** ① LightGCN `train_step/eval_step` 内只传播一次并复用 `final_emb`，邻接矩阵改 CSR；② 评估掩码索引只构建一次，topk 分块到 GPU；③ 三模型 DataLoader 的 `num_workers`/`persistent_workers` 全部配置化。

**Tech Stack:** Python 3.12 / PyTorch 2.5.1+cu121 / stdlib unittest / scipy.sparse

## Global Constraints

- 测试只允许 stdlib unittest，文件放 `TPA/tests/test_*.py`；运行命令（在 `G:\Idea\TPA` 下）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`。
- 仓库文档与提交信息使用中文；提交格式 Conventional Commits `type(scope): 中文描述`。
- 只用 `git add` 加明确路径，禁止 `git add -A` / `git add -f`。
- `compute_metrics` / `_topk_by_batch` / `rank_values` / `expected_percentile_rank` 的默认签名与行为不变（攻击链路与 WMF 依赖）。
- DataLoader 配置缺省时行为保持旧值（`num_workers=0`、`persistent_workers=False`）；`persistent_workers=True` 仅在 `num_workers>0` 时传给 DataLoader，避免其报错。
- 单测全部 CPU 可跑（不得依赖 CUDA）。

---

### Task 1: 评估快速路径（掩码索引预计算 + 分块 topk）

**Files:**
- Modify: `TPA/evaluation/metrics.py`
- Test: `TPA/tests/test_evaluation_fast.py`

**Interfaces:**
- Produces: `build_train_mask_indices(train_user_items: Dict[int, set], user_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]`，rows 为分数矩阵行号（按 `user_ids` 顺序），cols 为物品 id。
- Produces: `compute_metrics(scores, train_user_items, test_user_items, k=20, mask_indices=None, topk_device=None, chunk_size=1024)`；新参数全缺省时行为与旧版一致。

- [ ] **Step 1: 写失败测试**

新建 `TPA/tests/test_evaluation_fast.py`：

```python
"""评估快速路径单测：掩码索引预计算 + 分块 topk 与默认路径逐位一致。"""
import unittest

import torch

from evaluation.metrics import build_train_mask_indices, compute_metrics


class BuildTrainMaskIndicesTest(unittest.TestCase):

    def test_mask_indices_cover_expected_entries(self):
        train_user_items = {0: {1, 2}, 2: {5}, 4: {0, 7}}
        rows, cols = build_train_mask_indices(train_user_items, list(range(6)))
        got = set(zip(rows.tolist(), cols.tolist()))
        expected = {(0, 1), (0, 2), (2, 5), (4, 0), (4, 7)}
        self.assertEqual(got, expected)


class ComputeMetricsFastPathTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.scores = torch.randn(6, 10)
        self.train_user_items = {0: {1, 2}, 2: {5}, 4: {0, 7}}
        self.test_user_items = {0: {3}, 2: {4}, 5: {8}}
        self.rows, self.cols = build_train_mask_indices(
            self.train_user_items, list(range(6)))

    def _assert_same(self, a, b):
        self.assertEqual(sorted(a), sorted(b))
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=10)

    def test_mask_indices_equals_default(self):
        default = compute_metrics(self.scores.clone(), self.train_user_items,
                                  self.test_user_items, k=3)
        fast = compute_metrics(self.scores.clone(), self.train_user_items,
                               self.test_user_items, k=3,
                               mask_indices=(self.rows, self.cols))
        self._assert_same(default, fast)

    def test_chunked_topk_equals_default(self):
        default = compute_metrics(self.scores.clone(), self.train_user_items,
                                  self.test_user_items, k=3)
        fast = compute_metrics(self.scores.clone(), self.train_user_items,
                               self.test_user_items, k=3,
                               mask_indices=(self.rows, self.cols),
                               topk_device="cpu", chunk_size=2)
        self._assert_same(default, fast)
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_evaluation_fast -v`
Expected: FAIL with `ImportError: cannot import name 'build_train_mask_indices'`。

- [ ] **Step 3: 最小实现**

在 `TPA/evaluation/metrics.py` 中新增 `build_train_mask_indices`，并给 `_topk_by_batch` / `compute_metrics` 增加可选参数：

```python
def build_train_mask_indices(train_user_items, user_ids):
    """把训练集已交互物品掩码编译为 (rows, cols) 索引张量，供向量化掩码复用。"""
    rows = []
    cols = []
    for r, u in enumerate(user_ids):
        items = train_user_items.get(u)
        if items:
            rows.extend([r] * len(items))
            cols.extend(items)
    return torch.LongTensor(rows), torch.LongTensor(cols)
```

`_topk_by_batch` 签名改为：

```python
def _topk_by_batch(scores: torch.Tensor, k: int,
                   train_user_items: Dict[int, set],
                   test_user_items: Dict[int, set],
                   mask_indices=None, topk_device=None,
                   chunk_size: int = 1024):
```

函数体开头（替换原掩码循环）：

```python
    n_users = scores.shape[0]
    if mask_indices is None:
        for u in range(n_users):
            if u in train_user_items:
                for i in train_user_items[u]:
                    scores[u, i] = float('-inf')
    else:
        rows, cols = mask_indices
        scores[rows.to(scores.device), cols.to(scores.device)] = float('-inf')
```

topk 部分（替换原 `_, topk_indices = torch.topk(scores, k, dim=1)`）：

```python
    if topk_device is None:
        _, topk_indices = torch.topk(scores, k, dim=1)
    else:
        topk_indices = torch.empty((n_users, k), dtype=torch.long)
        for start in range(0, n_users, chunk_size):
            chunk = scores[start:start + chunk_size].to(topk_device)
            _, idx = torch.topk(chunk, k, dim=1)
            topk_indices[start:start + chunk_size] = idx.to("cpu")
```

`compute_metrics` 改为：

```python
def compute_metrics(scores: torch.Tensor, train_user_items: Dict[int, set],
                    test_user_items: Dict[int, set],
                    k: int = 20, mask_indices=None, topk_device=None,
                    chunk_size: int = 1024) -> Dict[str, float]:
    recall, ndcg = _topk_by_batch(
        scores, k, train_user_items, test_user_items,
        mask_indices=mask_indices, topk_device=topk_device,
        chunk_size=chunk_size)
    return {f"recall@{k}": recall, f"ndcg@{k}": ndcg}
```

- [ ] **Step 4: 运行测试确认通过**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_evaluation_fast -v`
Expected: `OK`（3 个用例全过）。

- [ ] **Step 5: 提交**

```bash
git add TPA/evaluation/metrics.py TPA/tests/test_evaluation_fast.py
git commit -m "feat(eval): 评估掩码索引预计算与分块 topk 快速路径"
```

---

### Task 2: LightGCN 传播缓存 + CSR 邻接矩阵

**Files:**
- Modify: `TPA/models/lightgcn/model.py`
- Test: `TPA/tests/test_lightgcn_cache.py`

**Interfaces:**
- Consumes: 无（Task 1 独立）。
- Produces: `forward(users, items=None, final_emb=None)`；`self.A_hat` 变为 CSR 格式；`train_step/eval_step` 内部单次传播。

- [ ] **Step 1: 写失败测试**

新建 `TPA/tests/test_lightgcn_cache.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_lightgcn_cache -v`
Expected: FAIL——`test_cached_step_matches_reference` 中 `ref.forward` 与旧实现一致，但新实现 `train_step` 尚未改（loss/grad 仍相等则说明测试未覆盖新行为，需确认失败原因）；`test_forward_final_emb_reuse` 因 `forward` 不接受 `final_emb` 报 `TypeError`。

> 说明：`test_forward_final_emb_reuse` 是"新接口缺失"型失败（TypeError），`test_cached_step_matches_reference` 在改完后验证数值等价。两者共同构成 Task 2 的红-绿约束。

- [ ] **Step 3: 最小实现**

`TPA/models/lightgcn/model.py`：

`_build_adj` 末尾（生成 COO 后）追加一行，转为 CSR：

```python
        self.A_hat = self.A_hat.to_sparse_csr()
```

`forward` 改为：

```python
    def forward(self, users: torch.Tensor, items: torch.Tensor = None,
                final_emb: torch.Tensor = None):
        if final_emb is None:
            final_emb = self._compute_final_emb()
        if items is not None:
            user_emb = final_emb[users]
            item_emb = final_emb[self.num_users + items]
            return (user_emb * item_emb).sum(dim=1)
        return final_emb
```

`train_step` 中 `pos_scores` 之前加一行，并把两次 `self.forward(...)` 调用改为传入 `final_emb=final_emb`：

```python
        final_emb = self._compute_final_emb()
        pos_scores = self.forward(users, pos_items, final_emb=final_emb)
```

```python
        neg_scores = self.forward(users_expanded, neg_items.reshape(-1),
                                  final_emb=final_emb)
```

`eval_step` 同理：`with torch.no_grad():` 内先 `final_emb = self._compute_final_emb()`，两处 `forward` 都传 `final_emb=final_emb`。

- [ ] **Step 4: 运行测试确认通过**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_lightgcn_cache -v`
Expected: `OK`。

- [ ] **Step 5: 提交**

```bash
git add TPA/models/lightgcn/model.py TPA/tests/test_lightgcn_cache.py
git commit -m "feat(models): LightGCN 传播缓存与 CSR 邻接矩阵提速"
```

---

### Task 3: 三模型 DataLoader 配置化

**Files:**
- Modify: `TPA/models/lightgcn/dataset.py`、`TPA/models/mf/dataset.py`、`TPA/models/wmf/dataset.py`
- Modify: `TPA/models/lightgcn/config.yaml`、`TPA/models/mf/config.yaml`、`TPA/models/wmf/config.yaml`
- Test: `TPA/tests/test_dataloader_workers.py`

**Interfaces:**
- Consumes: 无。
- Produces: 三个 Loader 从 `config.get("num_workers", 0)` / `config.get("persistent_workers", False)` 读取 DataLoader 参数。

- [ ] **Step 1: 写失败测试**

新建 `TPA/tests/test_dataloader_workers.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_dataloader_workers -v`
Expected: FAIL——`lightgcn` / `mf` 的 loader 仍是 `num_workers=0`，`test_lightgcn_reads_config` 断言 `num_workers == 2` 失败。

- [ ] **Step 3: 最小实现**

三个 dataset.py 各加一个私有辅助方法：

`TPA/models/lightgcn/dataset.py`、`TPA/models/mf/dataset.py`、`TPA/models/wmf/dataset.py` 中分别加入：

```python
    def _loader_kwargs(self) -> Dict[str, Any]:
        num_workers = int(self.config.get("num_workers", 0))
        persistent = bool(self.config.get("persistent_workers", False))
        return {
            "num_workers": num_workers,
            "persistent_workers": persistent and num_workers > 0,
        }
```

然后：
- lightgcn 的 `train_loader` / `val_loader` / `test_loader`：`num_workers=0` 改为 `**self._loader_kwargs()`。
- mf 的 `train_loader` / `val_loader` / `test_loader`：同上。
- wmf 的 `_full_loader` 与 `train_loader`：`num_workers=0` 改为 `**self._loader_kwargs()`。

三个 config.yaml 的 `training:` 段末尾追加：

lightgcn / mf：
```yaml
  num_workers: 4                  # DataLoader 子进程数（0=主进程串行）
  persistent_workers: true        # 跨 epoch 复用 worker（需 num_workers>0）
```

wmf：
```yaml
  num_workers: 0                  # 单批全量矩阵，worker 无收益；0=主进程
  persistent_workers: false       # 需 num_workers>0 才生效
```

- [ ] **Step 4: 运行测试确认通过**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_dataloader_workers -v`
Expected: `OK`。

- [ ] **Step 5: 提交**

```bash
git add TPA/models/lightgcn/dataset.py TPA/models/mf/dataset.py TPA/models/wmf/dataset.py TPA/models/lightgcn/config.yaml TPA/models/mf/config.yaml TPA/models/wmf/config.yaml TPA/tests/test_dataloader_workers.py
git commit -m "feat(dataset): 三模型 DataLoader 并发参数配置化"
```

---

### Task 4: lightgcn / mf 评估回调接入快速路径

**Files:**
- Modify: `TPA/models/lightgcn/train.py`、`TPA/models/mf/train.py`
- Test: `TPA/tests/test_full_ranking_callback.py`

**Interfaces:**
- Consumes: Task 1 的 `build_train_mask_indices` 与 `compute_metrics(mask_indices=..., topk_device=...)`。
- Produces: 两个 `FullRankingCallback` 在 `__init__` 预计算 `self.test_users` / `self.mask_indices`，评估时传入快速路径参数。

- [ ] **Step 1: 写失败测试**

新建 `TPA/tests/test_full_ranking_callback.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_full_ranking_callback -v`
Expected: FAIL——`seen` 为空，`assertIn("mask_indices", seen)` 不通过（回调尚未传新参数）。

- [ ] **Step 3: 最小实现**

`TPA/models/lightgcn/train.py` 与 `TPA/models/mf/train.py`：

import 行把 `from evaluation.metrics import compute_metrics` 改为：

```python
from evaluation.metrics import build_train_mask_indices, compute_metrics
```

`__init__` 中构建 `self.test_pos` / `self.train_pos` 之后追加：

```python
        self.test_users = sorted(self.test_pos.keys())
        self.mask_indices = build_train_mask_indices(
            self.train_pos, self.test_users)
```

`on_epoch_end` 中把 `test_users = sorted(self.test_pos.keys())` 替换为：

```python
            test_users = self.test_users
```

并把 `compute_metrics(...)` 调用改为：

```python
            K: compute_metrics(scores, self.train_pos, self.test_pos, k=K,
                               mask_indices=self.mask_indices,
                               topk_device=user_emb.device)
```

- [ ] **Step 4: 运行测试确认通过**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_full_ranking_callback -v`
Expected: `OK`。

- [ ] **Step 5: 提交**

```bash
git add TPA/models/lightgcn/train.py TPA/models/mf/train.py TPA/tests/test_full_ranking_callback.py
git commit -m "feat(eval): lightgcn/mf 评估回调接入掩码预计算与分块 topk"
```

---

### Task 5: 文档同步

**Files:**
- Modify: `TPA/models/lightgcn/docs/USAGE.md`、`TPA/models/mf/docs/USAGE.md`、`TPA/models/wmf/docs/USAGE.md`
- Modify: `TPA/models/lightgcn/docs/IMPLEMENTATION_DOCS.md`

- [ ] **Step 1: 更新 lightgcn / mf / wmf 的 USAGE.md config 表**

三个 USAGE.md 的 training 配置表各加两行（默认值按各模型 yaml）：

```markdown
| num_workers | DataLoader 子进程数 | lightgcn/mf: 4；wmf: 0（单批全量） | 0~8 |
| persistent_workers | 跨 epoch 复用 worker（需 num_workers>0） | lightgcn/mf: true；wmf: false | true/false |
```

- [ ] **Step 2: 更新 IMPLEMENTATION_DOCS.md**

`TPA/models/lightgcn/docs/IMPLEMENTATION_DOCS.md` 中"邻接矩阵"小节：
- "格式: PyTorch sparse COO" 改为 "格式: PyTorch sparse CSR（构建时由 COO 转换，加速 GPU 稀疏乘法）"；
- 新增一行：训练优化——`train_step/eval_step` 内只传播一次并复用 `final_emb`，数学等价；`num_workers` / `persistent_workers` 可在 config.yaml 配置。

- [ ] **Step 3: 验证文档与实现一致**

Run（在 `G:\Idea\TPA`）：`rg -n "num_workers|persistent_workers" models/lightgcn/docs models/mf/docs models/wmf/docs`
Expected: 三个 USAGE.md 均含两键；IMPLEMENTATION_DOCS.md 含 CSR 说明。

- [ ] **Step 4: 提交**

```bash
git add TPA/models/lightgcn/docs/USAGE.md TPA/models/mf/docs/USAGE.md TPA/models/wmf/docs/USAGE.md TPA/models/lightgcn/docs/IMPLEMENTATION_DOCS.md
git commit -m "docs(models): 同步 num_workers/persistent_workers 配置与 CSR 存储说明"
```

---

### Task 6: 全量回归 + 基准验证

**Files:** 无新增。

- [ ] **Step 1: 全量单测**

Run（在 `G:\Idea\TPA`）：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v`
Expected: 全部通过（新增 4 个测试文件 + 既有 wmf/attack 测试）。

- [ ] **Step 2: 基准复测（不提交任何产物）**

在 `G:\Idea` 运行内联脚本：构建 gowalla LightGCN（batch=256），计时 5 个 `train_step` 并外推每 epoch 耗时；再用 ml100k 跑 1 epoch 冒烟训练（num_workers=0），确认 loss 正常下降、无异常。
Expected: 每 epoch 训练耗时 ~2 分钟量级（原 10.9 分钟）；冒烟训练正常。

- [ ] **Step 3: 收尾检查**

Run（在 `G:\Idea`）：`git status --short` 与 `git log --oneline -8`。
Expected: 仅本计划相关文件待提交（若有未提交改动则补齐提交）；提交记录包含 spec/plan/实现。
