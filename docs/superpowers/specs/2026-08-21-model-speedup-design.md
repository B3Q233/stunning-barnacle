# 模型训练与评估提速（方案 A）设计文档

> 日期：2026-08-21
> 状态：已获用户批准（含补充需求：`num_workers` / `persistent_workers` 对全部模型独立成 yaml 配置）
> 关联计划：`docs/superpowers/plans/2026-08-21-model-speedup.md`

## 1. 背景与问题

用户反馈模型运行速度很慢。实测定位（gowalla，RTX 3050 Laptop GPU，torch 2.5.1+cu121）：

| 环节 | 实测耗时 | 说明 |
|------|---------|------|
| 单个 `train_step` | 218 ms | 其中约 102 ms 为全图传播（`_compute_final_emb` 51 ms × 2 次） |
| 每 epoch 训练（batch=256） | 约 10.9 分钟 | yelp2018 数据更大，300 epochs 达数十小时 |
| 每 epoch 全量评估 `_topk_by_batch` | 11.4 秒 | Python 掩码循环 ~1.1 s + CPU topk ~10 s |
| 每 step 只传播一次后的 BPR 查询 | 0.08 ms | 传播本身是唯一大头 |

根因不是"未开多线程"，而是：① LightGCN 每个 `forward` 都重算整图传播且每 batch 调用两次；② 评估掩码用 Python 双重循环且 topk 在 CPU 上做；③ DataLoader 负采样在主进程串行（`num_workers=0`）。

## 2. 目标与不变量

- **目标**：在保持训练与评估结果逐位一致（数学等价）的前提下，将 LightGCN/MF 的训练与评估显著提速。
- **不变量**：
  - `compute_metrics` / `_topk_by_batch` / `rank_values` / `expected_percentile_rank` 的**默认行为与签名不变**（攻击链路 `evaluation/attack_eval.py`、WMF 训练与报告、既有测试均依赖）。
  - LightGCN `train_step` 的 loss 与梯度与改动前逐位一致。
  - `LightGCNDataLoader` / `MFDataLoader` / `WMFDataLoader` 在配置缺失时保持 `num_workers=0` 的旧行为。
- **明确不做**：epoch 级传播缓存（会改变训练语义且收益有限）；WMF ALS 用户循环改造（本期不动）。

## 3. 方案 A 细节

### 3.1 LightGCN 传播缓存 + CSR（`TPA/models/lightgcn/model.py`）

- `forward(users, items=None, final_emb=None)` 新增可选 `final_emb` 参数：传入时不再重算 `_compute_final_emb()`；不传时行为与旧版完全一致。
- `train_step` / `eval_step` 内只计算一次 `final_emb`，pos/neg 共用（数学等价；实测 step 218 ms → 108 ms）。
- `_build_adj` 生成 COO 后调用 `.to_sparse_csr()`，`self.A_hat` 改为 CSR 存储；`_compute_final_emb` 的 `torch.sparse.mm` 用法不变（实测再提速 ~3×，step 约 37 ms）。
- `self.A_hat` 属性名保留（现仅 `_compute_final_emb` 使用），避免破坏外部引用。

### 3.2 评估优化（`TPA/evaluation/metrics.py` + lightgcn/mf 的 callback）

- 新增 `build_train_mask_indices(train_user_items, user_ids) -> (rows, cols)`：把"训练集已交互物品掩码"编译成索引张量，**每轮训练只构建一次**，各 epoch 复用。
- `_topk_by_batch` / `compute_metrics` 新增可选关键字参数：
  - `mask_indices=None`：提供时用向量化 `scores[rows, cols] = -inf` 替代 Python 双重循环；
  - `topk_device=None`：提供时按 `chunk_size`（默认 1024）分块在该设备上做 `topk`，索引结果回 CPU；不提供时保持旧路径。
  - 所有新参数缺省时行为与旧版逐位一致。
- lightgcn/mf 的 `FullRankingCallback`：`__init__` 中预计算 `self.test_users` 与 `self.mask_indices`；评估时传入 `mask_indices` 与 `topk_device=model 设备`。
- 分数矩阵仍按现状分块计算并转 CPU（gowalla 全量分数 ~4.9 GB，不能整块放 GPU），仅在 topk 时逐块搬上 GPU。

### 3.3 DataLoader 多进程配置化（lightgcn / mf / wmf 的 dataset.py + 三个 config.yaml）

- 三个模型的 DataLoader 均从 `TrainingConfig` 读取：
  - `num_workers`（默认 0，保持旧行为）
  - `persistent_workers`（默认 False；仅当 `num_workers > 0` 时生效，避免 DataLoader 报错）
- 配置统一放在各模型 `config.yaml` 的 `training:` 段（现有 flatten 逻辑会把该段并入配置）：
  - lightgcn：`num_workers: 4`、`persistent_workers: true`
  - mf：`num_workers: 4`、`persistent_workers: true`
  - wmf：`num_workers: 0`、`persistent_workers: false`（单批全量矩阵，worker 无收益，显式声明）
- Windows 下 DataLoader 为进程池；worker 由 PyTorch 自动播种，负采样可复现性不受影响；`persistent_workers` 避免每 epoch 重复拷贝大字典。

## 4. 测试策略（stdlib unittest，CPU 可跑）

新增 `TPA/tests/test_lightgcn_cache.py`：
- CSR 传播结果与 COO 传播结果一致（小图，CPU）。
- `train_step`（缓存后）与"两次独立 forward"参考实现的 loss、梯度逐位一致。
- `forward` 传入/不传 `final_emb` 结果一致。

新增 `TPA/tests/test_evaluation_fast.py`：
- `build_train_mask_indices` 掩码效果与旧 Python 循环一致。
- `compute_metrics(mask_indices=...)` 与默认路径结果一致。
- `compute_metrics(topk_device='cpu')` 分块路径与默认路径结果一致。
- 三个模型 DataLoader 正确读取 yaml 中的 `num_workers` / `persistent_workers`；配置缺失时回退 0 / False。

## 5. 文档同步

- `models/lightgcn/docs/USAGE.md`：config 表新增 `num_workers` / `persistent_workers`。
- `models/lightgcn/docs/IMPLEMENTATION_DOCS.md`：邻接矩阵格式 COO → CSR、训练提速说明。
- `models/mf/docs/USAGE.md`、`models/wmf/docs/USAGE.md`：config 表新增上述两键。

## 6. 验收标准

- 全量回归：`G:\Idea\.venv\Scripts\python.exe -m unittest tests.test_* -v` 全部通过。
- 基准复测：gowalla 单 epoch 训练耗时从 ~10.9 分钟降到 ~2 分钟量级；评估单次从 ~11.4 s 降到 ~2 s 量级。
- 逐位一致性单测通过（见第 4 节）。
