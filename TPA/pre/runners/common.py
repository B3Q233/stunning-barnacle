# -*- coding: utf-8 -*-
"""pre 编排层公共设施：路径、配置、随机性锚定、meta 读写、模型/数据统一构造。

为什么需要这一层（设计动机）：
    pre 要同时对多个模型（LightGCN / MF）做同一套操作，而这些模型的 config、
    Dataset、train_step 契约各不相同。本模块把差异收敛到两个函数
    （build_model / build_loader），其余编排代码对模型无感。

关键约定（实验结果可比的前提）：
    1) **同初始化**：构造模型前先 set_global_seed(seed)，使不同条件的模型从
       完全相同的随机初始化出发。实测：同初始化重训的物品行平均余弦 0.956~0.975，
       异初始化只有 −0.002 —— 不锚定种子，跨条件的 embedding 比较没有意义。
    2) **确定性删除**：删除哪些交互由 (seed, item_id, ratio) 决定，并落盘缓存，
       避免"重跑一次就换了一批删除结果"。
    3) **不修改既有框架**：只 import models/*/ 的类与 attacks/*/generate.main()，
       不写入 TPA 下任何既有目录。

用法示例：
    from pre.runners.common import load_config, build_model, build_loader
    cfg = load_config(Path("pre/configs/default.yaml"))
    model = build_model("lightgcn", tcfg, num_users, num_items, edge_index)
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

TPA_ROOT = Path(__file__).resolve().parents[2]        # G:\Idea\TPA
REPO_ROOT = TPA_ROOT.parent                           # G:\Idea
if str(TPA_ROOT) not in sys.path:
    sys.path.insert(0, str(TPA_ROOT))

from models.registry import get_model_cls  # noqa: E402
from training.framework import TrainingConfig  # noqa: E402


# ── 路径 ──────────────────────────────────────────────────────────────────────
def pre_root() -> Path:
    return TPA_ROOT / "pre"


def load_config(path: Path) -> Dict[str, Any]:
    """读取 pre 配置（yaml）。返回原始 dict，不做 canonical 展开。"""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_root(cfg: Dict[str, Any]) -> Path:
    """产物根目录：<TPA>/<output.dir>（缺省 pre/outputs）。"""
    out = Path(str(cfg.get("output", {}).get("dir", "pre/outputs")))
    return out if out.is_absolute() else (TPA_ROOT / out)


def experiment_dir(cfg: Dict[str, Any], tag: str) -> Path:
    """单次实验目录：outputs/<tag>/，其下按 original/degraded/attacks/embeddings/results 分层。"""
    d = output_root(cfg) / tag
    for sub in ("original", "degraded", "attacks", "embeddings", "results"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


# ── 随机性锚定 ────────────────────────────────────────────────────────────────
def set_global_seed(seed: int) -> None:
    """把 torch/numpy/random 三个全局 RNG 都锚定到同一个种子。

    为什么三个都要：构造模型用 torch（nn.init），负采样用 random（Dataset 内部
    random.randint），数值线性代数（Procrustes）用 numpy。只锚一个会留下随机源。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: Any) -> int:
    """由若干标识稳定派生一个 32 位种子（用于退化/攻击等需要可复现的随机选择）。

    为什么不用 Python 的 hash()：它对 str 有进程级随机化，跨进程不稳定，会导致
    同一 (item, ratio) 在不同运行里删掉不同的交互。这里用 md5 保证跨进程稳定。
    """
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# ── meta 读写 ─────────────────────────────────────────────────────────────────
def load_meta(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_meta(meta: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(meta, f)
    return path


def save_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def item_interactions(pairs: Sequence[Tuple[int, int]], item: int
                      ) -> List[Tuple[int, int]]:
    """返回目标物品的全部交互对，顺序与 pairs 一致（用于确定性分层）。"""
    return [(u, i) for u, i in pairs if i == item]


def item_counts(pairs: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    """训练集每个物品的交互数。"""
    counts: Dict[int, int] = {}
    for _, i in pairs:
        counts[i] = counts.get(i, 0) + 1
    return counts


def user_items_of(pairs: Sequence[Tuple[int, int]]) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for u, i in pairs:
        out.setdefault(u, set()).add(i)
    return out


def normalize_user_items(meta: Dict[str, Any]) -> Dict[int, set]:
    """把 user_items 归一化成 {user_id: set(item)}。

    为什么需要（实测的接口不兼容）：仓库内部的 Dataset（LightGCNDataset /
    MFDataset）要求 user_items 是 dict（`self.users = list(user_items.keys())`），
    但 attacks/advinject 产出的中毒 meta 里 user_items 是**列表**（索引即 user id）。
    两者直接对接会抛 AttributeError: 'list' object has no attribute 'keys'。
    pre 作为纯编排层在这里做格式桥接：不改攻击、也不改模型代码。
    """
    ui = meta.get("user_items")
    if isinstance(ui, dict):
        return {int(u): set(v) for u, v in ui.items()}
    if ui is None:
        return user_items_of(meta["train_pairs"])
    return {int(u): set(v) for u, v in enumerate(ui) if v}


# ── 统一模型/数据构造 ─────────────────────────────────────────────────────────
def build_training_config(cfg: Dict[str, Any], model_name: str
                          ) -> TrainingConfig:
    """把 pre.training 与 model.overrides 合成为 TrainingConfig。

    键名与 models/*/config.yaml 对齐：emb_dim / n_layers / init_method 供 LightGCN，
    emb_dim 供 MF；lr / weight_decay / batch_size / neg_ratio / device 通用。
    """
    t = dict(cfg.get("pre", {}).get("training", {}))
    merged: Dict[str, Any] = {
        "device": t.get("device", "cuda"),
        "lr": t.get("lr", 1e-3),
        "weight_decay": t.get("weight_decay", 1e-4),
        "batch_size": t.get("batch_size", 256),
        "neg_ratio": t.get("neg_ratio", 1),
        "num_workers": t.get("num_workers", 0),
        "epochs": t.get("epochs", 30),
        # LightGCN 超参（MF 会忽略无关键）
        "emb_dim": 64,
        "n_layers": 3,
        "init_method": "normal",
    }
    # 覆盖顺序：通用训练键 → 模型适配器的结构性默认 → 用户显式 overrides
    # （例如 WMF 必须用 factors/alpha/optimizer=als，这些由适配器提供）
    from pre.runners.model_adapters import get_adapter
    merged.update(dict(get_adapter(model_name).defaults))
    merged.update(cfg.get("model", {}).get("overrides", {}) or {})
    return TrainingConfig(merged)


def build_model(model_name: str, tcfg: TrainingConfig, num_users: int,
                num_items: int, edge_index: torch.Tensor | None) -> Any:
    """按统一签名构造受害模型：model_cls(cfg, num_users, num_items, edge_index)。

    LightGCN 用 edge_index 构图（A_hat）；MF 忽略该参数。两者的构造签名一致，
    因此 pre 不需要为不同模型写分支。
    """
    cls = get_model_cls(model_name)
    return cls(tcfg, num_users, num_items, edge_index)


def build_train_loader(model_name: str, pairs: Sequence[Tuple[int, int]],
                       num_users: int, num_items: int, user_items: Dict[int, set],
                       neg_ratio: int, batch_size: int, seed: int,
                       num_workers: int = 0):
    """构造 pairwise 家族的训练 DataLoader（BPR 三元组）。

    为什么不用 model.build_dataloader(config)：那些 DataLoader 把 meta 路径硬编码在
    models/{model}/data/processed/{dataset}/meta.pkl，无法指向 pre 造出的退化数据。
    这里直接实例化模型的 Dataset 类（契约与官方一致：(users, pos, neg) 三元组），
    属于复用接口而非重写算法。
    """
    from pre.runners.model_adapters import get_adapter, load_dataset_cls
    cls = load_dataset_cls(get_adapter(model_name))
    ds = cls(list(pairs), num_items, user_items, num_users,
             mode="train", neg_ratio=neg_ratio)
    gen = torch.Generator()
    gen.manual_seed(seed)
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, generator=gen)


def build_full_batch(model_name: str, pairs: Sequence[Tuple[int, int]],
                     tcfg: TrainingConfig):
    """构造 full_batch 家族（如 WMF/ALS）的单个全量 batch。

    WMF 的 train_step 直接消费 (users, items, conf, p, user_obs, item_obs)，
    user_obs/item_obs 是预构建的观测分组（每个 epoch 复用，不重扫）。
    """
    from pre.runners.model_adapters import get_adapter, load_dataset_cls
    cls = load_dataset_cls(get_adapter(model_name))
    ds = cls(list(pairs), alpha=float(tcfg.get("alpha", 40.0)),
             epsilon=float(tcfg.get("epsilon", 1e-8)),
             scheme=str(tcfg.get("confidence_scheme", "minimal")))
    return (ds.users, ds.items, ds.conf, ds.p, ds.user_obs, ds.item_obs)


def train_victim(model_name: str, cfg: Dict[str, Any], meta: Dict[str, Any],
                 seed: int, device: str | None = None) -> Tuple[Any, List[Dict[str, float]]]:
    """在给定 meta 上从头训练受害模型（同初始化、固定轮数）。

    返回 (model, history)。history 每轮记录 loss，便于后续检查是否收敛。
    """
    tcfg = build_training_config(cfg, model_name)
    if device:
        tcfg.update({"device": device})
    from pre.runners.model_adapters import get_adapter
    adapter = get_adapter(model_name)
    num_users, num_items = meta["num_users"], meta["num_items"]
    pairs = meta["train_pairs"]
    user_items = normalize_user_items(meta)
    edge_index = torch.LongTensor([[u, i] for u, i in pairs]).T

    # 同初始化：必须在构造模型之前锚定种子（nn.init 从全局 RNG 取值）
    set_global_seed(seed)
    model = build_model(model_name, tcfg, num_users, num_items, edge_index)

    epochs = int(cfg.get("pre", {}).get("training", {}).get("epochs", 30))
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.set_train()
        total, n = 0.0, 0
        if adapter.loop == "pairwise":
            # 每个 epoch 重建 loader：重新洗牌（DataLoader 的 generator 固定种子，
            # 因此整体仍可复现）
            loader = build_train_loader(
                model_name, pairs, num_users, num_items, user_items,
                int(tcfg.get("neg_ratio", 1)), int(tcfg.get("batch_size", 256)),
                seed + ep, int(tcfg.get("num_workers", 0)))
            for batch in loader:
                m = model.train_step(batch)
                total += float(m.get("loss", 0.0))
                n += 1
        else:                                   # full_batch：每 epoch 一次全量 sweep
            batch = build_full_batch(model_name, pairs, tcfg)
            m = model.train_step(batch)
            total += float(m.get("loss", 0.0))
            n += 1
        history.append({"epoch": ep, "loss": total / max(1, n)})
    model.set_eval()
    history.append({"epoch_seconds_total": round(time.time() - t0, 1)})
    return model, history


def item_embedding(model: Any, item: int) -> torch.Tensor:
    """取目标物品的最终嵌入（float32 CPU）。"""
    return model.get_item_embeddings().detach().cpu().float()[item]


def item_matrix(model: Any) -> torch.Tensor:
    """取全量物品嵌入矩阵（Procrustes 拟合用）。"""
    return model.get_item_embeddings().detach().cpu().float()


def git_commit() -> str:
    """当前仓库 commit（记录用；失败返回 unknown，不影响实验）。"""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
