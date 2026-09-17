"""UBA —— estimate 阶段：用代理模型模拟实验估计处理效应 Y（w/ S_φ 支路）。

论文 §4.1 的第二种估计（有代理模型 S_φ）：

    对每个候选档位 t_u ∈ {1..H}：
        给**所有**目标用户等量分配 t_u 个假用户 → 由后端攻击者生成 D_f
        → 在 D_r ∪ D_f 上重新训练代理模型 S_φ
        → 检查目标物品 i 是否进入目标用户 u 的 Top-K
    重复 E 次（每轮换随机种子）取命中频率，作为 Y_{u,i}(D_f(t_u)) 的估计。

本文件是唯一需要 import 模型代码的处理效应支路（path 支路在 uplift.py 里纯 numpy），
因此单独作为 `estimate` 阶段存在，产物缓存到
`attacks/uba/data/estimate/{dataset}/{model}/item{i}_h{H}_surrogate_E{E}_...json`，
data 阶段只读缓存（缺失时才会回退调用本模块）。

成本提示：训练次数 = (H+1) × E。论文默认 H=6、E=10 → 70 次代理重训，
与论文 Appendix B.2 报告的"一次估计约 52–64 分钟"量级一致。
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.uba.generate import (  # noqa: E402
    build_fake_profiles,
    effect_cache_path,
    inject,
    load_meta,
    load_yaml_config,
    popular_pool,
    profile_cfg,
    raw_meta_path,
    resolve_target_item,
    save_json,
    target_users_cfg,
    treatment_cfg,
)
from attacks.uba.registry import get_dataset_cls, get_model_cls, load_model_config  # noqa: E402
from attacks.uba.uplift import interaction_sets, select_target_users  # noqa: E402
from training.framework import TrainingConfig  # noqa: E402
from training.timing import timed  # noqa: E402


def build_surrogate_config(config: Dict[str, Any], dataset: str) -> TrainingConfig:
    """构造代理模型训练配置：代理自身 config.yaml 默认值 ← surrogate.training 覆盖。

    为什么不用攻击的 training 段：论文里 S_φ 与受害模型是两套超参（Appendix B.2 说明
    各自沿用原论文设置），因此 surrogate 有独立的 training 子块。
    """
    sur = config.get("surrogate", {})
    name = str(sur.get("name", "mf"))
    model_cfg = load_model_config(name)
    model_defaults = model_cfg.get("model", {})
    defaults = model_cfg.get("training", {})
    tr = sur.get("training", {}) or {}
    overrides: Dict[str, Any] = {
        "dataset": dataset,
        "emb_dim": model_defaults.get("emb_dim", 64),
        "n_layers": model_defaults.get("n_layers", 3),
        "init_method": model_defaults.get("init_method", "normal"),
        "factors": model_defaults.get("factors", 100),
        "lr": tr.get("lr", defaults.get("lr", 0.001)),
        "epochs": tr.get("epochs", defaults.get("epochs", 30)),
        "batch_size": tr.get("batch_size", defaults.get("batch_size", 256)),
        "weight_decay": tr.get("weight_decay", defaults.get("weight_decay", 1e-4)),
        "neg_ratio": defaults.get("neg_ratio", 1),
        "device": str(config.get("training", {}).get("device",
                                                      defaults.get("device", "cuda"))),
        "k": int(config.get("k", 10)),
        "eval_every": tr.get("eval_every", 5),
        "num_workers": 0,           # 模拟实验里反复建 DataLoader，子进程开销得不偿失
        "persistent_workers": False,
    }
    return TrainingConfig(overrides=overrides)


def _split_train_val(pairs: Sequence[Tuple[int, int]], seed: int
                     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """95/5 随机划分（与受害模型同一划分协议、不同种子），保证代理独立训练。"""
    shuffled = [tuple(int(x) for x in p) for p in pairs]
    random.Random(seed).shuffle(shuffled)
    split = int(len(shuffled) * 0.95)
    return shuffled[:split], shuffled[split:]


def train_surrogate(config: Dict[str, Any], poisoned_meta: Dict[str, Any],
                    seed: int) -> Any:
    """在（可能已中毒的）数据上从零重训一个代理模型 S_φ。

    为什么每次重训：论文的处理效应定义就是"注入 D_f 之后模型重新训练后的响应"；
    用同一个已训练模型会漏掉"假数据改变训练轨迹"这一效应。
    """
    sur = config.get("surrogate", {})
    name = str(sur.get("name", "mf"))
    if not sur.get("enabled", False):
        raise ValueError(
            "treatment.method=surrogate 需要 surrogate.enabled=true 并配置代理模型"
            "（详见 attacks/uba/docs/USAGE.md）"
        )
    model_cls = get_model_cls(name)
    dataset_cls = get_dataset_cls(name)
    if not hasattr(model_cls, "train_step"):
        raise ValueError(
            f"代理模型 {name} 不支持 mini-batch 训练（无 train_step），"
            f"请改用 mf / lightgcn / ncf 等实现"
        )

    cfg = build_surrogate_config(config, str(config["dataset"]))
    num_users = int(poisoned_meta["num_users"])
    num_items = int(poisoned_meta["num_items"])
    user_items = poisoned_meta["user_items"]
    neg_ratio = int(cfg.get("neg_ratio", 1))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    edge_index = torch.LongTensor(
        [[int(u), int(i)] for u, i in poisoned_meta["train_pairs"]]).T
    model = model_cls(cfg, num_users, num_items, edge_index)

    ckpt = sur.get("checkpoint")
    if ckpt:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=model._device,
                               weights_only=True)
            state_dict = state.get("model_state_dict", state)
            try:
                model.load_state_dict(state_dict)
                print(f"[estimate] 代理模型热启动 ← {ckpt_path}")
            except RuntimeError as exc:  # 形状不匹配（用户数变化）时退回随机初始化
                print(f"[estimate] [!] 代理 checkpoint 形状不匹配，改为随机初始化: {exc}")

    train_pairs, val_pairs = _split_train_val(poisoned_meta["train_pairs"], seed)
    if not val_pairs:                       # 极小数据集兜底（合成测试）
        val_pairs = train_pairs
    train_ds = dataset_cls(train_pairs, num_items, user_items, num_users,
                           mode="train", neg_ratio=neg_ratio)
    val_ds = dataset_cls(val_pairs, num_items, user_items, num_users,
                         mode="train", neg_ratio=neg_ratio)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0)

    for _epoch in range(1, int(cfg.epochs) + 1):
        model.set_train()
        for batch in train_loader:
            model.train_step(batch)
    model.set_eval()
    return model


def target_user_hits(model: Any, clean_user_items: Dict[int, set],
                     target_users: Sequence[int], target_item: int,
                     hit_k: int, batch_size: int = 512) -> np.ndarray:
    """目标用户是否在 Top-hit_k 命中目标物品（0/1 数组，行序 = target_users）。

    矩阵变换：
        user_emb[U_t]  (|U_t|, d) @ item_emb.T (d, N) → scores (|U_t|, N)
        scores[已交互物品] = -inf                      → 过滤干净训练集已见物品
        topk(scores, hit_k)                            → (|U_t|, hit_k)
        hit = (topk == i).any(dim=1)                   → (|U_t|,) bool
    过滤只用**干净**训练集（clean_user_items）：与仓库评估协议一致，避免把假用户
    的交互算进用户的已见集合。
    """
    users = [int(u) for u in target_users]
    device = model._device
    with torch.no_grad():
        item_emb = model.get_item_embeddings()
        user_emb = model.get_user_embeddings()
        hits = np.zeros(len(users), dtype=np.float64)
        for start in range(0, len(users), batch_size):
            chunk = users[start:start + batch_size]
            ids = torch.LongTensor(chunk).to(device)
            scores = user_emb[ids] @ item_emb.T
            for r, u in enumerate(chunk):
                seen = clean_user_items.get(u, set())
                if seen:
                    seen_idx = torch.LongTensor(sorted(int(i) for i in seen)).to(device)
                    scores[r, seen_idx] = float("-inf")
            k = min(int(hit_k), int(scores.shape[1]))
            topk = torch.topk(scores, k, dim=1).indices.cpu()
            hits[start:start + len(chunk)] = (
                (topk == int(target_item)).any(dim=1).numpy().astype(np.float64))
    return hits


@timed("代理模型处理效应估计")
def treatment_effect_surrogate(config: Dict[str, Any], meta: Dict[str, Any],
                               target_item: int,
                               target_users: Sequence[int]) -> Dict[str, Any]:
    """跑完整模拟实验，返回处理效应矩阵 Y（( |U_t|, H+1)）。"""
    treatment = treatment_cfg(config)
    sur = config.get("surrogate", {})
    seed = int(config.get("seed", 42))
    max_per_user = int(treatment.get("max_per_user", 6))
    repeats = int(treatment.get("repeats", 10))
    hit_k = int(treatment.get("hit_k", 20))
    alpha = float(treatment.get("alpha", 1.0))
    beta = float(treatment.get("beta", 1.0))
    if repeats <= 0:
        raise ValueError(f"treatment.repeats 必须为正，实际 {repeats}")

    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    filler_size = int(config["attack"].get("filler_size", 36))
    filler_source = str(profile_cfg(config).get("filler_source", "template_user"))
    clean_user_items = interaction_sets(meta["train_pairs"], max_uid=num_users)
    popular = popular_pool(config, str(config.get("model", {}).get("name", "lightgcn")),
                           meta)
    users = [int(u) for u in target_users]

    effect = np.zeros((len(users), max_per_user + 1), dtype=np.float64)
    per_repeat: List[List[List[int]]] = []
    t_start = time.perf_counter()
    for t in range(max_per_user + 1):
        repeat_hits: List[np.ndarray] = []
        for e in range(repeats):
            # 每轮换种子：filler 采样与模型初始化都不同（论文：重复 E 次不同随机种子）
            round_seed = seed * 100003 + t * 1009 + e
            rng = random.Random(round_seed)
            templates = [u for u in users for _ in range(t)]
            profiles = build_fake_profiles(
                templates, int(target_item), filler_size, filler_source,
                clean_user_items, num_items, popular, rng)
            poisoned = inject(meta, profiles)
            model = train_surrogate(config, poisoned, round_seed)
            hits = target_user_hits(model, clean_user_items, users,
                                    int(target_item), hit_k)
            repeat_hits.append(hits)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        stacked = np.vstack(repeat_hits)
        effect[:, t] = alpha * np.power(stacked.mean(axis=0), beta)
        per_repeat.append([[int(x) for x in row] for row in stacked])
        elapsed = time.perf_counter() - t_start
        print(f"[estimate] t={t}/{max_per_user}：{repeats} 轮平均命中率 "
              f"{stacked.mean():.4f}（累计 {elapsed / 60:.1f} 分钟）")

    return {
        "method": "surrogate",
        "target_item": int(target_item),
        "target_users": users,
        "max_per_user": max_per_user,
        "repeats": repeats,
        "hit_k": hit_k,
        "alpha": alpha,
        "beta": beta,
        "surrogate": {
            "name": str(sur.get("name", "mf")),
            "epochs": int((sur.get("training", {}) or {}).get("epochs", 30)),
            "lr": float((sur.get("training", {}) or {}).get("lr", 0.001)),
            "checkpoint": sur.get("checkpoint"),
        },
        "effect": effect,
        "per_repeat_hits": per_repeat,
        "seconds": time.perf_counter() - t_start,
    }


def main(config: Dict[str, Any],
         meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """estimate 阶段入口：计算 Y 并写共享缓存（data 阶段直接读该缓存）。

    meta 参数存在的原因：run.py 走默认路径（读模型 processed 目录），
    单元测试/批量调度可以注入已加载的 meta，避免重复 IO。
    """
    model_name = str(config.get("model", {}).get("name", "lightgcn"))
    if meta is None:
        meta = load_meta(raw_meta_path(config, model_name))
    rng = random.Random(int(config.get("seed", 42)))
    target_item = int(resolve_target_item(config, meta, model_name, rng)["item_id"])
    tu_info = select_target_users(meta, target_item, target_users_cfg(config),
                                  int(config.get("seed", 42)))
    target_users = [int(u) for u in tu_info["users"]]
    payload = treatment_effect_surrogate(config, meta, target_item, target_users)
    save_json({**payload, "effect": np.asarray(payload["effect"]).tolist()},
              effect_cache_path(
                  config, model_name, target_item, "surrogate",
                  repeats=int(treatment_cfg(config).get("repeats", 10)),
                  hit_k=int(treatment_cfg(config).get("hit_k", 20)),
                  alpha=float(treatment_cfg(config).get("alpha", 1.0)),
                  beta=float(treatment_cfg(config).get("beta", 1.0)),
                  seed=int(config.get("seed", 42))))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UBA 代理模型处理效应估计")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "uba" / "config.yaml"))
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
