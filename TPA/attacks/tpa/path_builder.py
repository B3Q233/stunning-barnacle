"""TPA —— 路径构造模块（Step 2：传递路径构造）

已冻结的实现决策（与用户确认）：
- 不做 PGD：当前为纯图路径注入版，复合损失对抗优化留待下一阶段
- 路径策略：最短路径法（Dijkstra，边权 = CF 余弦距离）
- 距离度量：λ=0 纯 CF 距离（干净模型物品嵌入的余弦距离）；
  语义融合 d'(i,j) = λ·d_sem + (1-λ)·d_CF 留待有多模态物品特征的阶段
- 数据集：当前 ml100k（与 bandwagon / random 基线同口径）

功能：
1. build_cooccurrence_graph：从训练交互构建物品共现图（稀疏，边 = 共现次数）
2. weighted_adj：给共现边赋 CF 余弦距离权重（路径只允许走共现边）
3. shortest_path：带跳数上限的加权最短路径（state = (node, hops)）
4. build_tpa_profiles：平庸基座 + 传递路径 + 目标 的假用户画像

用法:
  python attacks/tpa/path_builder.py --config attacks/tpa/config.yaml
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.tpa.generate import (
    compute_item_popularity,
    load_rec_freq_cache,
    load_meta,
    load_yaml_config,
    raw_meta_path,
    select_target_items,
)
from attacks.tpa.classify import resolve_active_checkpoint
from attacks.tpa.fit import build_training_config
from attacks.tpa.registry import active_model_name, get_model_cls


def paths_dir(config: Dict[str, Any]) -> Path:
    dataset = config["dataset"]
    attack_name = config["attack"]["name"]
    return PROJECT_ROOT / "attacks" / attack_name / "data" / "paths" / dataset


def paths_cache_path(config: Dict[str, Any]) -> Path:
    tag = "_proxy" if config.get("surrogate", {}).get("enabled", False) else ""
    return paths_dir(config) / f"profiles{tag}.json"


def load_paths_cache(config: Dict[str, Any],
                     required: bool = False) -> Dict[str, Any] | None:
    """读取路径画像缓存（path_builder.py 产出）。"""
    path = paths_cache_path(config)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"路径画像缓存不存在: {path}\n"
                f"请先运行: python attacks/tpa/run.py --mode paths"
            )
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_cooccurrence_graph(train_pairs: List[Tuple[int, int]],
                             num_items: int) -> sp.csr_matrix:
    """物品共现图：同一用户交互集合内两两连边，边值 = 共现次数。

    返回对称稀疏邻接矩阵 (num_items, num_items)。路径只允许沿共现边前进，
    保证画像中相邻物品在真实用户行为中确实一起出现过。
    """
    user_items: Dict[int, set] = defaultdict(set)
    for _u, i in train_pairs:
        user_items[_u].add(i)

    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    for items in user_items.values():
        items = sorted(items)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                rows.append(items[a])
                cols.append(items[b])
                data.append(1)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_items, num_items))
    adj = adj + adj.T  # 对称化
    adj.sum_duplicates()
    return adj.tocsr()


def weighted_adj(adj: sp.csr_matrix, item_emb_norm: np.ndarray) -> sp.csr_matrix:
    """把共现边赋权：w(i,j) = 1 - cos(e_i, e_j)（CF 余弦距离，λ=0）。"""
    rows = np.repeat(np.arange(adj.shape[0]), np.diff(adj.indptr))
    cols = adj.indices
    cos = np.sum(item_emb_norm[rows] * item_emb_norm[cols], axis=1)
    weights = 1.0 - np.clip(cos, -1.0, 1.0)
    return sp.csr_matrix((weights, adj.indices, adj.indptr), shape=adj.shape)


def cf_distance(i: int, j: int, item_emb_norm: np.ndarray) -> float:
    return float(1.0 - np.dot(item_emb_norm[i], item_emb_norm[j]))


def shortest_path(start: int, target: int, adj_weighted: sp.csr_matrix,
                  max_hops: int,
                  per_hop_tau: Optional[float] = None) -> Optional[List[int]]:
    """带跳数上限的加权最短路径（Dijkstra，state=(node, hops)）。

    返回 [start, ..., target]（含端点），找不到满足上限的路径返回 None。
    per_hop_tau：可选每跳距离阈值（当前 CF 阶段默认 None，语义融合阶段启用）。
    """
    if start == target:
        return [start]
    if max_hops <= 0:
        return None

    best: Dict[Tuple[int, int], float] = {}
    heap: List[Tuple[float, int, int, Tuple[int, ...]]] = [
        (0.0, 0, start, (start,))
    ]
    while heap:
        cost, hops, node, path = heapq.heappop(heap)
        key = (node, hops)
        if best.get(key, math.inf) < cost:
            continue
        best[key] = cost
        if node == target:
            return list(path)
        if hops >= max_hops:
            continue

        row_start, row_end = adj_weighted.indptr[node], adj_weighted.indptr[node + 1]
        for k in range(row_start, row_end):
            nxt = int(adj_weighted.indices[k])
            w = float(adj_weighted.data[k])
            if per_hop_tau is not None and w > per_hop_tau:
                continue
            ncost = cost + w
            nkey = (nxt, hops + 1)
            if ncost < best.get(nkey, math.inf):
                heapq.heappush(heap, (ncost, hops + 1, nxt, path + (nxt,)))
    return None


def compute_shortest_paths_to(target: int, adj_weighted: sp.csr_matrix,
                              max_hops: int,
                              per_hop_tau: Optional[float] = None
                              ) -> Tuple[Dict[Tuple[int, int], float],
                                         Dict[Tuple[int, int], Tuple[int, int]]]:
    """从 target 出发的反向 Dijkstra（无向图 ⇒ 等价于"到 target"的最短路径）。

    稠密共现图上每个起点各跑一次 Dijkstra 代价过高；本函数对每个目标只跑一次，
    所有起点的最短路径都从 best/pred 表中重建。返回:
    - best[(node, hops)] = 到 target 的最小总距离
    - pred[(node, hops)] = (前驱节点, 前驱跳数)
    """
    best: Dict[Tuple[int, int], float] = {}
    pred: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heap: List[Tuple[float, int, int]] = [(0.0, 0, target)]
    while heap:
        cost, hops, node = heapq.heappop(heap)
        key = (node, hops)
        if best.get(key, math.inf) < cost:
            continue
        best[key] = cost
        if hops >= max_hops:
            continue
        row_start, row_end = adj_weighted.indptr[node], adj_weighted.indptr[node + 1]
        for k in range(row_start, row_end):
            nxt = int(adj_weighted.indices[k])
            w = float(adj_weighted.data[k])
            if per_hop_tau is not None and w > per_hop_tau:
                continue
            ncost = cost + w
            nkey = (nxt, hops + 1)
            if ncost < best.get(nkey, math.inf):
                best[nkey] = ncost
                pred[nkey] = (node, hops)
                heapq.heappush(heap, (ncost, hops + 1, nxt))
    return best, pred


def reconstruct_path(start: int, target: int,
                     best: Dict[Tuple[int, int], float],
                     pred: Dict[Tuple[int, int], Tuple[int, int]],
                     max_hops: int) -> Optional[List[int]]:
    """从 best/pred 表中重建 start → target 的最短路径，不可达返回 None。"""
    if start == target:
        return [start]
    best_key: Optional[Tuple[int, int]] = None
    for h in range(max_hops + 1):
        key = (start, h)
        if key in best and (best_key is None or best[key] < best[best_key]):
            best_key = key
    if best_key is None:
        return None

    path_rev: List[int] = [start]
    cur = best_key
    while cur[0] != target:
        prev = pred.get(cur)
        if prev is None:
            return None
        path_rev.append(prev[0])
        cur = prev
    return path_rev  # [start, ..., target]


def choose_path_start(base_items: List[int], target: int,
                      best: Dict[Tuple[int, int], float],
                      pred: Dict[Tuple[int, int], Tuple[int, int]],
                      item_emb_norm: np.ndarray,
                      max_hops: int
                      ) -> Optional[Tuple[float, int, List[int]]]:
    """从基座中选一个到目标存在可行路径、且 CF 距离最近的物品作为路径起点。"""
    candidates: List[Tuple[float, int, List[int]]] = []
    for b in base_items:
        if b == target:
            continue
        path = reconstruct_path(b, target, best, pred, max_hops)
        if path is not None:
            candidates.append((cf_distance(b, target, item_emb_norm), b, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def build_tpa_profiles(num_fake_users: int, targets: List[int],
                       base_pool: List[int], base_size: int,
                       adj_weighted: sp.csr_matrix, item_emb_norm: np.ndarray,
                       max_hops: int, rng: random.Random,
                       per_hop_tau: Optional[float] = None
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """构造 TPA 假用户画像：平庸基座 + 传递路径 + 目标商品。

    每个画像:
      items = base_size 个基座物品（高流行度采样）
              + 路径中间桥接物品（最短路径，≤ max_hops 跳）
              + 目标物品
    无可行路径时回退为"基座 + 目标"（fallback=direct），并计入统计。
    返回 (profiles, path_stats)。
    """
    if num_fake_users <= 0 or not targets:
        raise ValueError("假用户数和目标物品数必须为正")
    if base_size <= 0:
        raise ValueError("base_size 必须为正")
    if len(base_pool) < base_size:
        raise ValueError(
            f"基座物品池仅 {len(base_pool)} 个，小于 base_size={base_size}"
        )

    per_target = num_fake_users // len(targets)
    remainder = num_fake_users % len(targets)

    profiles: List[Dict[str, Any]] = []
    path_stats = {
        "total": 0,
        "paths_found": 0,
        "fallback_direct": 0,
        "hops": [],
        "cf_dist_start_target": [],
    }
    uid = 0
    for t_idx, target in enumerate(targets):
        # 每个目标只跑一次反向 Dijkstra，所有基座起点的路径从此表重建
        best, pred = compute_shortest_paths_to(
            target, adj_weighted, max_hops, per_hop_tau
        )
        n = per_target + (1 if t_idx < remainder else 0)
        for _ in range(n):
            base_pool_excl = [i for i in base_pool if i != target]
            base = rng.sample(base_pool_excl, min(base_size, len(base_pool_excl)))
            found = choose_path_start(
                base, target, best, pred, item_emb_norm, max_hops
            )
            if found is None:
                items = base + [target]
                path_info: Optional[Dict[str, Any]] = None
                path_stats["fallback_direct"] += 1
            else:
                _dist, start, path = found
                middle = path[1:-1]
                items = base + middle + [target]
                path_info = {
                    "start": start,
                    "middle": middle,
                    "hops": len(path) - 1,
                    "cf_dist_start_target": round(cf_distance(start, target, item_emb_norm), 6),
                }
                path_stats["paths_found"] += 1
                path_stats["hops"].append(len(path) - 1)
                path_stats["cf_dist_start_target"].append(path_info["cf_dist_start_target"])

            # 去重（基座/路径/目标可能重叠），保持目标在最后；顺序不影响隐式反馈
            items = list(dict.fromkeys(items + [target]))
            rng.shuffle(items)
            profiles.append({
                "fake_user": uid,
                "target": target,
                "items": items,
                "path": path_info,
            })
            uid += 1
            path_stats["total"] += 1

    path_stats["avg_hops"] = (
        round(sum(path_stats["hops"]) / len(path_stats["hops"]), 4)
        if path_stats["hops"] else None
    )
    path_stats["avg_cf_dist_start_target"] = (
        round(sum(path_stats["cf_dist_start_target"])
              / len(path_stats["cf_dist_start_target"]), 6)
        if path_stats["cf_dist_start_target"] else None
    )
    return profiles, path_stats


def main(config: Dict[str, Any]) -> Dict[str, Any]:
    """执行 TPA 路径构造，产出路径画像缓存（data/paths/{dataset}/profiles.json）。"""
    dataset = config["dataset"]
    attack_cfg = config["attack"]
    path_cfg = attack_cfg.get("path", {})
    seed = config.get("seed", 42)
    rng = random.Random(seed)
    model_name = active_model_name(config)
    attack_name = config["attack"]["name"]
    base_size = attack_cfg.get("base_size", 10)
    max_bridge = path_cfg.get("max_bridge_items", 3)
    per_hop_tau = path_cfg.get("per_hop_tau")
    max_hops = max_bridge + 1  # 中间桥接物品数上限 → 跳数上限
    mode = "代理(surrogate)" if config.get("surrogate", {}).get("enabled", False) else "白盒(victim)"

    meta = load_meta(raw_meta_path(config))
    num_users, num_items = meta["num_users"], meta["num_items"]
    popularity = compute_item_popularity(meta["train_pairs"])
    k = config.get("training", {}).get("k") or 20
    rec_cache = load_rec_freq_cache(config, model_name, k, required=False)

    # 目标选择（与 generate 阶段同口径）
    ti_cfg = attack_cfg["target_items"]
    targets = select_target_items(
        popularity, num_items,
        ti_cfg.get("strategy", "specified"),
        ti_cfg.get("count", 3),
        ti_cfg.get("ids", []),
        rng,
        categories=rec_cache["categories"] if rec_cache else None,
        category=ti_cfg.get("category", "cold"),
        rec_counts=rec_cache["counts"] if rec_cache else None,
    )

    # 假用户数量：显式优先，其次按 ratio
    num_fake = attack_cfg.get("num_fake_users")
    if num_fake is None:
        num_fake = max(1, int(num_users * attack_cfg.get("ratio", 0.01)))

    # 加载干净模型 → 物品嵌入（CF 距离）
    cfg = build_training_config(config, dataset, model_name)
    model_cls = get_model_cls(model_name)
    edge_index = torch.LongTensor([[u, i] for u, i in meta["train_pairs"]]).T
    model = model_cls(cfg, num_users, num_items, edge_index)
    ckpt_path = resolve_active_checkpoint(config)
    ckpt = torch.load(ckpt_path, map_location=model._device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[paths] 模型={model_name}（{mode}），checkpoint={ckpt_path}，"
          f"用户={num_users}，物品={num_items}")

    item_emb = model.get_item_embeddings().detach().cpu().numpy()
    item_emb_norm = item_emb / np.linalg.norm(item_emb, axis=1, keepdims=True)

    # 共现图 + CF 权重
    adj = build_cooccurrence_graph(meta["train_pairs"], num_items)
    adj_w = weighted_adj(adj, item_emb_norm)
    print(f"[paths] 共现图: {num_items} 物品, {adj.nnz // 2} 条无向边")

    # 平庸基座池：高流行度物品（P(i) 采样，语义接近 c 的约束留待语义阶段）
    base_pool = [i for i, _ in popularity.most_common(base_size * 5)]
    print(f"[paths] 基座池: Top-{len(base_pool)} 流行物品（base_size={base_size}）")

    profiles, path_stats = build_tpa_profiles(
        num_fake, targets, base_pool, base_size, adj_w, item_emb_norm,
        max_hops, rng, per_hop_tau,
    )
    print(f"[paths] 画像: {len(profiles)} 个，找到路径 {path_stats['paths_found']}，"
          f"回退直连 {path_stats['fallback_direct']}，"
          f"平均跳数 {path_stats['avg_hops']}，"
          f"起终点平均 CF 距离 {path_stats['avg_cf_dist_start_target']}")

    payload = {
        "dataset": dataset,
        "attack": attack_name,
        "model": model_name,
        "surrogate_enabled": config.get("surrogate", {}).get("enabled", False),
        "seed": seed,
        "base_size": base_size,
        "max_bridge_items": max_bridge,
        "per_hop_tau": per_hop_tau,
        "checkpoint": str(ckpt_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "targets": targets,
        "profiles": profiles,
        "path_stats": path_stats,
    }
    out = paths_cache_path(config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[paths] 缓存 → {out}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPA 路径构造")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "tpa" / "config.yaml"),
    )
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
