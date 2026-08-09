"""Bandwagon 攻击 —— 第 1 步：模型推荐频次分类

攻击流程的第一步（与用户确认的口径一致）：
1. 加载干净模型 checkpoint（模型由 config ``model.name`` 指定）
2. 计算全量评分矩阵（按批），对每个用户取 Top-K 推荐物品
   （过滤掉该用户训练集已交互物品，与评估协议一致）
3. 统计每个物品在 Top-K 中出现的次数
4. 按频次划分三档：
   - 流行物品: 出现次数最高的前 20%（热门 filler 池）
   - 普通物品: 有出现次数但不在前 20%
   - 冷门物品: Top-K 中出现次数为 0（通常也是攻击目标的首选）
5. 缓存到 attacks/bandwagon/data/rec_freq/{dataset}/{model}_top{k}.json，
   供 generate.py 选择目标 / 采样 filler 使用

用法:
  python attacks/bandwagon/classify.py --config attacks/bandwagon/config.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Counter as CounterType, Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.bandwagon.registry import get_model_cls
from attacks.bandwagon.generate import (
    DEFAULT_RAW_META as RAW_META_PATH,
    load_meta,
    load_yaml_config,
)
from attacks.bandwagon.fit import build_training_config


def rec_freq_dir(config: Dict[str, Any]) -> Path:
    dataset = config["dataset"]
    return PROJECT_ROOT / "attacks" / "bandwagon" / "data" / "rec_freq" / dataset


def rec_freq_path(config: Dict[str, Any], model_name: str, k: int) -> Path:
    return rec_freq_dir(config) / f"{model_name}_top{k}.json"


def resolve_clean_checkpoint(config: Dict[str, Any]) -> Path:
    """打分用的干净模型 checkpoint：classification.checkpoint > clean_checkpoint
    > warm_start.checkpoint。"""
    cls_cfg = config.get("classification", {})
    candidates = [
        cls_cfg.get("checkpoint"),
        config.get("clean_checkpoint"),
        config.get("warm_start", {}).get("checkpoint"),
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return p
    raise FileNotFoundError(
        "未找到可用于打分的干净模型 checkpoint。请在配置中设置 "
        "classification.checkpoint / clean_checkpoint / warm_start.checkpoint，"
        "并先训练干净模型。"
    )


def compute_topk_counts(model, meta: Dict[str, Any], k: int,
                        batch_size: int = 1024) -> CounterType[int]:
    """全量评分 → 每个用户 Top-K → 物品出现次数统计。

    不落盘完整评分矩阵（大数据集 M×N 可能数十 GB），按批计算后只保留
    Top-K 索引并立即统计频次，与"先得评分矩阵再取前 K"等价。
    """
    num_users, num_items = meta["num_users"], meta["num_items"]
    user_items = meta["user_items"]

    model.set_eval()
    user_emb = model.get_user_embeddings()
    item_emb = model.get_item_embeddings()
    counts: CounterType[int] = Counter()

    with torch.no_grad():
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            ids = torch.arange(start, end, device=user_emb.device)
            scores = user_emb[ids] @ item_emb.T  # (b, num_items)

            # 过滤训练集已交互物品（-inf），与 all-ranking 评估协议一致
            for r, uid in enumerate(range(start, end)):
                for i in user_items.get(uid, ()):
                    scores[r, i] = float("-inf")

            topk = torch.topk(scores, k, dim=1).indices.cpu()  # (b, k)
            counts.update(topk.reshape(-1).tolist())
    return counts


def classify_counts(counts: CounterType[int], num_items: int,
                    popular_ratio: float = 0.2) -> Tuple[Dict[str, List[int]], Dict[str, Any]]:
    """按推荐频次划分 流行 / 普通 / 冷门 三档。

    - 流行: 出现次数最高的前 popular_ratio（出现物品不足时向上取整至少 1 个）
    - 普通: 有出现次数但未进前 20%
    - 冷门: 出现次数为 0
    """
    appearing = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    n_appearing = len(appearing)
    n_popular = math.ceil(n_appearing * popular_ratio) if n_appearing else 0
    popular = [i for i, _ in appearing[:n_popular]]
    ordinary = [i for i, _ in appearing[n_popular:]]
    cold = [i for i in range(num_items) if counts.get(i, 0) == 0]

    summary = {
        "num_items": num_items,
        "appearing_items": n_appearing,
        "popular_count": len(popular),
        "ordinary_count": len(ordinary),
        "cold_count": len(cold),
        "popular_ratio": popular_ratio,
        "top_frequency_items": [(i, counts[i]) for i, _ in appearing[:10]],
        "bottom_frequency_items": [(i, counts[i]) for i, _ in appearing[-10:]],
    }
    if popular:
        summary["min_popular_count"] = counts[popular[-1]]
    if ordinary:
        summary["max_ordinary_count"] = counts[ordinary[0]]
    return {"popular": popular, "ordinary": ordinary, "cold": cold}, summary


def save_cache(config: Dict[str, Any], model_name: str, k: int,
               checkpoint: Path, counts: CounterType[int],
               categories: Dict[str, List[int]], summary: Dict[str, Any],
               batch_size: int) -> Path:
    out = rec_freq_path(config, model_name, k)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config["dataset"],
        "model": model_name,
        "k": k,
        "checkpoint": str(checkpoint),
        "batch_size": batch_size,
        "popular_ratio": summary["popular_ratio"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {str(i): c for i, c in counts.items()},
        "categories": categories,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def load_cache(config: Dict[str, Any], model_name: str, k: int,
               required: bool = False) -> Dict[str, Any] | None:
    """读取分类缓存；required=True 时缺失直接报错并提示先跑 classify。"""
    path = rec_freq_path(config, model_name, k)
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"推荐频次分类不存在: {path}\n"
                f"请先运行: python attacks/bandwagon/run.py --mode classify"
            )
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["counts"] = {int(i): c for i, c in data["counts"].items()}
    return data


def main(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = config["dataset"]
    model_name = config.get("model", {}).get("name", "lightgcn")
    cls_cfg = config.get("classification", {})
    k = cls_cfg.get("k") or config.get("training", {}).get("k") or 20
    popular_ratio = cls_cfg.get("popular_ratio", 0.2)
    batch_size = cls_cfg.get("batch_size", 1024)
    ckpt = resolve_clean_checkpoint(config)

    meta = load_meta(Path(str(RAW_META_PATH).format(dataset=dataset)))
    cfg = build_training_config(config, dataset, model_name)

    model_cls = get_model_cls(model_name)
    edge_index = torch.LongTensor(
        [[u, i] for u, i in meta["train_pairs"]]
    ).T
    model = model_cls(cfg, meta["num_users"], meta["num_items"], edge_index)
    ckpt_data = torch.load(ckpt, map_location=model._device, weights_only=True)
    model.load_state_dict(ckpt_data["model_state_dict"])
    print(f"[classify] 模型={model_name}，checkpoint={ckpt}，"
          f"用户={meta['num_users']}，物品={meta['num_items']}，k={k}")

    counts = compute_topk_counts(model, meta, k, batch_size=batch_size)
    categories, summary = classify_counts(counts, meta["num_items"], popular_ratio)
    out = save_cache(config, model_name, k, ckpt, counts, categories, summary,
                     batch_size)

    print(f"[classify] Top-{k} 覆盖物品 {summary['appearing_items']}/{summary['num_items']}"
          f"，划分：流行 {summary['popular_count']} / 普通 {summary['ordinary_count']}"
          f" / 冷门 {summary['cold_count']}")
    print(f"[classify] 流行阈值: 出现次数 ≥ {summary.get('min_popular_count')}；"
          f"普通最高频次: {summary.get('max_ordinary_count')}")
    print(f"[classify] 热门物品样例: {summary['top_frequency_items'][:5]}")
    print(f"[classify] 冷门物品样例: {categories['cold'][:5]}（共 "
          f"{len(categories['cold'])} 个）")
    print(f"[classify] 缓存 → {out}")
    return {"counts": counts, "categories": categories, "summary": summary,
            "cache_path": str(out)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bandwagon 推荐频次分类")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "bandwagon" / "config.yaml"),
    )
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
