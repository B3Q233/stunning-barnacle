"""TPA —— 数据层生成模块（Step 3 注入部分）

纯数据操作，不依赖 torch / 任何模型代码：
1. 读取 path_builder.py 产出的路径画像缓存（基座 + 传递路径 + 目标）
2. 把画像注入训练集 → 产出 poisoned meta.pkl + 注入统计

画像本身的构造（共现图 / 最短路径 / CF 距离）在 path_builder.py 中完成，
本模块只负责注入，保证数据层与模型解耦。

用法:
  python attacks/tpa/generate.py --config attacks/tpa/config.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from attacks.tpa.registry import active_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA


def raw_meta_path(config: Dict[str, Any]) -> Path:
    """干净数据 meta.pkl：models/{model.name}/data/processed/{dataset}/meta.pkl。"""
    dataset = config["dataset"]
    model_name = config.get("model", {}).get("name", "lightgcn")
    return PROJECT_ROOT / "models" / model_name / "data" / "processed" / dataset / "meta.pkl"


def poisoned_data_dir(config: Dict[str, Any]) -> Path:
    """中毒数据输出目录：attacks/{attack.name}/data/poisoned{_proxy}/{dataset}。

    代理模式带 _proxy 后缀，与白盒毒化数据隔离，避免互相覆盖。
    """
    dataset = config["dataset"]
    attack_name = config["attack"]["name"]
    tag = "_proxy" if config.get("surrogate", {}).get("enabled", False) else ""
    return PROJECT_ROOT / "attacks" / attack_name / "data" / f"poisoned{tag}" / dataset


def load_paths_cache(config: Dict[str, Any],
                     required: bool = False) -> Dict[str, Any] | None:
    """读取路径画像缓存（path_builder.py 产出）。"""
    from attacks.tpa.path_builder import load_paths_cache as _load
    return _load(config, required=required)


def load_rec_freq_cache(config: Dict[str, Any], model_name: str, k: int,
                        required: bool = False) -> Dict[str, Any] | None:
    """读取推荐频次分类缓存（classify.py 产出）。"""
    from attacks.tpa.classify import load_cache
    return load_cache(config, model_name, k, required=required)


# ── 数据 IO ──────────────────────────────────────────────
def load_meta(meta_path: Path) -> Dict[str, Any]:
    """加载预处理后的数据字典（与 LightGCN 预处理产出一致）。"""
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def save_meta(meta: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(meta, f)


def save_json(obj: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ── 攻击构造 ─────────────────────────────────────────────
def compute_item_popularity(train_pairs: List[Tuple[int, int]]) -> Counter:
    """全局物品流行度：每个物品在训练集中被交互的次数。"""
    return Counter(item for _, item in train_pairs)


def select_target_items(popularity: Counter, num_items: int, strategy: str,
                        count: int, ids: List[int], rng: random.Random,
                        categories: Dict[str, List[int]] | None = None,
                        category: str = "cold",
                        rec_counts: Dict[int, int] | None = None) -> List[int]:
    """选择攻击目标物品（与模板一致，供 path_builder 复用）。

    - specified: 手动指定 ID（推荐；目标物品由用户自行确定）
    - category:  按模型推荐频次分类挑选（需先跑 classify）
    - coldest:   训练集流行度最低的物品（旧口径，仅作对照）
    - random:    从有交互的物品中随机挑
    """
    candidates = sorted(i for i in range(num_items) if popularity[i] >= 1)
    if strategy == "specified":
        if not ids:
            raise ValueError(
                "strategy=specified 但 target_items.ids 为空，"
                "请填入要攻击的目标物品 ID"
            )
        selected = [int(i) for i in ids]
    elif strategy == "category":
        if categories is None:
            raise FileNotFoundError(
                "strategy=category 需要推荐频次分类缓存，"
                "请先运行 python attacks/tpa/run.py --mode classify"
            )
        if category not in categories:
            raise ValueError(
                f"未知分类 '{category}'，可选: {list(categories)}"
            )
        pool = categories[category]
        if category == "popular":
            selected = sorted(
                pool,
                key=lambda i: (-(rec_counts or {}).get(i, 0), popularity[i]),
            )[:count]
        else:
            selected = sorted(
                pool,
                key=lambda i: ((rec_counts or {}).get(i, 0), popularity[i]),
            )[:count]
    elif strategy == "random":
        selected = rng.sample(candidates, min(count, len(candidates)))
    elif strategy == "coldest":
        selected = sorted(candidates, key=lambda i: popularity[i])[:count]
    else:
        raise ValueError(
            f"未知的目标选择策略 '{strategy}'，可选: "
            f"specified | category | coldest | random"
        )

    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        print(f"[generate] 目标物品去重后仅 {len(selected)} 个（要求 {count}），使用可用数量")
    for i in selected:
        if not (0 <= i < num_items):
            raise ValueError(f"目标物品 ID {i} 超出范围 [0, {num_items})")
    return selected


def inject(meta: Dict[str, Any], profiles: List[Dict[str, Any]]
           ) -> Dict[str, Any]:
    """把 TPA 画像注入训练集，返回中毒后的 meta 字典。"""
    num_users = meta["num_users"]
    new_pairs = list(meta["train_pairs"])
    new_user_items = {u: set(items) for u, items in meta["user_items"].items()}

    for p in profiles:
        fake_uid = num_users + p["fake_user"]
        items = p["items"]
        new_pairs.extend((fake_uid, i) for i in items)
        new_user_items[fake_uid] = set(items)

    poisoned = dict(meta)
    poisoned["num_users"] = num_users + len(profiles)
    poisoned["train_pairs"] = new_pairs
    poisoned["user_items"] = new_user_items
    return poisoned


# ── 主流程 ───────────────────────────────────────────────
def main(config: Dict[str, Any], raw_meta: Path | None = None,
         out_dir: Path | None = None) -> Dict[str, Any]:
    """执行 TPA 数据注入，产出 poisoned meta.pkl。

    画像来自 path_builder 缓存（先运行 --mode paths）。
    返回注入统计字典（同时写入 stats.json）。
    """
    dataset = config["dataset"]
    seed = config.get("seed", 42)
    victim_model_name = config.get("model", {}).get("name", "lightgcn")
    embedding_model_name = active_model_name(config)
    attack_name = config["attack"]["name"]
    k = config.get("training", {}).get("k") or 20

    meta = load_meta(raw_meta or raw_meta_path(config))
    cache = load_paths_cache(config, required=True)
    profiles = cache["profiles"]
    targets = cache["targets"]
    path_stats = cache["path_stats"]

    num_users, num_items = meta["num_users"], meta["num_items"]
    popularity = compute_item_popularity(meta["train_pairs"])
    rec_cache = load_rec_freq_cache(config, embedding_model_name, k, required=False)
    categories = rec_cache["categories"] if rec_cache else None

    poisoned = inject(meta, profiles)

    # ── 统计与验证 ──
    before_cnt = len(meta["train_pairs"])
    after_cnt = len(poisoned["train_pairs"])
    added = sum(len(p["items"]) for p in profiles)
    assert after_cnt == before_cnt + added, (
        f"注入数量不一致: {after_cnt} != {before_cnt} + {added}"
    )

    per_target_counts: Dict[int, int] = {}
    for p in profiles:
        per_target_counts[p["target"]] = per_target_counts.get(p["target"], 0) + 1

    stats = {
        "dataset": dataset,
        "attack": attack_name,
        "model": victim_model_name,
        "embedding_model": embedding_model_name,
        "surrogate": {
            "enabled": bool(config.get("surrogate", {}).get("enabled", False)),
            "model_name": (
                config.get("surrogate", {}).get("model_name")
                if config.get("surrogate", {}).get("enabled", False) else None
            ),
        },
        "seed": seed,
        "num_users_before": num_users,
        "num_users_after": poisoned["num_users"],
        "num_fake_users": len(profiles),
        "targets": [
            {
                "item_id": t,
                "popularity_before": popularity[t],
                "rec_count": rec_cache["counts"].get(t, 0) if rec_cache else None,
                "category": (
                    "popular" if rec_cache and t in categories["popular"]
                    else "ordinary" if rec_cache and t in categories["ordinary"]
                    else "cold" if rec_cache and t in categories["cold"]
                    else None
                ),
                "fake_users": per_target_counts[t],
            }
            for t in targets
        ],
        "train_pairs_before": before_cnt,
        "train_pairs_after": after_cnt,
        "injected_pairs": added,
        "path_stats": path_stats,
    }

    out = out_dir or poisoned_data_dir(config)
    save_meta(poisoned, out / "meta.pkl")
    save_json(profiles, out / "profiles.json")
    save_json(stats, out / "stats.json")

    print(f"[{attack_name}] 数据集: {dataset}（{num_users} 用户 / {num_items} 物品）")
    print(f"[{attack_name}] 目标物品: {targets}，流行度: "
          f"{[popularity[t] for t in targets]}")
    if rec_cache:
        print(f"[{attack_name}] 目标物品分类: "
              f"{[stats['targets'][j]['category'] for j in range(len(targets))]}")
    print(f"[{attack_name}] 假用户: {len(profiles)}，画像 = 基座 + 路径 + 目标，"
          f"找到路径 {path_stats['paths_found']} / 回退直连 {path_stats['fallback_direct']}")
    print(f"[{attack_name}] 注入前训练交互: {before_cnt} → 注入后: {after_cnt} "
          f"（+{added}）")
    print(f"[{attack_name}] 输出 → {out / 'meta.pkl'}")
    return stats


def load_yaml_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPA 数据注入")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "tpa" / "config.yaml"))
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
