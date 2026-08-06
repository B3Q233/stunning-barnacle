"""Random（随机）攻击 —— 数据层生成模块

纯数据操作，不依赖 torch / 任何模型代码：
1. （可选前置）读取 classify.py 产出的推荐频次分类缓存
2. 选择目标物品（默认 strategy=specified，由用户自行指定 ID）
3. 为每个目标构造假用户画像：[K 个随机物品 + 目标物品]
   （经典 random attack：filler 从全量物品中均匀随机采样，不依赖流行度；
   参考 Lam & Riedl, Shilling Recommender Systems for Fun and Profit, WWW 2004）
4. 注入训练集 → 产出 poisoned meta.pkl + 注入统计

用法:
  python attacks/random/generate.py --config attacks/random/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA


def raw_meta_path(config: Dict[str, Any]) -> Path:
    """干净数据 meta.pkl：models/{model.name}/data/processed/{dataset}/meta.pkl。"""
    dataset = config["dataset"]
    model_name = config.get("model", {}).get("name", "lightgcn")
    return PROJECT_ROOT / "models" / model_name / "data" / "processed" / dataset / "meta.pkl"


def poisoned_data_dir(config: Dict[str, Any]) -> Path:
    """中毒数据输出目录：attacks/{attack.name}/data/poisoned/{dataset}。"""
    dataset = config["dataset"]
    attack_name = config["attack"]["name"]
    return PROJECT_ROOT / "attacks" / attack_name / "data" / "poisoned" / dataset


def load_rec_freq_cache(config: Dict[str, Any], model_name: str, k: int,
                        required: bool = False) -> Dict[str, Any] | None:
    """读取推荐频次分类缓存（classify.py 产出）。"""
    from attacks.random.classify import load_cache
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
    """选择攻击目标物品。

    - specified: 手动指定 ID（推荐；目标物品由用户自行确定）
    - category:  按模型推荐频次分类挑选（需先跑 classify）
                 popular=最热 / ordinary、cold=该分类中相对最冷的
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
                "请先运行 python attacks/random/run.py --mode classify"
            )
        if category not in categories:
            raise ValueError(
                f"未知分类 '{category}'，可选: {list(categories)}"
            )
        pool = categories[category]
        if category == "popular":
            # 流行：取推荐频次最高的 count 个（平手按训练流行度）
            selected = sorted(
                pool,
                key=lambda i: (-(rec_counts or {}).get(i, 0), popularity[i]),
            )[:count]
        else:
            # 普通/冷门：取该分类中相对最冷的 count 个（攻击意义最大）
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

    # 去重并校验
    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        print(f"[generate] 目标物品去重后仅 {len(selected)} 个（要求 {count}），使用可用数量")
    for i in selected:
        if not (0 <= i < num_items):
            raise ValueError(f"目标物品 ID {i} 超出范围 [0, {num_items})")
    return selected


def generate_fake_profiles(num_fake_users: int, filler_size: int,
                           targets: List[int], filler_pool: List[int],
                           rng: random.Random) -> List[Tuple[int, int, List[int]]]:
    """为每个目标生成假用户画像（random 攻击语义）。

    每个假用户画像 = 目标物品 + filler_size 个从 filler_pool（全量物品）
    中均匀随机采样的物品，与物品流行度无关。
    返回 profiles: [(fake_user_id, target_item, [filler_items + target])]
    假用户 ID 从 0 起顺序分配（与真实用户 ID 不冲突，由调用方传入起始值语义保证：
    这里返回的是组内序号，注入时统一偏移到 num_users 之后）。
    """
    if num_fake_users <= 0 or not targets:
        raise ValueError("假用户数和目标物品数必须为正")
    if filler_size <= 0:
        raise ValueError("filler_size 必须为正")

    if len(filler_pool) < filler_size:
        raise ValueError(
            f"filler 池仅 {len(filler_pool)} 个，小于 filler_size={filler_size}"
        )

    # 假用户均分给各目标（余数分配给前几个目标）
    per_target = num_fake_users // len(targets)
    remainder = num_fake_users % len(targets)

    profiles: List[Tuple[int, int, List[int]]] = []
    uid = 0
    for t_idx, target in enumerate(targets):
        n = per_target + (1 if t_idx < remainder else 0)
        for _ in range(n):
            pool = [i for i in filler_pool if i != target]
            fillers = rng.sample(pool, min(filler_size, len(pool)))
            items = [target] + fillers
            rng.shuffle(items)  # 交互顺序无关紧要（隐式反馈），仅增加画像多样性
            profiles.append((uid, target, items))
            uid += 1
    return profiles


def inject(meta: Dict[str, Any], profiles: List[Tuple[int, int, List[int]]]
           ) -> Dict[str, Any]:
    """把假用户画像注入训练集，返回中毒后的 meta 字典。"""
    num_users = meta["num_users"]
    new_pairs = list(meta["train_pairs"])
    new_user_items = {u: set(items) for u, items in meta["user_items"].items()}

    for uid, _target, items in profiles:
        fake_uid = num_users + uid
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
    """执行攻击数据注入，产出 poisoned meta.pkl。

    返回注入统计字典（同时写入 stats.json）。
    """
    dataset = config["dataset"]
    attack_cfg = config["attack"]
    seed = config.get("seed", 42)
    rng = random.Random(seed)
    model_name = config.get("model", {}).get("name", "lightgcn")
    k = config.get("training", {}).get("k") or 20
    attack_name = config["attack"]["name"]

    meta_path = raw_meta or raw_meta_path(config)
    meta = load_meta(meta_path)

    num_users, num_items = meta["num_users"], meta["num_items"]
    popularity = compute_item_popularity(meta["train_pairs"])
    rec_cache = load_rec_freq_cache(config, model_name, k, required=False)
    categories = rec_cache["categories"] if rec_cache else None

    # 假用户数量：显式优先，其次按 ratio
    num_fake = attack_cfg.get("num_fake_users")
    if num_fake is None:
        num_fake = max(1, int(num_users * attack_cfg.get("ratio", 0.01)))

    ti_cfg = attack_cfg["target_items"]
    targets = select_target_items(
        popularity, num_items,
        ti_cfg.get("strategy", "specified"),
        ti_cfg.get("count", 3),
        ti_cfg.get("ids", []),
        rng,
        categories=categories,
        category=ti_cfg.get("category", "cold"),
        rec_counts=rec_cache["counts"] if rec_cache else None,
    )

    # random 攻击：filler 池 = 全量物品，均匀随机采样（与流行度无关）
    filler_pool = list(range(num_items))
    print(f"[generate] filler 池 = 全量物品（random 攻击，共 "
          f"{len(filler_pool)} 个，均匀随机采样）")

    profiles = generate_fake_profiles(
        num_fake,
        attack_cfg.get("filler_size", 20),
        targets,
        filler_pool,
        rng,
    )

    poisoned = inject(meta, profiles)

    # ── 统计与验证 ──
    before_cnt = len(meta["train_pairs"])
    after_cnt = len(poisoned["train_pairs"])
    added = sum(len(items) for _u, _t, items in profiles)
    assert after_cnt == before_cnt + added, (
        f"注入数量不一致: {after_cnt} != {before_cnt} + {added}"
    )

    per_target_counts: Dict[int, int] = {}
    for _u, t, _items in profiles:
        per_target_counts[t] = per_target_counts.get(t, 0) + 1

    stats = {
        "dataset": dataset,
        "attack": attack_name,
        "model": model_name,
        "seed": seed,
        "num_users_before": num_users,
        "num_users_after": poisoned["num_users"],
        "num_fake_users": len(profiles),
        "filler_size": attack_cfg.get("filler_size", 20),
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
    }

    out = out_dir or poisoned_data_dir(config)
    save_meta(poisoned, out / "meta.pkl")
    save_json(
        [{"fake_user": u, "target": t, "items": items} for u, t, items in profiles],
        out / "profiles.json",
    )
    save_json(stats, out / "stats.json")

    print(f"[{attack_name}] 数据集: {dataset}（{num_users} 用户 / {num_items} 物品）")
    print(f"[{attack_name}] 目标物品: {targets}，流行度: "
          f"{[popularity[t] for t in targets]}")
    if rec_cache:
        print(f"[{attack_name}] 目标物品分类: "
              f"{[stats['targets'][j]['category'] for j in range(len(targets))]}")
    print(f"[{attack_name}] 假用户: {len(profiles)}，每个交互 "
          f"{attack_cfg.get('filler_size', 20)} 随机 + 1 目标")
    print(f"[{attack_name}] 注入前训练交互: {before_cnt} → 注入后: {after_cnt} "
          f"（+{added}）")
    for t in targets:
        print(f"  [target {t}] 假用户数={per_target_counts[t]}，"
              f"注入后该物品交互数 = {popularity[t] + per_target_counts[t]}")
    print(f"[{attack_name}] 输出 → {out / 'meta.pkl'}")
    return stats


def load_yaml_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="攻击数据注入")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "random" / "config.yaml"))
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
