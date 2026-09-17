"""UBA 攻击 —— 数据层生成模块（第 2/3 步：处理效应读取 → 预算分配 → 画像注入）

纯数据操作：不 import torch、不构造模型实例（只有 treatment.method=surrogate 且
缓存缺失时才会惰性调用 estimate.py 去补算，因为那条支路本身依赖代理模型）。

流程：
1. 读预处理 meta（train/test 成对交互）
2. 选目标物品（默认 specified；类别策略需要先跑 classify）
3. 选目标用户 U_t（uplift.select_target_users）
4. 取处理效应矩阵 Y（estimate 缓存；path 支路缺失时现算并回写）
5. 预算分配 T*（uplift.allocate：uba / uniform_target / random_all）
6. 按画像规则实例化假档案（filler 来源：模板用户 / 流行池 / 全量随机）
7. 注入训练集 → meta.pkl + profiles.json + stats.json + config 快照 + latest.json

用法:
  python attacks/uba/generate.py --config attacks/uba/config.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 默认干净 meta（batch/aggregate.py 依赖该常量名做兜底；{dataset} 由调用方展开）
DEFAULT_RAW_META = (PROJECT_ROOT / "models" / "lightgcn" / "data" / "processed"
                    / "{dataset}" / "meta.pkl")
DEFAULT_OUT_DIR = (PROJECT_ROOT / "attacks" / "uba" / "data" / "poisoned"
                   / "{dataset}" / "{model}")

from attacks.uba.uplift import (  # noqa: E402
    accessible_template_users,
    allocate,
    three_hop_path_effect,
    interaction_sets,
    select_target_users,
)
from training.config_utils import DEFAULT_K  # noqa: E402
from training.run_tag import (  # noqa: E402
    resolve_run_tag,
    save_config_snapshot,
    write_latest_pointer,
)
from training.timing import timed  # noqa: E402


VALID_FILLER_SOURCES = ("template_user", "popular", "random")


# ── 配置 / 数据 IO ────────────────────────────────────────────
def load_yaml_config(path: Path) -> Dict[str, Any]:
    """读 yaml → canonicalize → 展开 {k} 模板（与其它攻击模块一致）。"""
    from training.config_utils import apply_k, load_config

    return apply_k(load_config(path))


def raw_meta_path(config: Dict[str, Any],
                  model_name: str | None = None) -> Path:
    """干净 meta 路径：优先用受害模型自己的 processed 目录，缺失时回退 lightgcn。

    为什么：仓库里 lightgcn/mf/wmf 都各自保存了一份相同口径的 ml100k meta，
    攻击侧只需要"成对交互 + 用户/物品数"，因此优先跟随受害模型，避免路径写死。
    """
    dataset = str(config["dataset"])
    name = model_name or config.get("model", {}).get("name", "lightgcn")
    primary = PROJECT_ROOT / "models" / name / "data" / "processed" / dataset / "meta.pkl"
    if primary.exists():
        return primary
    return Path(str(DEFAULT_RAW_META).format(dataset=dataset))


def load_meta(meta_path: Path) -> Dict[str, Any]:
    """加载预处理后的数据字典（num_users / num_items / train_pairs / test_pairs / user_items）。"""
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


# ── 目标物品选择（与其它攻击模块同口径）──────────────────────
def compute_item_popularity(train_pairs: Sequence[Tuple[int, int]]) -> Counter:
    """全局物品流行度 = 该物品在训练集中被交互的次数（含假用户? 否，调用方过滤）。"""
    return Counter(int(i) for _, i in train_pairs)


def select_target_items(popularity: Counter, num_items: int, strategy: str,
                        count: int, ids: List[int], rng: random.Random,
                        categories: Dict[str, List[int]] | None = None,
                        category: str = "cold",
                        rec_counts: Dict[int, int] | None = None) -> List[int]:
    """选择攻击目标物品（specified | category | coldest | random）。

    与 bandwagon/random/tpa 的策略语义一致；UBA 只接受单目标物品，因此
    count>1 会在 main() 里直接报错，而不是静默取第一个。
    """
    candidates = sorted(i for i in range(num_items) if popularity[i] >= 1)
    if strategy == "specified":
        if not ids:
            raise ValueError(
                "strategy=specified 但 attack.target_items.ids 为空，请填入目标物品 ID"
            )
        selected = [int(i) for i in ids]
    elif strategy == "category":
        if categories is None:
            raise FileNotFoundError(
                "strategy=category 需要交互数分类缓存，"
                "请先运行 python attacks/uba/run.py --mode classify"
            )
        if category not in categories:
            raise ValueError(f"未知分类 {category!r}，可选: {list(categories)}")
        pool = list(categories[category])
        if category == "popular":
            selected = sorted(
                pool, key=lambda i: (-(rec_counts or {}).get(i, 0), popularity[i])
            )[:count]
        else:
            selected = sorted(
                pool, key=lambda i: ((rec_counts or {}).get(i, 0), popularity[i])
            )[:count]
    elif strategy == "random":
        selected = rng.sample(candidates, min(count, len(candidates)))
    elif strategy == "coldest":
        selected = sorted(candidates, key=lambda i: popularity[i])[:count]
    else:
        raise ValueError(
            f"未知的目标选择策略 {strategy!r}，可选: specified | category | coldest | random"
        )

    selected = list(dict.fromkeys(selected))
    for i in selected:
        if not (0 <= i < num_items):
            raise ValueError(f"目标物品 ID {i} 超出范围 [0, {num_items})")
    return selected


# ── 处理效应缓存 ─────────────────────────────────────────────
def estimate_dir(config: Dict[str, Any], model_name: str) -> Path:
    """处理效应共享缓存目录（与 run_tag 无关：同一目标物品的 Y 可复用）。"""
    return (PROJECT_ROOT / "attacks" / "uba" / "data" / "estimate"
            / str(config["dataset"]) / model_name)


def effect_cache_path(config: Dict[str, Any], model_name: str, target_item: int,
                      method: str, repeats: int = 0, hit_k: int = 0,
                      alpha: float = 1.0, beta: float = 1.0,
                      seed: int = 0) -> Path:
    """缓存文件名把"会改变 Y 的所有参数"编进去，避免不同配置互相覆盖。"""
    treatment = treatment_cfg(config)
    max_per_user = int(treatment.get("max_per_user", 6))
    if method == "path":
        name = (f"item{int(target_item)}_h{max_per_user}_path"
                f"_a{alpha:g}_b{beta:g}.json")
    else:
        name = (f"item{int(target_item)}_h{max_per_user}_surrogate"
                f"_E{int(repeats)}_k{int(hit_k)}_s{int(seed)}_a{alpha:g}_b{beta:g}.json")
    return estimate_dir(config, model_name) / name


def treatment_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """取 attack.uba.treatment 段；缺失时给出明确报错（不静默用默认值）。"""
    uba = config.get("attack", {}).get("uba")
    if not isinstance(uba, dict) or "treatment" not in uba:
        raise ValueError(
            "配置缺少 attack.uba.treatment 段；请参照 "
            "TPA/docs/config-template.unified.yaml 的 canonical 键补齐"
        )
    return uba["treatment"]


def profile_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    uba = config.get("attack", {}).get("uba", {})
    return uba.get("profile", {}) if isinstance(uba, dict) else {}


def allocation_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    uba = config.get("attack", {}).get("uba", {})
    return uba.get("allocation", {}) if isinstance(uba, dict) else {}


def target_users_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    uba = config.get("attack", {}).get("uba", {})
    return uba.get("target_users", {}) if isinstance(uba, dict) else {}


def _effect_profile_fn(filler_source: str, filler_size: int,
                       user_items: Dict[int, set], num_items: int,
                       popular_items: List[int],
                       rng: random.Random):
    """构造给 uplift.three_hop_path_effect 用的画像函数。

    为什么：path 支路只有"假档案长什么样"与真实注入一致时，Y 才代表真实攻击；
    因此这里复用与 build_fake_profiles 相同的 filler 采样规则（同一函数）。
    """
    def profile_fn(template_user: int) -> List[int]:
        pool = filler_pool(filler_source, template_user, user_items,
                           popular_items, num_items)
        fillers = sample_fillers(pool, filler_size,
                                 popular_items or list(range(num_items)), rng)
        return fillers

    return profile_fn


def load_or_build_effect(config: Dict[str, Any], meta: Dict[str, Any],
                         target_item: int, target_users: List[int],
                         model_name: str,
                         popular_items: List[int] | None = None) -> Dict[str, Any]:
    """取处理效应矩阵 Y：命中缓存则读，否则按 method 计算并回写缓存。

    - method=path：纯数据（uplift.three_hop_path_effect），可在 data 阶段直接补算；
    - method=surrogate：需要代理模型，惰性 import estimate.py（模块级不 import torch，
      保证 path 支路仍是纯数据层）。
    """
    treatment = treatment_cfg(config)
    method = str(treatment.get("method", "path"))
    if method not in ("path", "surrogate"):
        raise ValueError(f"未知 treatment.method {method!r}，可选 path | surrogate")
    max_per_user = int(treatment.get("max_per_user", 6))
    repeats = int(treatment.get("repeats", 10))
    hit_k = int(treatment.get("hit_k", 20))
    alpha = float(treatment.get("alpha", 1.0))
    beta = float(treatment.get("beta", 1.0))
    seed = int(config.get("seed", 42))

    path = effect_cache_path(config, model_name, target_item, method,
                             repeats=repeats, hit_k=hit_k, alpha=alpha,
                             beta=beta, seed=seed)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["effect"] = _effect_from_json(payload)
        print(f"[generate] 命中处理效应缓存 → {path}")
        return payload

    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    user_items = interaction_sets(meta["train_pairs"], max_uid=num_users)
    filler_size = int(config["attack"].get("filler_size", 36))
    filler_source = str(profile_cfg(config).get("filler_source", "template_user"))
    if popular_items is None:
        popular_items = popular_pool(config, model_name, meta)

    if method == "path":
        rng = random.Random(seed)
        profile_fn = _effect_profile_fn(filler_source, filler_size, user_items,
                                        num_items, popular_items, rng)
        effect = three_hop_path_effect(
            user_items, target_users, target_item, max_per_user, num_items,
            num_users=num_users, alpha=alpha, beta=beta,
            profile_fn=profile_fn, rng=rng)
        payload = {
            "method": "path", "target_item": int(target_item),
            "target_users": [int(u) for u in target_users],
            "max_per_user": max_per_user, "alpha": alpha, "beta": beta,
            "filler_source": filler_source, "filler_size": filler_size,
            "effect": effect,
        }
    else:
        # surrogate 支路：需要代理模型，惰性 import（避免 path 支路引入 torch）
        from attacks.uba.estimate import treatment_effect_surrogate

        payload = treatment_effect_surrogate(config, meta, target_item, target_users)
        payload["filler_source"] = filler_source
        payload["filler_size"] = filler_size

    effect = np.asarray(payload["effect"], dtype=np.float64)
    save_json({**payload, "effect": effect.tolist()}, path)
    print(f"[generate] 处理效应（method={method}）已计算并缓存 → {path}")
    payload["effect"] = effect
    return payload


def _effect_from_json(payload: Dict[str, Any]) -> np.ndarray:
    """缓存 JSON 里的 effect 是嵌套 list，读回来统一转 ndarray。"""
    return np.asarray(payload["effect"], dtype=np.float64)


def popular_pool(config: Dict[str, Any], model_name: str,
                 meta: Dict[str, Any]) -> List[int]:
    """流行物品池：优先用 classify 缓存的 popular 档；缺失时回退全局训练集热门。

    为什么：`filler_source=popular` 与 target_items.strategy=category 都依赖同一份
    分类缓存；缺失时回退全局热门是为了让 smoke 测试不必先跑 classify，但会打印告警。
    """
    from attacks.uba.classify import load_cache

    cache = load_cache(config, model_name, int(config.get("k", DEFAULT_K)))
    if cache is not None:
        return [int(i) for i in cache["categories"]["popular"]]
    popularity = compute_item_popularity(meta["train_pairs"])
    top_n = max(50, int(config["attack"].get("filler_size", 36)) * 5)
    print("[generate] [!] 无 classify 缓存，流行池回退为训练集交互数 Top-"
          f"{top_n}；建议先运行 --mode classify")
    return [int(i) for i, _ in popularity.most_common(top_n)]


# ── 假档案构造与注入 ─────────────────────────────────────────
def filler_pool(filler_source: str, template_user: int,
                user_items: Dict[int, set], popular_items: List[int],
                num_items: int) -> List[int]:
    """按 filler_source 取候选物品池（画像 filler 的采样范围）。"""
    if filler_source == "template_user":
        return sorted(int(i) for i in user_items.get(int(template_user), set()))
    if filler_source == "popular":
        return [int(i) for i in popular_items]
    if filler_source == "random":
        return list(range(int(num_items)))
    raise ValueError(
        f"未知 profile.filler_source {filler_source!r}，可选 {VALID_FILLER_SOURCES}"
    )


def sample_fillers(pool: Sequence[int], filler_size: int,
                   fallback: Sequence[int], rng: random.Random) -> List[int]:
    """无放回采样 filler_size 个物品；池不足时用 fallback（流行池 / 全量物品）补齐。

    为什么需要补齐：模板用户（尤其冷门目标物品的轻交互用户）可能只有两三个交互，
    直接采样会得到远小于 filler_size 的画像，攻击强度与其他用户不可比。
    """
    if filler_size <= 0:
        raise ValueError(f"filler_size 必须为正，实际 {filler_size}")
    picked: List[int] = []
    candidates = [int(i) for i in dict.fromkeys(pool)]
    if candidates:
        picked = rng.sample(candidates, min(filler_size, len(candidates)))
    if len(picked) < filler_size and fallback:
        chosen = set(picked)
        rest = [int(i) for i in dict.fromkeys(fallback) if int(i) not in chosen]
        rng.shuffle(rest)
        picked.extend(rest[: filler_size - len(picked)])
    return picked


def build_fake_profiles(template_users: Sequence[int], target_item: int,
                        filler_size: int, filler_source: str,
                        user_items: Dict[int, set], num_items: int,
                        popular_items: List[int],
                        rng: random.Random) -> List[Dict[str, Any]]:
    """按"模板用户序列"实例化假档案（UBA 的 instantiation 环节）。

    为什么用 template_users 而不是 [uid] * t_u：官方代码里 way 1/2/3 最终都归约为
    `idx = [uid, ...]`（第 k 个假用户照哪个真实用户的画像生成）再 `train_data_array[idx]`；
    把三种分配策略统一成同一个序列，注入代码就只有一条路径，不会出现"某种策略
    绕过画像规则"的静默差异。

    返回 [{fake_user, target, template_user, items}, ...]；
    - fake_user 是从 0 起的**组内序号**，注入时统一偏移到 num_users 之后；
    - items = filler 物品（去重、剔除目标物品）+ 目标物品（恰好出现一次）。
    """
    profiles: List[Dict[str, Any]] = []
    fallback = popular_items or list(range(int(num_items)))
    for uid, template in enumerate(template_users):
        pool = filler_pool(filler_source, int(template), user_items,
                           popular_items, num_items)
        fillers = sample_fillers(pool, int(filler_size), fallback, rng)
        items = sorted({int(i) for i in fillers if int(i) != int(target_item)})
        items.append(int(target_item))
        profiles.append({
            "fake_user": uid,
            "target": int(target_item),
            "template_user": int(template),
            "items": items,
        })
    return profiles


def inject(meta: Dict[str, Any],
           profiles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把假档案注入训练集，返回中毒后的 meta（不改动入参 meta）。"""
    num_users = int(meta["num_users"])
    new_pairs = list(meta["train_pairs"])
    new_user_items = {int(u): set(items) for u, items in meta["user_items"].items()}

    for p in profiles:
        fake_uid = num_users + int(p["fake_user"])
        items = [int(i) for i in p["items"]]
        new_pairs.extend((fake_uid, i) for i in items)
        new_user_items[fake_uid] = set(items)

    poisoned = dict(meta)
    poisoned["num_users"] = num_users + len(profiles)
    poisoned["train_pairs"] = new_pairs
    poisoned["user_items"] = new_user_items
    return poisoned


def allocation_histogram(allocation: Dict[int, int]) -> Dict[str, int]:
    """分配直方图 {t_u: 用户数}（写进 stats.json，便于核对 DP 结果形状）。"""
    hist: Dict[str, int] = {}
    for t in allocation.values():
        hist[str(int(t))] = hist.get(str(int(t)), 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def resolve_target_item(config: Dict[str, Any], meta: Dict[str, Any],
                        model_name: str,
                        rng: random.Random | None = None) -> Dict[str, Any]:
    """解析唯一的目标物品（UBA 单目标约束集中在此校验，generate/estimate 共用）。

    为什么把 count!=1 放在这里报错：UBA 的预算分配是"目标用户 × 档位"的分组背包，
    多目标物品需要跨物品联合分配（论文未定义），静默取第一个会给出看似正常但语义
    错误的结果。

    返回 {"item_id", "popularity", "rec_cache"}；popularity/rec_cache 供 stats 复用，
    避免重复统计。
    """
    from attacks.uba.classify import load_cache

    attack_cfg = config["attack"]
    ti_cfg = attack_cfg["target_items"]
    if int(ti_cfg.get("count", 1)) != 1:
        raise ValueError(
            "UBA 只支持单目标物品（attack.target_items.count 必须为 1）；"
            "多目标需要跨物品联合分配预算，本版本未实现"
        )
    rng = rng or random.Random(int(config.get("seed", 42)))
    num_items = int(meta["num_items"])
    popularity = compute_item_popularity(meta["train_pairs"])
    rec_cache = load_cache(config, model_name, int(config.get("k", DEFAULT_K)))
    categories = rec_cache["categories"] if rec_cache else None
    targets = select_target_items(
        popularity, num_items, ti_cfg.get("strategy", "specified"),
        int(ti_cfg.get("count", 1)), list(ti_cfg.get("ids", [])), rng,
        categories=categories, category=ti_cfg.get("category", "cold"),
        rec_counts=rec_cache["counts"] if rec_cache else None)
    if len(targets) != 1:
        raise ValueError(f"UBA 需要恰好 1 个目标物品，实际得到 {targets}")
    return {"item_id": int(targets[0]), "popularity": popularity,
            "rec_cache": rec_cache}


# ── 主流程 ───────────────────────────────────────────────────
@timed("UBA 数据注入")
def main(config: Dict[str, Any], raw_meta: Path | None = None,
         out_dir: Path | None = None) -> Dict[str, Any]:
    """data 阶段入口：产出中毒 meta.pkl + profiles.json + stats.json。"""
    dataset = str(config["dataset"])
    attack_cfg = config["attack"]
    model_name = config.get("model", {}).get("name", "lightgcn")
    seed = int(config.get("seed", 42))
    k = int(config.get("k", DEFAULT_K))
    rng = random.Random(seed)
    tag = resolve_run_tag(config)

    meta_path = raw_meta or raw_meta_path(config, model_name)
    meta = load_meta(meta_path)
    num_users, num_items = int(meta["num_users"]), int(meta["num_items"])

    # 0) 预算 N
    budget = attack_cfg.get("num_fake_users")
    if budget is None:
        budget = int(round(num_users * float(attack_cfg.get("ratio") or 0.0)))
    budget = int(budget)
    if budget <= 0:
        raise ValueError(
            f"假用户预算 N 必须为正（attack.num_fake_users={attack_cfg.get('num_fake_users')}，"
            f"attack.ratio={attack_cfg.get('ratio')}）"
        )

    # 1) 目标物品（UBA 只支持单目标：多目标需要跨物品联合分配，本版本不支持）
    target_info = resolve_target_item(config, meta, model_name, rng)
    target_item = int(target_info["item_id"])
    popularity = target_info["popularity"]
    rec_cache = target_info["rec_cache"]
    categories = rec_cache["categories"] if rec_cache else None

    # 2) 目标用户 U_t
    tu_cfg = target_users_cfg(config)
    tu_info = select_target_users(meta, target_item, tu_cfg, seed)
    target_users = [int(u) for u in tu_info["users"]]

    # 3) 处理效应 Y（缓存优先）
    popular_items = popular_pool(config, model_name, meta)
    effect_info = load_or_build_effect(config, meta, target_item, target_users,
                                       model_name, popular_items=popular_items)
    effect = np.asarray(effect_info["effect"], dtype=np.float64)
    max_per_user = int(treatment_cfg(config).get("max_per_user", 6))
    if effect.shape != (len(target_users), max_per_user + 1):
        raise ValueError(
            f"处理效应矩阵 shape {effect.shape} 与 "
            f"(|U_t|={len(target_users)}, H+1={max_per_user + 1}) 不一致；"
            f"缓存文件可能来自其它目标用户配置，请删除后重跑"
        )

    # 4) 预算分配
    alloc_cfg = allocation_cfg(config)
    strategy = str(alloc_cfg.get("strategy", "uba"))
    template_pool: List[int] = []
    if strategy == "random_all":
        pool_ratio = float(tu_cfg.get("accessible_ratio", 1.0))
        template_pool = accessible_template_users(meta, ratio=pool_ratio, seed=seed)
    allocation = allocate(effect, target_users, strategy, budget, max_per_user,
                          seed=seed, template_pool=template_pool, rng=rng)

    # 5) 画像实例化 + 注入
    user_items = interaction_sets(meta["train_pairs"], max_uid=num_users)
    filler_size = int(attack_cfg.get("filler_size", 36))
    filler_source = str(profile_cfg(config).get("filler_source", "template_user"))
    if filler_source not in VALID_FILLER_SOURCES:
        raise ValueError(
            f"未知 profile.filler_source {filler_source!r}，可选 {VALID_FILLER_SOURCES}")
    # 复现预算：random_all baseline 也需要同一个随机流位置，因此先于画像构造初始化
    profile_rng = random.Random(seed)
    profiles = build_fake_profiles(
        allocation["template_users"], target_item, filler_size, filler_source,
        user_items, num_items, popular_items, profile_rng)
    poisoned = inject(meta, profiles)

    # ── 统计与硬门禁 ──
    before_cnt = len(meta["train_pairs"])
    after_cnt = len(poisoned["train_pairs"])
    injected = sum(len(p["items"]) for p in profiles)
    assert after_cnt == before_cnt + injected, (
        f"注入数量不一致: {after_cnt} != {before_cnt} + {injected}"
    )
    assert len(profiles) == allocation["num_fake_users"], (
        f"画像数 {len(profiles)} 与分配出的假用户数 "
        f"{allocation['num_fake_users']} 不一致"
    )
    for p in profiles:  # 每个假画像必须恰好包含一次目标物品
        assert p["items"].count(target_item) == 1, p
    for p in profiles:  # 假用户 id 不得与真实用户冲突
        assert (num_users + int(p["fake_user"])) < poisoned["num_users"]

    template_counts = Counter(int(p["template_user"]) for p in profiles)
    stats = {
        "dataset": dataset,
        "attack": "uba",
        "model": model_name,
        "run_tag": tag,
        "seed": seed,
        "k": k,
        "num_users_before": num_users,
        "num_users_after": int(poisoned["num_users"]),
        "num_fake_users": len(profiles),
        "budget": budget,
        "max_per_user": max_per_user,
        "filler_size": filler_size,
        "filler_source": filler_source,
        "treatment": {
            "method": str(treatment_cfg(config).get("method", "path")),
            "alpha": float(treatment_cfg(config).get("alpha", 1.0)),
            "beta": float(treatment_cfg(config).get("beta", 1.0)),
            "hit_k": int(treatment_cfg(config).get("hit_k", 20)),
            "repeats": int(treatment_cfg(config).get("repeats", 10)),
        },
        "allocation": {
            "strategy": allocation["strategy"],
            "estimated_value": allocation["estimated_value"],
            "unused_budget": allocation["unused_budget"],
            "histogram": allocation_histogram(allocation["allocation"]),
        },
        "target_users": {
            "strategy": tu_info["strategy"],
            "count": len(target_users),
            "n_candidates": tu_info["n_candidates"],
            "ids": target_users,
            "category_size": int(tu_cfg.get("category_size", 20)),
            "max_category_interactions": int(
                tu_cfg.get("max_category_interactions", 10)),
            "category_items": [int(i) for i in tu_info["category_items"]],
        },
        "targets": [{
            "item_id": target_item,
            "popularity_before": int(popularity[target_item]),
            "interaction_count": (rec_cache["counts"].get(target_item)
                                  if rec_cache else None),
            "category": (
                "popular" if rec_cache and target_item in categories["popular"]
                else "ordinary" if rec_cache and target_item in categories["ordinary"]
                else "cold" if rec_cache and target_item in categories["cold"]
                else None
            ),
            "fake_users": len(profiles),
            "distinct_templates": len(template_counts),
        }],
        "train_pairs_before": before_cnt,
        "train_pairs_after": after_cnt,
        "injected_pairs": injected,
    }

    out = out_dir or Path(str(DEFAULT_OUT_DIR).format(
        dataset=dataset, model=model_name)) / tag
    save_meta(poisoned, out / "meta.pkl")
    save_json(profiles, out / "profiles.json")
    save_json(stats, out / "stats.json")
    save_config_snapshot(config, out)
    write_latest_pointer(out.parent, tag)

    print(f"[uba] 数据集: {dataset}（{num_users} 用户 / {num_items} 物品），"
          f"run_tag: {tag}")
    print(f"[uba] 目标物品: {target_item}（流行度 {popularity[target_item]}，"
          f"分类 {stats['targets'][0]['category']}）")
    print(f"[uba] 目标用户: {len(target_users)} 个"
          f"（策略 {tu_info['strategy']}，候选 {tu_info['n_candidates']}，"
          f"类别交互上界 {stats['target_users']['max_category_interactions']}）")
    print(f"[uba] 处理效应: method={stats['treatment']['method']}，"
          f"H={max_per_user}，hit_k={stats['treatment']['hit_k']}")
    print(f"[uba] 分配策略 {allocation['strategy']}：预算 {budget} → 实投 "
          f"{allocation['num_fake_users']}，直方图 {stats['allocation']['histogram']}，"
          f"估计收益 {allocation['estimated_value']:.4f}")
    print(f"[uba] 假用户画像: {filler_source} filler × {filler_size} + 1 目标物品")
    print(f"[uba] 注入前训练交互: {before_cnt} → 注入后: {after_cnt}（+{injected}）")
    print(f"[uba] 输出 → {out / 'meta.pkl'}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UBA 数据注入")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "uba" / "config.yaml"))
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
