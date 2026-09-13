"""UBA —— uplift 纯算法层（不 import torch / 任何模型代码）。

本文件只做三件事，全部是可独立单测的纯函数：

1. **目标用户选择**（论文 §5）：与目标物品"同类别"、类别交互数 < 10、且未交互过
   目标物品的用户，从中取 `count` 个。仓库数据没有 genre 元数据，用"共现邻域"作为
   类别代理（[ai] 推断，依据与取舍见 docs/DESIGN.md）。
2. **处理效应估计（w/o S_φ 支路）**：A'³ 三跳路径计数（论文 Proposition 1/2）。
3. **预算分配**（论文 Algorithm 1）：分组背包 DP + 三种分配策略。

矩阵记号（与理解文档 2.3 一致）：

- D_r ∈ {0,1}^{M×N}：真实用户-物品二元交互（行 = 用户，列 = 物品）
- D_f ∈ {0,1}^{K×N}：假用户-物品二元交互
- D' = [D_r; D_f] ∈ {0,1}^{(M+K)×N}
- A' = [[0, D'], [D'ᵀ, 0]] ∈ {0,1}^{(M+K+N)×(M+K+N)}：对称交互图邻接矩阵

三跳路径计数的等价化简（避免构造 A'³ 全矩阵，大矩阵会爆内存）：

    (A'³)_{u,i} = Σ_{u'} ⟨D'_u, D'_{u'}⟩ · D'_{u',i}
                = [ (D'[U_t] @ D'ᵀ) @ D'[:, i] ]

矩阵变换过程（shape 与行列含义）：
    D'[U_t]      (|U_t|, N)   —— 目标用户行子集，行 = 目标用户，列 = 物品
    D'ᵀ          (N, M+K)     —— 转置，行 = 物品，列 = 全部用户（真实 + 假）
    S = D'[U_t] @ D'ᵀ  →  (|U_t|, M+K) 稀疏矩阵，S[r, u'] = 用户 r 与用户 u' 的共同交互数
    D'[:, i]     (M+K,)      —— 目标物品 i 这一列（0/1），标记"谁点击过 i"
    S @ D'[:, i] →  (|U_t|,)  —— 对"点击过 i 的中间用户"按共同交互数加权求和

物理含义：与目标用户相似（共同交互多）、且点击了目标物品 i 的中间用户的加权和，
即论文 Proposition 2 的直觉；Proposition 1 进一步说明它与预测分数的正相关性。
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse import vstack as sparse_vstack
except ImportError as exc:  # pragma: no cover - 环境缺 scipy 时给出明确提示
    raise ImportError(
        "UBA uplift 的三跳路径计数依赖 scipy.sparse（仓库 requirements.txt 已含）"
    ) from exc


VALID_TREATMENT_METHODS = ("path", "surrogate")
VALID_ALLOCATION_STRATEGIES = ("uba", "uniform_target", "random_all")
VALID_TARGET_USER_STRATEGIES = ("cooccurrence", "specified", "random")

ProfileFn = Callable[[int], Sequence[int]]


# ─────────────────────────────────────────────────────────────
# 0. 交互集合工具（所有统计都必须能过滤假用户）
# ─────────────────────────────────────────────────────────────
def interaction_sets(pairs: Sequence[Tuple[int, int]],
                     max_uid: Optional[int] = None) -> Dict[int, set]:
    """把 (user, item) 成对列表转成 {uid: {iid, ...}}。

    为什么：攻击流程会注入 uid ≥ num_users 的假用户；统计真实用户时必须能把这些
    行排除（否则假用户的交互会污染"真实用户画像/候选池"），因此统一用 max_uid
    过滤，而不是在各处各写一遍判断。

    功能：返回 {uid: 物品集合}；同一 (u, i) 重复出现只保留一次（隐式反馈语义）。
    参考：[ai] 仓库既有攻击模块（bandwagon/generate.py）的 user_items 语义。
    使用举例：interaction_sets(meta["train_pairs"], max_uid=meta["num_users"])
    """
    out: Dict[int, set] = {}
    for u, i in pairs:
        u, i = int(u), int(i)
        if max_uid is not None and u >= max_uid:
            continue
        out.setdefault(u, set()).add(i)
    return out


def item_interaction_sets(pairs: Sequence[Tuple[int, int]],
                          max_uid: Optional[int] = None,
                          num_items: Optional[int] = None
                          ) -> Dict[int, set]:
    """把 (user, item) 成对列表转成 {iid: {uid, ...}}（物品侧倒排）。

    功能：共现统计 / 类别代理只需要"物品 → 用户集合"，避免反复遍历成对列表。
    使用举例：item_interaction_sets(meta["train_pairs"], max_uid=608, num_items=6298)
    """
    out: Dict[int, set] = {}
    for u, i in pairs:
        u, i = int(u), int(i)
        if max_uid is not None and u >= max_uid:
            continue
        if num_items is not None and not (0 <= i < num_items):
            continue
        out.setdefault(i, set()).add(u)
    return out


def accessible_template_users(meta: Dict[str, Any], ratio: float = 1.0,
                              seed: int = 42) -> List[int]:
    """攻击者可访问的"模板用户池"（用于 baseline 的 random_all 策略）。

    为什么：论文 Appendix B.1 规定攻击者只能访问 20% 的用户交互；官方代码
    (`--way 1`) 从该子集随机取攻击者模板。本函数给出这一子集。
    功能：返回有训练交互的真实用户 id 列表；ratio<1 时按 seed 随机抽 ratio 比例。
    使用举例：accessible_template_users(meta, ratio=0.2, seed=42)
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError(f"accessible_ratio 必须落在 (0, 1]，实际 {ratio}")
    num_users = int(meta["num_users"])
    user_items = interaction_sets(meta["train_pairs"], max_uid=num_users)
    pool = sorted(u for u, items in user_items.items() if items)
    if ratio >= 1.0:
        return pool
    k = max(1, int(round(len(pool) * ratio)))
    return sorted(random.Random(seed).sample(pool, k))


# ─────────────────────────────────────────────────────────────
# 1. 目标用户选择（论文 §5）
# ─────────────────────────────────────────────────────────────
def cooccurrence_neighbors(user_items: Dict[int, set], target_item: int,
                           size: int) -> List[int]:
    """目标物品的"类别代理"：与它共现最多的物品集合（含目标物品自身）。

    为什么：论文按"与目标物品同类别"挑用户，但仓库 ml100k 预处理产物
    (`meta.pkl`) 只有 user/item 二元组，没有 genre/类别字段；共现邻域是与类别
    语义最接近、且仅用训练集就能算出的代理（详见 docs/DESIGN.md 的关键决策）。
    功能：统计"交互过目标物品"的用户还交互了哪些物品，按共现次数降序（并列按
    物品 id 升序）取前 size-1 个，与目标物品一起构成类别集合。
    参考：[ai] 物品共现/协同过滤邻域的标准定义；论文 §5 的"same category"。
    使用举例：cooccurrence_neighbors(user_items, target_item=251, size=20)
    """
    if size <= 0:
        raise ValueError(f"category_size 必须为正，实际 {size}")
    counts: Counter = Counter()
    for items in user_items.values():
        if target_item in items:
            for j in items:
                if j != target_item:
                    counts[j] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = [int(target_item)] + [int(j) for j, _ in ranked[: max(0, size - 1)]]
    return picked


def select_target_users(meta: Dict[str, Any], target_item: int,
                        cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """按策略选出目标用户集合 U_t（论文 §5 的目标用户选择）。

    三种策略：
    - ``cooccurrence``（默认）：候选 = 未交互目标物品、且与类别代理交互数落在
      (0, max_category_interactions) 的用户；再按 seed 随机抽 count 个
      （论文原文即"随机采样 50 个"，因此不在候选内部再按大小排序）。
    - ``specified``：用户显式给出 ids（便于复现官方代码里硬编码的 50 个用户）。
    - ``random``：未交互目标物品的用户中随机抽 count 个（对照用）。

    为什么不用"类别交互数 > threshold"：论文刻意挑**轻交互**用户（<10），他们对该
    类别尚未形成强偏好，更易被注入的画像撬动；本实现保留这一方向。
    功能：返回 {users, strategy, n_candidates, category_items, target_item}。
    参考：[paper] §5 目标用户选择；[ai] 共现类别代理。
    使用举例：select_target_users(meta, 251, config["attack"]["uba"]["target_users"], 42)
    """
    strategy = str(cfg.get("strategy", "cooccurrence"))
    if strategy not in VALID_TARGET_USER_STRATEGIES:
        raise ValueError(
            f"未知 target_users.strategy {strategy!r}，可选 {VALID_TARGET_USER_STRATEGIES}"
        )
    num_users = int(meta["num_users"])
    num_items = int(meta["num_items"])
    if not (0 <= int(target_item) < num_items):
        raise ValueError(f"目标物品 {target_item} 超出范围 [0, {num_items})")
    count = int(cfg.get("count", 50))
    if count <= 0:
        raise ValueError(f"target_users.count 必须为正，实际 {count}")

    user_items = interaction_sets(meta["train_pairs"], max_uid=num_users)
    rng = random.Random(seed)

    if strategy == "specified":
        users = [int(u) for u in cfg.get("ids", [])]
        if not users:
            raise ValueError("target_users.strategy=specified 但 ids 为空")
        for u in users:
            if not (0 <= u < num_users):
                raise ValueError(f"目标用户 id {u} 超出范围 [0, {num_users})")
            if int(target_item) in user_items.get(u, set()):
                print(f"[uplift] [!] 指定目标用户 {u} 在训练集已交互物品 "
                      f"{target_item}，评估时会被算作已见（请确认这是有意为之）")
        return {"users": users, "strategy": strategy,
                "n_candidates": len(users), "category_items": [],
                "target_item": int(target_item)}

    category_items: List[int] = []
    if strategy == "cooccurrence":
        category_items = cooccurrence_neighbors(
            user_items, int(target_item), int(cfg.get("category_size", 20)))
        max_cat = int(cfg.get("max_category_interactions", 10))
        if max_cat <= 0:
            raise ValueError(f"max_category_interactions 必须为正，实际 {max_cat}")
        category_set = set(category_items)
        candidates = [
            u for u in range(num_users)
            if int(target_item) not in user_items.get(u, set())
            and 0 < len(user_items.get(u, set()) & category_set) < max_cat
        ]
    else:  # strategy == "random"
        candidates = [
            u for u in range(num_users)
            if int(target_item) not in user_items.get(u, set())
        ]

    if not candidates:
        raise ValueError(
            f"目标用户候选为空（strategy={strategy}, target_item={target_item}）；"
            f"请放宽 max_category_interactions / category_size 或改用 specified"
        )
    picked = rng.sample(candidates, min(count, len(candidates)))
    users = sorted(picked)
    if len(users) < count:
        print(f"[uplift] [!] 目标用户候选仅 {len(users)} 个（要求 {count}），按可用数量使用")
    return {"users": users, "strategy": strategy, "n_candidates": len(candidates),
            "category_items": category_items, "target_item": int(target_item)}


# ─────────────────────────────────────────────────────────────
# 2. 处理效应（w/o S_φ）：A'³ 三跳路径计数
# ─────────────────────────────────────────────────────────────
def _clean_adjacency(user_items: Dict[int, set], num_users: int,
                     num_items: int) -> csr_matrix:
    """构造 D_r ∈ {0,1}^{num_users × num_items} 的 CSR 稀疏矩阵。"""
    rows: List[int] = []
    cols: List[int] = []
    for u, items in user_items.items():
        if not (0 <= u < num_users):
            continue
        for i in items:
            if 0 <= i < num_items:
                rows.append(int(u))
                cols.append(int(i))
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(num_users, num_items))


def _fake_adjacency(templates: Sequence[int], user_items: Dict[int, set],
                    profile_fn: ProfileFn, target_item: int,
                    num_items: int) -> csr_matrix:
    """把一批假用户画像拼成 F ∈ {0,1}^{len(templates) × num_items}。

    行序与 ``templates`` 一一对应（第 r 行 = 第 r 个假用户），因为三跳路径只看
    行对应的"模板用户"，行序错位会静默给出错误的处理效应。
    """
    rows: List[int] = []
    cols: List[int] = []
    for r, tpl in enumerate(templates):
        items = {int(i) for i in profile_fn(int(tpl))}
        items.add(int(target_item))  # 假用户必然"点击"目标物品（shilling 语义）
        for i in items:
            if 0 <= i < num_items:
                rows.append(r)
                cols.append(i)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(len(templates), num_items))


def three_hop_path_counts(d_plus: csr_matrix, target_rows: np.ndarray,
                          target_item: int) -> np.ndarray:
    """计算 (A'³)_{u,i}：只算目标用户行、只取目标物品列。

    为什么：直接算 A'³ 需要 (M+K+N)² 稠密矩阵（ml100k 已 6906²，Gowalla 量级直接
    爆内存）；三跳路径里只有目标用户 × 目标物品这一列有用，因此用
    ``(D'[U_t] @ D'ᵀ) @ D'[:, i]`` 做稀疏化简。
    功能：返回 shape (|U_t|,) 的三跳路径计数（float）。
    参考：[paper] Proposition 1/2 与 Eq.4 的 A'³ 定义；[ai] 稀疏化简推导。
    使用举例：three_hop_path_counts(Dp, np.array([3, 7]), target_item=251)
    """
    rows = d_plus[target_rows]                       # (|U_t|, N)
    column = np.asarray(d_plus[:, target_item].todense()).ravel()  # (M+K,)
    user_user = rows.dot(d_plus.T)                   # (|U_t|, M+K) 共同交互数
    counts = np.asarray(user_user.dot(column)).ravel()
    return np.maximum(counts, 0.0)


def three_hop_path_effect(user_items: Dict[int, set],
                          target_users: Sequence[int], target_item: int,
                          max_per_user: int, num_items: int,
                          num_users: Optional[int] = None,
                          alpha: float = 1.0, beta: float = 1.0,
                          profile_fn: Optional[ProfileFn] = None,
                          rng: Optional[random.Random] = None
                          ) -> np.ndarray:
    """处理效应矩阵 Y（w/o S_φ 支路）：Y[u, t] = α · ((A')³_{u,i})^β。

    为什么：论文 4.1 的第二种估计不训练代理模型，用三跳路径数作为
    "目标物品进入用户 Top-K 概率"的可计算代理（时间成本 1.1 分钟 vs 50+ 分钟）。
    功能：对 t = 0..H 逐档构造"每个目标用户等量 t 个假用户"的中毒邻接矩阵 D'，
    计算目标物品位置的三跳路径计数并按 α、β 变换，返回 (|U_t|, H+1) 矩阵。

    默认 profile_fn（未传入时）构造论文描述的"与目标用户最相似、且点击了目标物品"
    的假用户：直接复用目标用户本人的交互物品（相似度最大）+ 目标物品。
    传入 generate.py 的画像构造器时，Y 与真实注入的 D_f 完全一致。

    参考：[paper] §4.1 + Eq.4 + Appendix B.1（α=β=1.0 默认，调参 {0.5,1}/{0.3,1}）。
    使用举例：three_hop_path_effect(user_items, users, 251, 6, 6298)
    """
    if max_per_user < 0:
        raise ValueError(f"max_per_user 必须非负，实际 {max_per_user}")
    users = [int(u) for u in target_users]
    if not users:
        raise ValueError("target_users 不能为空")
    if num_users is None:
        num_users = max(user_items.keys()) + 1 if user_items else 0
    if profile_fn is None:
        def profile_fn(uid: int) -> Sequence[int]:  # type: ignore[misc]
            """默认画像：目标用户本人的历史交互（最相似的模板）+ 目标物品。"""
            return sorted(user_items.get(int(uid), set()))
    rng = rng or random.Random(0)

    clean = _clean_adjacency(user_items, int(num_users), int(num_items))
    rows = np.asarray(users, dtype=np.int64)
    effect = np.zeros((len(users), max_per_user + 1), dtype=np.float64)

    for t in range(max_per_user + 1):
        if t == 0:
            d_plus = clean
        else:
            templates = [u for u in users for _ in range(t)]
            fake = _fake_adjacency(templates, user_items, profile_fn,
                                   int(target_item), int(num_items))
            d_plus = sparse_vstack([clean, fake], format="csr")
        counts = three_hop_path_counts(d_plus, rows, int(target_item))
        # α、β 为论文 Appendix B.1 的两个估计器超参（默认 1.0）
        effect[:, t] = float(alpha) * np.power(counts, float(beta))
    return effect


# ─────────────────────────────────────────────────────────────
# 3. 预算分配（论文 Algorithm 1）
# ─────────────────────────────────────────────────────────────
def dp_allocate(values: np.ndarray, budget: int,
                max_per_user: int) -> Tuple[np.ndarray, float]:
    """分组背包 DP：求最优分配 T*（论文 Algorithm 1）。

    为什么：论文 Eq.2 的目标是在总预算 N 约束下最大化 Σ_u Y_{u,i}(t_u)；每个目标
    用户是一"组"，组内只能选一个档位 t_u ∈ {0..H}，因此是分组背包而不是普通背包。
    功能：输入 (|U_t|, H+1) 的价值矩阵与预算 N，返回 (T*, 最优值)。

    递推（与论文 Algorithm 1 等价）：
        dp[i][c] = max_{0 ≤ t ≤ min(H, c)} dp[i-1][c-t] + v[i-1][t]
        dp[0][c] = 0（允许预算用不满；Σ t_u ≤ N）
    并列时取更小的 t_u（省预算），因此 `>` 而非 `>=`。

    参考：[paper] Algorithm 1；[官方代码] DPA.py::pack5（同样的分组背包 + 回溯）。
    使用举例：T, val = dp_allocate(Y, budget=100, max_per_user=6)
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"values 必须是二维 (|U_t|, H+1)，实际 shape={v.shape}")
    n, m = v.shape
    if m != max_per_user + 1:
        raise ValueError(
            f"values 第二维 {m} 与 max_per_user+1 = {max_per_user + 1} 不一致"
        )
    if budget < 0:
        raise ValueError(f"预算必须非负，实际 {budget}")

    neg_inf = -np.inf
    dp = np.full((n + 1, budget + 1), neg_inf, dtype=np.float64)
    choice = np.zeros((n + 1, budget + 1), dtype=np.int64)
    dp[0, :] = 0.0
    for i in range(1, n + 1):
        for c in range(budget + 1):
            best, best_t = neg_inf, 0
            for t in range(0, min(max_per_user, c) + 1):
                prev = dp[i - 1, c - t]
                if prev == neg_inf:
                    continue
                cand = prev + v[i - 1, t]
                if cand > best:
                    best, best_t = cand, t
            dp[i, c] = best
            choice[i, c] = best_t

    allocation = np.zeros(n, dtype=np.int64)
    capacity = budget
    for i in range(n, 0, -1):
        t = int(choice[i, capacity])
        allocation[i - 1] = t
        capacity -= t
    return allocation, float(dp[n, budget])


def _allocation_value(values: np.ndarray, users: Sequence[int],
                      allocation: Dict[int, int]) -> float:
    """按 Y 矩阵累加给定分配方案的估计收益（用于报告，不参与求解）。"""
    index = {int(u): r for r, u in enumerate(users)}
    total = 0.0
    for u, t in allocation.items():
        r = index.get(int(u))
        if r is None:
            continue
        t = int(t)
        if 0 <= t < values.shape[1]:
            total += float(values[r, t])
    return total


def allocate(values: np.ndarray, target_users: Sequence[int], strategy: str,
             budget: int, max_per_user: int, seed: int = 42,
             template_pool: Optional[Sequence[int]] = None,
             rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """三种分配策略（复现论文对比行的关键开关）。

    - ``uba``：分组背包 DP 最优 T*（论文核心，+UBA 行）。
    - ``uniform_target``：预算在目标用户上尽量均匀 + 余量随机补齐（+Target 行）。
    - ``random_all``：从攻击者可访问的模板池随机取 budget 个模板用户（baseline，
      对应官方代码 `--way 1`：假用户由随机真实用户画像生成，不针对目标用户）。

    功能：统一返回 "模板用户序列"（template_users）—— 即"第 k 个假用户照着哪个真实
    用户的画像生成"，与官方代码 `idx = [uid] * t_u` 后 `train_data_array[idx]` 完全
    对应；下游 generate.py 只消费这一个字段，因此三种策略共用同一套注入代码。

    参考：[paper] §4.1/Eq.2（+Target 与 UBA 的定义）；[官方代码] aushplus.py::generate_fakeMatrix
    的 way 1/2/3 分支。
    使用举例：allocate(Y, users, "uba", 100, 6, seed=42, template_pool=pool)
    """
    if strategy not in VALID_ALLOCATION_STRATEGIES:
        raise ValueError(
            f"未知 allocation.strategy {strategy!r}，可选 {VALID_ALLOCATION_STRATEGIES}"
        )
    if budget < 0:
        raise ValueError(f"num_fake_users（预算 N）必须非负，实际 {budget}")
    values = np.asarray(values, dtype=np.float64)
    users = [int(u) for u in target_users]
    rng = rng or random.Random(seed)

    allocation: Dict[int, int] = {u: 0 for u in users}
    template_users: List[int] = []

    if strategy == "uba":
        if not users:
            raise ValueError("allocation.strategy=uba 需要非空目标用户")
        t_star, best_value = dp_allocate(values, budget, max_per_user)
        for u, t in zip(users, t_star):
            allocation[u] = int(t)
            template_users.extend([u] * int(t))
        estimated = best_value
    elif strategy == "uniform_target":
        if not users:
            raise ValueError("allocation.strategy=uniform_target 需要非空目标用户")
        base, remainder = divmod(budget, len(users))
        for k, u in enumerate(users):
            allocation[u] = min(max_per_user, base + (1 if k < remainder else 0))
        leftover = budget - sum(allocation.values())
        if leftover > 0:  # base+1 被 max_per_user 截断时，把余量随机补给未满用户
            order = list(users)
            rng.shuffle(order)
            for u in order:
                while leftover > 0 and allocation[u] < max_per_user:
                    allocation[u] += 1
                    leftover -= 1
        template_users = [u for u in users for _ in range(allocation[u])]
        estimated = _allocation_value(values, users, allocation)
    else:  # random_all
        pool = [int(u) for u in (template_pool or [])]
        if not pool:
            raise ValueError(
                "allocation.strategy=random_all 需要非空模板用户池"
                "（attack.uba.target_users.accessible_ratio 抽样得到）"
            )
        # 官方 way=1 用 np.random.choice(..., attack_num)（有放回），此处保持有放回
        template_users = rng.choices(pool, k=budget)
        estimated = float("nan")  # baseline 不按目标用户口径估值

    used = len(template_users)
    if used > budget:
        raise AssertionError(f"分配出的假用户数 {used} 超过预算 {budget}")
    return {
        "strategy": strategy,
        "target_users": users,
        "allocation": allocation,
        "template_users": template_users,
        "num_fake_users": used,
        "budget": int(budget),
        "max_per_user": int(max_per_user),
        "estimated_value": float(estimated),
        "unused_budget": int(budget - used),
    }
