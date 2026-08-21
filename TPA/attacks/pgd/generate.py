"""PGD 攻击 —— 数据层：投影梯度上升伪造假用户画像 + 注入

论文依据：Li et al. (NIPS 2016) §3-§4.1
- 攻击模型：加入 αm 个恶意用户，每个用户最多给 B 个物品打分，评分有界 [−Λ, Λ]
  （Eq. 6 的可行集 M）
- PGA 更新：M̃^(t+1) = Proj_M(M̃^(t) + s_t · ∇_{M̃} R)（Eq. 10，Algorithm 1）
- 梯度：链式法则 ∇_{M̃} R = ∇_{M̃} Θ · ∇_Θ R（Eq. 11），其中 ∇_{M̃} Θ 由
  ALS 的 KKT 条件近似：∂v_j/∂M̃_ij ≈ (λ_V I + Σ_V^(j))^{-1} ũ_i（§4.1）
- 效用：混合效用 R = μ1·R^av + μ2·R^in（Eq. 9）：
  - R^av = ‖R_{Ω^C}(M̂ − M̄)‖_F²（availability，Eq. 7）
  - R^in = Σ_u Σ_{j∈J0} w(j)·M̂_uj（integrity push，Eq. 8）

参考实现（A 级）：fuying-wang/Data-poisoning-attacks-on-factorization-based-
collaborative-filtering 的 main.py / compute_grad.py / ALS_optimize.py。
本实现把参考的逐元素双重循环向量化为逐物品的 k×k 求解，并把算法泛化到
LightGCN（one-step 线性化响应代理，见 DESIGN.md 的 [ai] 标注）。

与 bandwagon 模板的关系：本文件只替换"假用户画像生成"一处（generate_fake_profiles），
其余 classify → data → model 流程与模板一致。
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RAW_META = PROJECT_ROOT / "models" / "lightgcn" / "data" / "processed" / "{dataset}" / "meta.pkl"
# 中毒数据按 数据集/模型/实验标签 分层隔离；{model} 与 {tag} 在 main() 中替换
DEFAULT_OUT_DIR = PROJECT_ROOT / "attacks" / "pgd" / "data" / "poisoned" / "{dataset}" / "{model}"

from training.run_tag import (
    resolve_run_tag,
    save_config_snapshot,
    write_latest_pointer,
)


def load_rec_freq_cache(config: Dict[str, Any], model_name: str, k: int,
                        required: bool = False) -> Dict[str, Any] | None:
    """读取推荐频次分类缓存（classify.py 产出）。"""
    from attacks.pgd.classify import load_cache
    return load_cache(config, model_name, k, required=required)


# ─── 数据 IO ─────────────────────────────────────────────
def load_meta(meta_path: Path) -> Dict[str, Any]:
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


# ─── 目标/热度选择（与 bandwagon 模板一致） ────────────────
def compute_item_popularity(train_pairs: List[Tuple[int, int]]) -> Counter:
    return Counter(item for _, item in train_pairs)


def select_target_items(popularity: Counter, num_items: int, strategy: str,
                        count: int, ids: List[int], rng: random.Random,
                        categories: Dict[str, List[int]] | None = None,
                        category: str = "cold",
                        rec_counts: Dict[int, int] | None = None) -> List[int]:
    """选择攻击目标物品（与 bandwagon 模板完全一致）。"""
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
                "请先运行 python attacks/pgd/run.py --mode classify"
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
            f"未知的目标选择策略 '{strategy}'，可选 "
            f"specified | category | coldest | random"
        )

    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        print(f"[generate] 目标物品去重后仅 {len(selected)} 个（要求 {count}），使用可用数量")
    for i in selected:
        if not (0 <= i < num_items):
            raise ValueError(f"目标物品 ID {i} 超出范围 [0, {num_items})")
    return selected


# ─── PGD 梯度引擎 ─────────────────────────────────────────
def load_clean_embeddings(config: Dict[str, Any], meta: Dict[str, Any],
                          model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """通过 registry 加载干净模型 checkpoint，取出 (用户嵌入 U, 物品嵌入 V)。
    这就是"不同模型的权重"——MF 为查表嵌入，LightGCN 为 LGC 传播后的最终嵌入。
    """
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import torch
    from attacks.pgd.registry import get_model_cls
    from attacks.pgd.fit import build_training_config, resolve_clean_checkpoint

    cfg = build_training_config(config, config["dataset"], model_name)
    model_cls = get_model_cls(model_name)
    edge_index = torch.LongTensor(
        [[u, i] for u, i in meta["train_pairs"]]
    ).T
    model = model_cls(cfg, meta["num_users"], meta["num_items"], edge_index)
    ckpt = torch.load(
        resolve_clean_checkpoint(config),
        map_location=model._device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_eval()
    with torch.no_grad():
        U = model.get_user_embeddings().detach().cpu().numpy().astype(np.float64)
        V = model.get_item_embeddings().detach().cpu().numpy().astype(np.float64)
    return U, V


def als_solve(U: np.ndarray, U_tilde: np.ndarray, V: np.ndarray,
              train_pairs: List[Tuple[int, int]],
              fake_support: List[List[int]],
              fake_ratings: List[Dict[int, float]],
              lambda_u: float, lambda_v: float, iterations: int) -> None:
    """在 (正常数据 ∪ 恶意数据) 上交替最小化求解 Eq.(4) 的近似解（就地更新）。
    与参考实现 ALS_optimize.py 一致，但正则采用论文 KKT 的无 n 缩放形式
    （A = Σ xx^T + λI），保证隐式梯度公式与固定点一致。
    评分语义：隐式反馈中正常交互视为 1，恶意评分由 PGD 优化（[−Λ, Λ]）。
    """
    M = U.shape[0]
    F = U_tilde.shape[0]
    N = V.shape[0]
    k = U.shape[1]
    I_k = np.eye(k)

    user_items: Dict[int, List[int]] = defaultdict(list)
    item_users: Dict[int, List[int]] = defaultdict(list)
    for u, i in train_pairs:
        user_items[u].append(i)
        item_users[i].append(u)
    item_fake_users: Dict[int, List[int]] = defaultdict(list)
    for f, sup in enumerate(fake_support):
        for j in sup:
            item_fake_users[j].append(f)

    for _ in range(iterations):
        # 正常用户更新
        for u in range(M):
            items = user_items.get(u)
            if not items:
                continue
            Vj = V[items]
            A = Vj.T @ Vj + lambda_u * I_k
            U[u] = np.linalg.solve(A, Vj.sum(axis=0))  # 隐式评分=1
        # 恶意用户更新
        for f in range(F):
            items = fake_support[f]
            Vj = V[items]
            r = np.array([fake_ratings[f][j] for j in items])
            A = Vj.T @ Vj + lambda_u * I_k
            U_tilde[f] = np.linalg.solve(A, Vj.T @ r)
        # 物品更新（正常 + 恶意联合）
        for j in range(N):
            n_users = item_users.get(j, [])
            f_users = item_fake_users.get(j, [])
            if not n_users and not f_users:
                continue
            A = lambda_v * I_k
            rhs = np.zeros(k)
            if n_users:
                Uj = U[n_users]
                A = A + Uj.T @ Uj
                rhs = rhs + Uj.sum(axis=0)  # 隐式评分=1
            if f_users:
                Utj = U_tilde[f_users]
                A = A + Utj.T @ Utj
                rhs = rhs + Utj.T @ np.array(
                    [fake_ratings[f][j] for f in f_users])
            V[j] = np.linalg.solve(A, rhs)


def neighbor_fake_embeddings(V: np.ndarray, fake_support: List[List[int]],
                             fake_ratings: List[Dict[int, float]],
                             lambda_u: float) -> np.ndarray:
    """LightGCN 代理：假用户嵌入用 ALS 式用户更新闭式解近似。
    ũ_f = (λ_u I + Σ_{j∈S_f} v_j v_j^T)^{-1} Σ_{j∈S_f} r̃_fj v_j
    其中 v_j 为干净 LightGCN 的物品最终嵌入（"模型的权重"）。
    标注 [ai]：这是对 warm-start 后新用户嵌入的一阶近似。
    """
    F = len(fake_support)
    k = V.shape[1]
    I_k = np.eye(k)
    U_tilde = np.zeros((F, k))
    for f, items in enumerate(fake_support):
        Vj = V[items]
        r = np.array([fake_ratings[f][j] for j in items])
        A = Vj.T @ Vj + lambda_u * I_k
        U_tilde[f] = np.linalg.solve(A, Vj.T @ r)
    return U_tilde


def item_response(U: np.ndarray, V: np.ndarray, U_tilde: np.ndarray,
                  train_pairs: List[Tuple[int, int]],
                  fake_support: List[List[int]],
                  fake_ratings: List[Dict[int, float]],
                  lambda_v: float, item_users: Dict[int, List[int]]) -> np.ndarray:
    """[ai] LightGCN 代理的物品侧一阶响应：
    v_j^resp = v_j + (λ_v I + Σ_{i∈N_j} u_i u_i^T + Σ_f Ũ_f Ũ_f^T)^{-1} Σ_f r̃_fj Ũ_f
    模拟投毒后物品嵌入沿恶意评分方向的线性化移动（对 ALS 即一步精确更新，
    对 LightGCN 为 warm-start 微调的一阶近似）。
    """
    N, k = V.shape
    I_k = np.eye(k)
    item_fake_users: Dict[int, List[int]] = defaultdict(list)
    for f, sup in enumerate(fake_support):
        for j in sup:
            item_fake_users[j].append(f)

    V_resp = V.copy()
    for j in range(N):
        n_users = item_users.get(j, [])
        f_users = item_fake_users.get(j, [])
        if not f_users:
            continue
        A = lambda_v * I_k
        rhs = np.zeros(k)
        if n_users:
            Uj = U[n_users]
            A = A + Uj.T @ Uj
        Utj = U_tilde[f_users]
        A = A + Utj.T @ Utj
        rhs = rhs + Utj.T @ np.array([fake_ratings[f][j] for f in f_users])
        V_resp[j] = V_resp[j] + np.linalg.solve(A, rhs)
    return V_resp


def utility_gradient(U: np.ndarray, V: np.ndarray,
                     U_clean: np.ndarray, V_clean: np.ndarray,
                     user_items: Dict[int, set],
                     targets: List[int], mu1: float, mu2: float,
                     w_target: float) -> np.ndarray:
    """攻击者效用对预测矩阵 M̂ 的梯度 ∇_Θ R（论文 Eq.7-9，Appendix A 易求项）。
    - availability: ∂R^av/∂M̂_ij = 2(M̂_ij − M̄_ij)，仅对未观测项（Ω^C）
    - integrity:    ∂R^in/∂M̂_ij = w(j)，对 j ∈ J0（指定目标）
    - 混合: μ1·av + μ2·in
    """
    M = U.shape[0]
    grad = np.zeros((M, V.shape[0]))
    if mu1 != 0.0:
        M_hat = U @ V.T
        M_bar = U_clean @ V_clean.T
        grad_av = 2.0 * (M_hat - M_bar)
        for m in range(M):
            for i in user_items.get(m, ()):
                grad_av[m, i] = 0.0
        grad += mu1 * grad_av
    if mu2 != 0.0:
        for t in targets:
            grad[:, t] += mu2 * w_target
    return grad


def pgd_gradient(U: np.ndarray, U_tilde: np.ndarray, V_ref: np.ndarray,
                 grad_R: np.ndarray, train_pairs: List[Tuple[int, int]],
                 fake_support: List[List[int]],
                 fake_ratings: List[Dict[int, float]],
                 lambda_u: float, lambda_v: float,
                 use_cross: bool = True) -> Tuple[np.ndarray, List[int]]:
    """计算 ∇_{M̃} R（论文 Eq.11 + §4.1 KKT 近似 + 一阶交叉项）：

    基础项（论文）：∇_{r̃_fj} R ⊇ (Σ_m ∂R/∂M̂_mj · u_m)^T A_j^{-1} ũ_f
    其中 A_j = λ_V I + Σ_{i∈N_j} u_i u_i^T + Σ_{f∈F_j} ũ_f ũ_f^T。

    交叉项 [ai 扩展]：物品嵌入对假用户评分的完整一阶链式路径
    ∂v_j/∂r̃_fj' = A_j^{-1} r̃_fj A_f^{-1} v_ref_j'（经 ũ_f 传播，j' ≠ j）。
    记 h_j = A_j^{-1} Σ_m ∂R_mj u_m，则
    ∇_{r̃_fj'} R ⊇ (Σ_{j∈S_f} r̃_fj h_j)^T A_f^{-1} v_ref_j'
    —— 这正是"filler 如何让假用户嵌入对准真实用户、从而推动目标物品"的梯度，
    让 PGD 对 filler 的选择/评分有真实的优化信号。

    返回 (grad, union_items)：grad 为 F×len(union_items)，列对齐 union_items。
    """
    F = U_tilde.shape[0]
    k = U.shape[1]
    I_k = np.eye(k)
    union_items = sorted({j for sup in fake_support for j in sup})
    col_of = {j: c for c, j in enumerate(union_items)}
    item_users: Dict[int, List[int]] = defaultdict(list)
    for u, i in train_pairs:
        item_users[i].append(u)
    item_fake_users: Dict[int, List[int]] = defaultdict(list)
    for f, sup in enumerate(fake_support):
        for j in sup:
            item_fake_users[j].append(f)

    # 每个假用户的 A_f^{-1}（用 v_ref 构造：ALS 引擎用求解后 V，neighbor 用干净 V0）
    A_f_inv: List[np.ndarray] = []
    for f, sup in enumerate(fake_support):
        Vj = V_ref[sup]
        A_f_inv.append(np.linalg.inv(Vj.T @ Vj + lambda_u * I_k))

    H = np.zeros((len(union_items), k))  # h_j
    for col, j in enumerate(union_items):
        n_users = item_users.get(j, [])
        f_users = item_fake_users.get(j, [])
        if not f_users:
            continue
        rows = []
        if n_users:
            rows.append(U[n_users])
        rows.append(U_tilde[f_users])
        Rj = np.vstack(rows)
        A = Rj.T @ Rj + lambda_v * I_k
        if n_users:
            H[col] = np.linalg.solve(A, grad_R[n_users, j] @ U[n_users])

    grad = np.zeros((F, len(union_items)))
    for f in range(F):
        # 基础项（论文 §4.1）
        for j in fake_support[f]:
            grad[f, col_of[j]] = U_tilde[f] @ H[col_of[j]]
        # 交叉项 [ai]：filler 对准目标推送方向
        if use_cross:
            s = np.zeros(k)
            for j in fake_support[f]:
                s += fake_ratings[f][j] * H[col_of[j]]
            c_f = A_f_inv[f] @ s
            for j in fake_support[f]:
                grad[f, col_of[j]] += c_f @ V_ref[j]
    return grad, union_items


def pgd_optimize(U0: np.ndarray, V0: np.ndarray,
                 train_pairs: List[Tuple[int, int]],
                 user_items: Dict[int, set],
                 targets: List[int], num_fake: int, filler_size: int,
                 pgd_cfg: Dict[str, Any], hot_pool: List[int],
                 rng: random.Random,
                 engine: str) -> List[Tuple[int, int, List[int]]]:
    """PGD 主循环（论文 Algorithm 1）：
    初始化每个假用户的 B 个物品支撑集（固定）→ 迭代：求解代理模型 →
    计算隐式梯度 → 梯度上升更新评分 → 投影到 [−Λ, Λ]。
    返回 profiles: [(fake_id, target, [items])]，与 bandwagon 模板同构。
    """
    num_items = V0.shape[0]
    iterations = int(pgd_cfg.get("iterations", 10))
    step_size = float(pgd_cfg.get("step_size", 0.2))
    lam = float(pgd_cfg.get("lambda_rating", 1.0))
    lambda_u = float(pgd_cfg.get("lambda_u", 0.05))
    lambda_v = float(pgd_cfg.get("lambda_v", 0.05))
    als_iters = int(pgd_cfg.get("als_iterations", 10))
    mu1 = float(pgd_cfg.get("utility", {}).get("mu1", -1.0))
    mu2 = float(pgd_cfg.get("utility", {}).get("mu2", 1.0))
    w_target = float(pgd_cfg.get("utility", {}).get("w_target", 2.0))
    use_cross = bool(pgd_cfg.get("utility", {}).get("cross_target", True))
    include_target = bool(pgd_cfg.get("include_target", True))
    init_mode = pgd_cfg.get("init", "popularity")

    if filler_size <= 0:
        raise ValueError("filler_size（物品预算 B）必须为正")

    # ── 初始化支撑集与评分 ──
    pool = hot_pool if init_mode == "popularity" else list(range(num_items))
    if include_target and len(pool) < filler_size - 1:
        pool = list(range(num_items))
    fake_support: List[List[int]] = []
    fake_ratings: List[Dict[int, float]] = []
    for f in range(num_fake):
        target = targets[f % len(targets)]
        if include_target:
            cand = [i for i in pool if i != target]
            fillers = rng.sample(cand, min(filler_size - 1, len(cand)))
            sup = [target] + fillers
        else:
            cand = [i for i in pool]
            sup = rng.sample(cand, min(filler_size, len(cand)))
        fake_support.append(sup)
        ratings = {}
        for j in sup:
            if include_target and j == target:
                ratings[j] = lam
            else:
                ratings[j] = lam if rng.random() < 0.5 else -lam
        fake_ratings.append(ratings)

    item_users: Dict[int, List[int]] = defaultdict(list)
    for u, i in train_pairs:
        item_users[i].append(u)

    print(f"[pgd] engine={engine}, iterations={iterations}, step={step_size}, "
          f"Λ={lam}, λ_u={lambda_u}, λ_v={lambda_v}, "
          f"utility=(μ1={mu1}, μ2={mu2}, w={w_target}), "
          f"include_target={include_target}, B={filler_size}")

    # ── PGD 迭代 ──
    for it in range(1, iterations + 1):
        if engine == "als":
            U = U0.copy()
            U_tilde = neighbor_fake_embeddings(V0, fake_support, fake_ratings, lambda_u)
            V = V0.copy()
            als_solve(U, U_tilde, V, train_pairs, fake_support, fake_ratings,
                      lambda_u, lambda_v, als_iters)
        else:  # neighbor
            U = U0
            U_tilde = neighbor_fake_embeddings(V0, fake_support, fake_ratings, lambda_u)
            V = item_response(U0, V0, U_tilde, train_pairs, fake_support,
                              fake_ratings, lambda_v, item_users)

        grad_R = utility_gradient(U, V, U0, V0, user_items, targets,
                                  mu1, mu2, w_target)
        # V_ref：ALS 引擎用求解后的 V（固定点），neighbor 用干净 V0（响应模型基准）
        V_ref = V if engine == "als" else V0
        grad, union_items = pgd_gradient(
            U, U_tilde, V_ref, grad_R, train_pairs, fake_support,
            fake_ratings, lambda_u, lambda_v, use_cross=use_cross)
        col_of = {j: c for c, j in enumerate(union_items)}

        max_abs = 0.0
        for f in range(num_fake):
            for j in fake_support[f]:
                r_new = fake_ratings[f][j] + step_size * grad[f, col_of[j]]
                r_new = max(-lam, min(lam, r_new))
                fake_ratings[f][j] = r_new
                max_abs = max(max_abs, abs(grad[f, col_of[j]]))
        print(f"  [pgd iter {it}/{iterations}] max|grad|={max_abs:.4f}, "
              f"target ratings={[round(fake_ratings[f][targets[f % len(targets)]], 3) for f in range(min(3, num_fake))]}")

    # ── 生成最终画像：target + 评分最高的 (B−1) 个 filler（隐式反馈只保留正向交互）──
    profiles: List[Tuple[int, int, List[int]]] = []
    for f in range(num_fake):
        target = targets[f % len(targets)]
        sup_sorted = sorted(fake_support[f],
                            key=lambda j: fake_ratings[f][j], reverse=True)
        if include_target:
            items = [target] + [j for j in sup_sorted if j != target][:filler_size - 1]
        else:
            items = sup_sorted[:filler_size]
        profiles.append((f, target, items))
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


# ─── 主流程 ───────────────────────────────────────────────
from training.timing import timed


@timed("数据注入")
def main(config: Dict[str, Any], raw_meta: Path | None = None,
         out_dir: Path | None = None) -> Dict[str, Any]:
    """执行 PGD 画像生成 + 注入，产出 poisoned meta.pkl / profiles.json / stats.json。"""
    dataset = config["dataset"]
    attack_cfg = config["attack"]
    seed = config.get("seed", 42)
    rng = random.Random(seed)
    model_name = config.get("model", {}).get("name", "lightgcn")
    tag = resolve_run_tag(config)
    k = config.get("training", {}).get("k") or 20

    meta_path = raw_meta or Path(str(DEFAULT_RAW_META).format(dataset=dataset))
    meta = load_meta(meta_path)

    num_users, num_items = meta["num_users"], meta["num_items"]
    popularity = compute_item_popularity(meta["train_pairs"])
    rec_cache = load_rec_freq_cache(config, model_name, k, required=False)
    categories = rec_cache["categories"] if rec_cache else None

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

    # filler 池：优先模型推荐频次 Top 20% 流行物品（无缓存时回退训练集热门）
    if rec_cache is not None:
        hot_pool = categories["popular"][:]
        print(f"[generate] filler 池 = 模型推荐频次流行物品 "
              f"({len(hot_pool)} 个，来自 classify 缓存)")
    else:
        hot_pool = [i for i, _ in popularity.most_common(
            attack_cfg.get("filler_size", 20) * 5)]
        print(f"[generate] [!] 无 classify 缓存，filler 池回退为训练集热门物品 "
              f"Top-{len(hot_pool)}；建议先运行 --mode classify")

    # PGD 引擎选择：auto → mf 用 als（论文精确），其余（lightgcn）用 neighbor 代理
    pgd_cfg = attack_cfg.get("pgd", {})
    engine_cfg = pgd_cfg.get("engine", "auto")
    if engine_cfg == "auto":
        engine = "als" if model_name == "mf" else "neighbor"
    else:
        engine = engine_cfg
    if engine not in ("als", "neighbor"):
        raise ValueError(f"未知 PGD 引擎 '{engine}'，可选 als | neighbor | auto")

    U0, V0 = load_clean_embeddings(config, meta, model_name)
    print(f"[pgd] 干净模型权重: {model_name} "
          f"(U {U0.shape}, V {V0.shape})")

    profiles = pgd_optimize(
        U0, V0, meta["train_pairs"], meta["user_items"], targets, num_fake,
        attack_cfg.get("filler_size", 20), pgd_cfg, hot_pool, rng, engine,
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
        "attack": "pgd",
        "model": model_name,
        "run_tag": tag,
        "engine": engine,
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
        "pgd": {
            "iterations": pgd_cfg.get("iterations", 10),
            "step_size": pgd_cfg.get("step_size", 0.2),
            "lambda_rating": pgd_cfg.get("lambda_rating", 1.0),
            "lambda_u": pgd_cfg.get("lambda_u", 0.05),
            "lambda_v": pgd_cfg.get("lambda_v", 0.05),
            "utility": pgd_cfg.get("utility", {}),
            "include_target": pgd_cfg.get("include_target", True),
            "init": pgd_cfg.get("init", "popularity"),
        },
    }

    out = out_dir or Path(str(DEFAULT_OUT_DIR).format(
        dataset=dataset, model=model_name)) / tag
    save_meta(poisoned, out / "meta.pkl")
    save_json(
        [{"fake_user": u, "target": t, "items": items} for u, t, items in profiles],
        out / "profiles.json",
    )
    save_json(stats, out / "stats.json")
    save_config_snapshot(config, out)
    write_latest_pointer(out.parent, tag)

    print(f"[pgd] 数据集: {dataset}（{num_users} 用户 / {num_items} 物品）")
    print(f"[pgd] run_tag: {tag}")
    print(f"[pgd] 目标物品: {targets}，流行度: {[popularity[t] for t in targets]}")
    if rec_cache:
        print(f"[pgd] 目标物品分类: "
              f"{[stats['targets'][j]['category'] for j in range(len(targets))]}")
    print(f"[pgd] 假用户: {len(profiles)}，每个交互 "
          f"{attack_cfg.get('filler_size', 20)} 个物品（含目标）")
    print(f"[pgd] 注入前训练交互 {before_cnt} → 注入后 {after_cnt}（+{added}）")
    for t in targets:
        print(f"  [target {t}] 假用户数={per_target_counts[t]}，"
              f"注入后该物品交互数= {popularity[t] + per_target_counts[t]}")
    print(f"[pgd] 输出 -> {out / 'meta.pkl'}")
    return stats


def load_yaml_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGD 画像生成 + 注入")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "pgd" / "config.yaml"))
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
