"""Data-stage compatibility functions for AdvInject."""
from __future__ import annotations
import numpy as np
import torch
from attacks.advinject.common import canonical_config, initialize_fake_data, pairs_to_csr, resolve_target_items, save_fake_artifacts

def build_fake_tensor(meta, targets, attack_cfg, seed=42):
    """构造假用户交互张量：先按预算取模板行，再把目标列强制为 1。

    为什么这样做：push 攻击要求每个假用户都“点击”目标物品；模板行提供真实
    用户形态。矩阵形状：(n_fakes, n_items) → 同形状 0/1 张量。
    使用举例：build_fake_tensor(meta, [4], cfg, seed=42)。
    """
    attack_cfg=canonical_config(attack_cfg)
    attack=attack_cfg.get("attack",{})
    train=pairs_to_csr(meta["train_pairs"],meta["num_users"],meta["num_items"])
    n_fakes=int(attack.get("num_fake_users",attack.get("ratio",1)))
    if 0<n_fakes<1: n_fakes=max(1,int(meta["num_users"]*n_fakes))
    fake=initialize_fake_data(train,n_fakes,seed).toarray(); fake[:,targets]=1.0; return torch.tensor(fake,dtype=torch.float32)

def poisoned_meta(meta, fake_binary):
    """把假用户二值矩阵追加到 meta，返回 (中毒 meta, 新增 pair 列表)。

    功能：假用户 id 从 num_users 起编号，train_pairs/user_items 同步扩展；
    num_users 递增。用于 fit/evaluate 阶段按统一 meta 契约重训 victim。
    """
    fake=np.asarray(fake_binary); pairs=[(int(meta["num_users"]+u),int(i)) for u,row in enumerate(fake) for i in np.flatnonzero(row)]
    result=dict(meta); result["num_users"]=int(meta["num_users"]+len(fake)); result["train_pairs"]=list(meta["train_pairs"])+pairs; result["user_items"]=[set() for _ in range(result["num_users"])]
    for u,i in result["train_pairs"]: result["user_items"][u].add(i)
    return result,pairs

def select_targets(meta, attack_cfg):
    """解析 canonical attack.target_items 并采样目标物品（ids 优先，其次 zone）。"""
    return resolve_target_items(meta, attack_cfg, seed=42).tolist()
