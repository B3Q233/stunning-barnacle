"""Data-stage compatibility functions for AdvInject."""
from __future__ import annotations
import numpy as np
import torch
from attacks.advinject.common import load_meta, sample_target_items, initialize_fake_data, pairs_to_csr, save_fake_artifacts

def build_fake_tensor(meta, targets, attack_cfg, seed=42):
    train=pairs_to_csr(meta["train_pairs"],meta["num_users"],meta["num_items"])
    n_fakes=int(attack_cfg.get("num_fake_users",attack_cfg.get("n_fakes",1)))
    if 0<n_fakes<1: n_fakes=max(1,int(meta["num_users"]*n_fakes))
    fake=initialize_fake_data(train,n_fakes,seed).toarray(); fake[:,targets]=1.0; return torch.tensor(fake,dtype=torch.float32)

def poisoned_meta(meta, fake_binary):
    fake=np.asarray(fake_binary); pairs=[(int(meta["num_users"]+u),int(i)) for u,row in enumerate(fake) for i in np.flatnonzero(row)]
    result=dict(meta); result["num_users"]=int(meta["num_users"]+len(fake)); result["train_pairs"]=list(meta["train_pairs"])+pairs; result["user_items"]=[set() for _ in range(result["num_users"])]
    for u,i in result["train_pairs"]: result["user_items"][u].add(i)
    return result,pairs

def select_targets(meta, attack_cfg):
    explicit=attack_cfg.get("target_items")
    count=int(attack_cfg.get("n_target_items",len(explicit or []) or 1))
    return sample_target_items(meta,count,attack_cfg.get("target_item_popularity",attack_cfg.get("target_category","head")),explicit,42).tolist()
