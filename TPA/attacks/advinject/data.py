"""Data-stage compatibility functions for AdvInject."""
from __future__ import annotations
import numpy as np
import torch
from attacks.advinject.common import canonical_config, initialize_fake_data, pairs_to_csr, resolve_target_items, save_fake_artifacts

def build_fake_tensor(meta, targets, attack_cfg, seed=42):
    attack_cfg=canonical_config(attack_cfg)
    attack=attack_cfg.get("attack",{})
    train=pairs_to_csr(meta["train_pairs"],meta["num_users"],meta["num_items"])
    n_fakes=int(attack.get("num_fake_users",attack.get("ratio",1)))
    if 0<n_fakes<1: n_fakes=max(1,int(meta["num_users"]*n_fakes))
    fake=initialize_fake_data(train,n_fakes,seed).toarray(); fake[:,targets]=1.0; return torch.tensor(fake,dtype=torch.float32)

def poisoned_meta(meta, fake_binary):
    fake=np.asarray(fake_binary); pairs=[(int(meta["num_users"]+u),int(i)) for u,row in enumerate(fake) for i in np.flatnonzero(row)]
    result=dict(meta); result["num_users"]=int(meta["num_users"]+len(fake)); result["train_pairs"]=list(meta["train_pairs"])+pairs; result["user_items"]=[set() for _ in range(result["num_users"])]
    for u,i in result["train_pairs"]: result["user_items"][u].add(i)
    return result,pairs

def select_targets(meta, attack_cfg):
    return resolve_target_items(meta, attack_cfg, seed=42).tolist()
