"""AdvInject 共享工具：配置解析、稀疏矩阵、目标/假用户采样、结果落盘。

为什么这样做：把“配置读取/数据变换/指标计算”收敛到本模块，业务文件
（generate/evaluate/fit）只调用这里的能力，避免各文件重复实现。
功能：
- canonical_config：统一入口，保证业务代码只读 canonical 配置键；
- AttackConfig：把 canonical 配置映射为运行时参数快照；
- 稀疏/采样/指标：pairs→CSR、按交互数百分位选目标、初始化假用户、攻击指标。
参考公式：目标分层使用交互数百分位区间（head 95–100 / upper_torso 75–95 /
  lower_torso 50–75 / tail 0–50，见 sample_target_items）；攻击指标 HR/NDCG
  与平均排名见 attack_metrics（NDCG = 1/log2(rank+1)）。
使用举例：
  from attacks.advinject.common import canonical_config, load_meta
  cfg = canonical_config(yaml.safe_load(open("config.yaml")))
  meta, _ = load_meta(cfg)
"""
from __future__ import annotations
import json
import pickle
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import torch
from scipy import sparse

PROJECT_ROOT=Path(__file__).resolve().parents[2]


def canonical_config(config):
    """入口统一 canonicalize；业务代码只读 canonical 键。"""
    from training.config_utils import canonicalize_config

    return canonicalize_config(config)


@dataclass
class AttackConfig:
    """canonical 配置的运行时快照（内部字段，非新配置协议）。

    为什么这样做：上游 generate 循环需要一次解析好的参数对象，避免每个 epoch
    反复 get()。字段与 canonical 键的映射见 from_dict：n_fakes 保留混合语义
    （>1 为绝对数量，0<值≤1 为真实用户比例，generate 阶段再按 meta 换算）。
    """
    dataset: str="gowalla"
    seed: int=1
    use_cuda: bool=False
    n_fakes: float=0.01
    n_target_items: int=5
    target_item_popularity: str="head"
    adv_epochs: int=30
    unroll_steps: int=5
    adv_lr: float=1.0
    adv_momentum: float=0.95
    proj_threshold: float=0.1
    click_targets: bool=False
    surrogate_name: str="item_ae"
    surrogate_epochs: int=50
    surrogate_batch_size: int=2048
    surrogate_lr: float=1e-3
    surrogate_l2: float=1e-6
    weight_alpha: float=20.0
    output_dir: str="outputs"

    @classmethod
    def from_dict(cls, config: dict[str,Any]):
        config=canonical_config(config)
        attack=config.get("attack",{})
        surrogate=config.get("surrogate",{})
        adv=attack.get("adv",{})
        target_items = attack.get("target_items") or {}
        ids = target_items.get("ids") or []
        n_target_items = target_items.get("count", len(ids) or 5)
        surrogate_training=surrogate.get("training",{})
        device=config.get("training",{}).get("device","cpu")
        explicit_num = attack.get("num_fake_users")
        return cls(
            dataset=config.get("dataset","gowalla"), seed=int(config.get("seed",1)),
            use_cuda=str(device).startswith("cuda"),
            n_fakes=float(explicit_num if explicit_num is not None
                          else attack.get("ratio",0.01)),
            n_target_items=int(n_target_items),
            target_item_popularity=target_items.get("zone","head"),
            adv_epochs=int(adv.get("epochs",30)),
            unroll_steps=int(surrogate_training.get("unroll_steps",5)),
            adv_lr=float(adv.get("lr",1.0)),
            adv_momentum=float(adv.get("momentum",0.95)),
            proj_threshold=float(adv.get("proj_threshold",0.1)),
            click_targets=bool(adv.get("click_targets",False)),
            surrogate_name=surrogate.get("name","item_ae"),
            surrogate_epochs=int(surrogate_training.get("epochs",50)),
            surrogate_batch_size=int(surrogate_training.get("batch_size",2048)),
            surrogate_lr=float(surrogate_training.get("lr",1e-3)),
            surrogate_l2=float(surrogate_training.get("weight_decay",1e-6)),
            weight_alpha=float(surrogate_training.get("weight_alpha",20.0)),
            output_dir=config.get("output",{}).get("dir","outputs"))

def set_seed(seed, cuda=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cuda and torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def resolve_meta_path(config):
    configured=config.get("data_path") or config.get("data",{}).get("processed_data_path")
    if configured:
        path=Path(configured)
        return path/"meta.pkl" if path.is_dir() else path
    dataset=config.get("dataset", "ml100k")
    model=config.get("model",{}).get("name","wmf")
    candidates=[PROJECT_ROOT/"models"/model/"data"/"processed"/dataset/"meta.pkl"]
    candidates += [PROJECT_ROOT/"models"/name/"data"/"processed"/dataset/"meta.pkl" for name in ("wmf","mf","lightgcn")]
    for path in candidates:
        if path.exists(): return path
    raise FileNotFoundError(f"找不到处理后数据 meta.pkl: {candidates[0]}")

def load_meta(config):
    path=resolve_meta_path(config)
    with path.open("rb") as handle: meta=pickle.load(handle)
    required={"num_users","num_items","train_pairs","test_pairs"}
    missing=required-set(meta)
    if missing: raise KeyError(f"meta.pkl 缺少字段: {sorted(missing)}")
    meta=dict(meta); meta["train_pairs"]=[(int(u),int(i)) for u,i in meta["train_pairs"]]
    meta["test_pairs"]=[(int(u),int(i)) for u,i in meta["test_pairs"]]
    meta.setdefault("user_items",build_user_items(meta["train_pairs"],meta["num_users"]))
    return meta,path

def build_user_items(pairs, n_users):
    result=[set() for _ in range(int(n_users))]
    for user,item in pairs: result[int(user)].add(int(item))
    return result

def pairs_to_csr(pairs,n_users,n_items):
    if not pairs: return sparse.csr_matrix((n_users,n_items),dtype=np.float32)
    rows,cols=zip(*pairs)
    return sparse.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(n_users,n_items))

def item_counts(meta): return Counter(i for _,i in meta["train_pairs"])

def sample_target_items(meta, n_samples, popularity="head", fixed=None, seed=1):
    """按交互数百分位区间或显式 id 采样目标物品。

    为什么这样做：AdvInject 上游把物品按交互数分为 head/upper_torso/
    lower_torso/tail 四个区间，直接按 np.percentile(counts, low/high) 找候选；
    固定 id 时跳过采样。功能：返回升序目标物品数组。使用举例：
    sample_target_items(meta, 5, "head", seed=1)。
    """
    if fixed is not None: targets=np.asarray(fixed,dtype=np.int64)
    else:
        counts=np.asarray([item_counts(meta).get(i,0) for i in range(meta["num_items"])])
        # 交互数百分位区间 → (counts 低百分位, 高百分位)，如 head=(95,100)
        percentiles={"head":(95,100),"upper_torso":(75,95),"lower_torso":(50,75),"tail":(0,50)}
        if popularity not in percentiles: raise ValueError(f"unknown popularity: {popularity}")
        low,high=percentiles[popularity]; lo=np.percentile(counts,low); hi=np.percentile(counts,high)
        if popularity=="head": valid=np.flatnonzero(counts>lo)
        elif popularity=="tail": valid=np.flatnonzero(counts<hi)
        else: valid=np.flatnonzero((counts>lo)&(counts<hi))
        if len(valid)<n_samples: raise ValueError("not enough target items for popularity category")
        targets=np.random.default_rng(seed).choice(valid,size=n_samples,replace=False)
    targets=np.sort(targets.astype(np.int64))
    if len(targets)!=n_samples or np.any(targets<0) or np.any(targets>=meta["num_items"]): raise ValueError("invalid target items")
    return targets


def resolve_target_items(meta, config, seed=1):
    """从 canonical attack.target_items 解析目标物品（ids 优先，其次 zone）。"""
    config = canonical_config(config)
    ti = config.get("attack", {}).get("target_items", {})
    ids = ti.get("ids")
    count = int(ti.get("count", len(ids or []) or 5))
    zone = ti.get("zone", "head")
    return sample_target_items(meta, count, zone, ids, seed)

def initialize_fake_data(train_csr,n_fakes,seed=1):
    """从训练矩阵中抽取“模板用户”作为假用户初始交互。

    为什么这样做：用真实用户行为做冷启动，比纯随机更像正常档案（沿用上游
    revisit_adv_rec 做法）。功能：返回 (n_fakes, n_items) 稀疏矩阵，每行是
    被抽样真实用户的交互行；合格用户限定在交互数 ≤100（上游阈值）。
    使用举例：initialize_fake_data(train_csr, 50, seed=1)。
    """
    n_fakes=int(n_fakes); rng=np.random.default_rng(seed); dense=train_csr.toarray(); clicks=dense.sum(1); qualified=np.flatnonzero(clicks<=100)
    if len(qualified)<n_fakes: qualified=np.arange(dense.shape[0])
    sampled=rng.choice(qualified,size=n_fakes,replace=len(qualified)<n_fakes)
    return sparse.csr_matrix(dense[sampled],shape=(n_fakes,dense.shape[1]),dtype=np.float32)

def project_fake(fake, threshold, targets, click_targets):
    """把可微连续档案投影为 0/1 隐式交互。

    为什么这样做：上游 surrogate 输出连续“点击强度”，注入训练集需要二值；
    投影阈值 threshold 后，若 click_targets=True 再把目标物品强制置 1（保证
    push 生效）。矩阵说明：(n_fakes, n_items) → 同形状 0/1 矩阵。
    参考：upstream project_fake（revisit_adv_rec）。
    """
    result=(fake>float(threshold)).to(fake.dtype)
    if click_targets: result[:,targets]=1.0
    return result

def attack_metrics(scores, targets, train_items, cutoff=50):
    scores=np.asarray(scores).copy(); positions=[]; ndcgs=[]
    for row in range(scores.shape[0]):
        if row < len(train_items): scores[row,list(train_items[row])]=-np.inf
        order=np.argsort(-scores[row],kind="stable")
        for target in targets:
            matches=np.flatnonzero(order==int(target))
            rank=int(matches[0])+1 if matches.size else scores.shape[1]+1
            positions.append(rank); ndcgs.append(1.0/np.log2(rank+1) if rank<=cutoff else 0.0)
    if not positions:
        return {"TargetAvgRank":float("nan"),f"TargetHR@{cutoff}":0.0,f"TargetNDCG@{cutoff}":0.0}
    return {"TargetAvgRank":float(np.mean(positions)),f"TargetHR@{cutoff}":float(np.mean(np.asarray(positions)<=cutoff)),f"TargetNDCG@{cutoff}":float(np.mean(ndcgs))}
def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=float),encoding="utf-8")

def save_fake_artifacts(out_dir,fake_csr,targets,history,meta,config):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    sparse.save_npz(out/"fake_data.npz",fake_csr)
    save_json(out/"target_items.json",{"target_items":[int(i) for i in targets]})
    save_json(out/"history.json",history)
    poisoned=dict(meta); fake_pairs=[(int(meta["num_users"]+u),int(i)) for u,row in enumerate(fake_csr.toarray()) for i in np.flatnonzero(row)]
    poisoned["num_users"]=int(meta["num_users"]+fake_csr.shape[0]); poisoned["train_pairs"]=list(meta["train_pairs"])+fake_pairs; poisoned["user_items"]=build_user_items(poisoned["train_pairs"],poisoned["num_users"])
    with (out/"meta.pkl").open("wb") as handle: pickle.dump(poisoned,handle)
    save_json(out/"stats.json",{"num_fake_users":fake_csr.shape[0],"injected_pairs":len(fake_pairs),"created_at":datetime.now().isoformat(timespec="seconds")})
    return poisoned


def load_poisoned_meta(data_dir):
    path=Path(data_dir)/"meta.pkl"
    if not path.exists(): raise FileNotFoundError(f"投毒数据不存在: {path}")
    with path.open("rb") as handle: return pickle.load(handle)

