"""Shared training helpers for victim models."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.revisit_common import PairDataset, build_user_items, compute_ranking_metrics

def sample_negative_items(users, user_items, num_items, generator=None):
    negatives=[]
    for user in users.detach().cpu().tolist():
        positives=user_items[int(user)]
        if len(positives) >= num_items:
            raise ValueError(f"user {user} has no negative item")
        candidate=int(torch.randint(num_items,(1,),generator=generator).item())
        while candidate in positives:
            candidate=int(torch.randint(num_items,(1,),generator=generator).item())
        negatives.append(candidate)
    return torch.tensor(negatives,dtype=torch.long,device=users.device)

def pair_loader(pairs: Iterable[tuple[int,int]], batch_size: int, shuffle=True):
    return DataLoader(PairDataset(list(pairs)), batch_size=batch_size, shuffle=shuffle, drop_last=False)

def ranking_report(model, meta, k, device):
    users=sorted({u for u,_ in meta["test_pairs"]})
    test_items=build_user_items(meta["test_pairs"],meta["num_users"])
    if not users:
        return {f"recall@{k}":0.0,f"ndcg@{k}":0.0}
    with torch.no_grad():
        scores=model.predict_full_ranking(torch.tensor(users,dtype=torch.long,device=device)).detach().cpu().numpy()
    return compute_ranking_metrics(scores,[test_items[u] for u in users],int(k))

def train_bpr_epoch(model, loader, meta, num_items, optimizer, device, generator=None):
    model.train(); total=0.0; count=0
    user_items=meta.get("user_items") or build_user_items(meta["train_pairs"],meta["num_users"])
    for users, positives in loader:
        users, positives=users.to(device), positives.to(device)
        negatives=sample_negative_items(users,user_items,num_items,generator)
        optimizer.zero_grad(set_to_none=True)
        loss=(model.bpr_loss(users,positives,negatives) if hasattr(model,"bpr_loss") else model.margin_loss(users,positives,negatives))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {loss.item()}")
        loss.backward(); optimizer.step()
        total += float(loss.item())*len(users); count += len(users)
    return total/max(1,count)

def prepare_experiment_dir(model_name: str, config: dict) -> Path:
    from training.run_tag import resolve_run_tag, save_config_snapshot
    dataset=config.get("dataset") or config.get("data",{}).get("dataset","unknown")
    tag=resolve_run_tag(config)
    out=Path(__file__).resolve().parent/model_name/"outputs"/dataset/tag
    save_config_snapshot(config,out)
    return out

def save_training_artifacts(out_dir, model, history, metrics=None):
    import json
    out_dir=Path(out_dir); (out_dir/"checkpoints").mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out_dir/"checkpoints"/"last.pt")
    (out_dir/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
    if metrics is not None:
        (out_dir/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")

