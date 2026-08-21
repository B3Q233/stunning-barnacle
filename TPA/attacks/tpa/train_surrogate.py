"""TPA —— 代理模型训练脚本（黑盒迁移的前置步骤）

攻击者视角：只拥有自己训练的代理模型，路径构造 / 频次分类 / 后续 PGD 优化
都只依赖代理模型，受害模型仅参与最终评估。

本脚本用与受害模型不同的训练/验证划分种子（config.seed + split_seed_offset）
在干净数据上训练一个代理模型，checkpoint 保存到 surrogate.checkpoint。

用法:
  python attacks/tpa/train_surrogate.py --config attacks/tpa/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lightgcn.dataset import LightGCNDataset
from evaluation.metrics import compute_metrics

from attacks.tpa.registry import active_model_name, get_dataset_cls, get_model_cls
from attacks.tpa.fit import build_training_config
from attacks.tpa.evaluate import ranking_scores
from attacks.tpa.generate import load_meta, load_yaml_config, raw_meta_path


def _split_train_val(pairs: List[Tuple[int, int]], seed: int
                     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """95/5 随机划分，与受害模型同一划分协议但使用不同种子。"""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.95)
    return shuffled[:split], shuffled[split:]


from training.timing import timed


@timed("代理模型训练")
def main(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = config["dataset"]
    if not config.get("surrogate", {}).get("enabled", False):
        raise ValueError("train_surrogate 需要 surrogate.enabled=true，"
                         "请先在 config.yaml 中启用代理模型")

    sur = config["surrogate"]
    model_name = active_model_name(config)
    sur_tr = sur.get("training", {})
    epochs = sur_tr.get("epochs", 30)
    eval_every = sur_tr.get("eval_every", 5)
    split_offset = sur_tr.get("split_seed_offset", 1)
    split_seed = config.get("seed", 42) + split_offset
    ckpt_out = Path(sur["checkpoint"])
    if not ckpt_out.is_absolute():
        ckpt_out = PROJECT_ROOT / ckpt_out

    meta = load_meta(raw_meta_path(config))
    num_users, num_items = meta["num_users"], meta["num_items"]
    user_items = meta["user_items"]
    cfg = build_training_config(config, dataset, model_name)
    k = cfg.get("k", 10)
    neg_ratio = cfg.get("neg_ratio", 1)

    model_cls = get_model_cls(model_name)
    dataset_cls = get_dataset_cls(model_name) or LightGCNDataset

    train_pairs, val_pairs = _split_train_val(meta["train_pairs"], split_seed)
    edge_index = torch.LongTensor([[u, i] for u, i in train_pairs]).T
    model = model_cls(cfg, num_users, num_items, edge_index)

    train_ds = dataset_cls(train_pairs, num_items, user_items, num_users,
                           mode="train", neg_ratio=neg_ratio)
    val_ds = dataset_cls(val_pairs, num_items, user_items, num_users,
                         mode="train", neg_ratio=neg_ratio)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    test_pos: Dict[int, set] = {}
    for u, i in meta["test_pairs"]:
        test_pos.setdefault(u, set()).add(i)

    best_recall = -1.0
    best_ndcg = 0.0
    best_epoch = 0
    print(f"[surrogate] 训练代理模型: {model_name}, split_seed={split_seed}, "
          f"train={len(train_pairs)}, val={len(val_pairs)}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        model.set_train()
        losses: List[float] = []
        for batch in train_loader:
            losses.append(model.train_step(batch)["loss"])

        model.set_eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(model.eval_step(batch)["val_loss"])
        avg_loss = sum(losses) / len(losses)
        avg_val = sum(val_losses) / len(val_losses)
        print(f"  [epoch {epoch}/{epochs}] train_loss={avg_loss:.4f} val_loss={avg_val:.4f}")

        if epoch % eval_every == 0 or epoch == 1:
            scores, _users, test_pos_local = ranking_scores(model, meta["test_pairs"])
            res = compute_metrics(scores, user_items, test_pos_local, k=k)
            print(f"    [eval] recall@{k}={res[f'recall@{k}']:.4f}, "
                  f"ndcg@{k}={res[f'ndcg@{k}']:.4f}")
            if res[f"recall@{k}"] > best_recall:
                best_recall = res[f"recall@{k}"]
                best_ndcg = res[f"ndcg@{k}"]
                best_epoch = epoch

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "recall": best_recall,
        "ndcg": best_ndcg,
    }, ckpt_out)
    meta_out = ckpt_out.with_name("surrogate_meta.json")
    meta_out.write_text(json.dumps({
        "model": model_name,
        "dataset": dataset,
        "split_seed": split_seed,
        "epochs": epochs,
        "best_recall": best_recall,
        "best_ndcg": best_ndcg,
        "best_epoch": best_epoch,
        "checkpoint": str(ckpt_out),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[surrogate] 代理模型保存 → {ckpt_out} "
          f"(best recall@{k}={best_recall:.4f} @epoch {best_epoch})")
    return {"checkpoint": str(ckpt_out), "best_recall": best_recall,
            "best_ndcg": best_ndcg, "best_epoch": best_epoch}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPA 代理模型训练")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "attacks" / "tpa" / "config.yaml"),
    )
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)))
