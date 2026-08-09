"""MF 训练组装车间
用法:
  python models/mf/main.py              # 新训练
  python models/mf/main.py --resume      # 断点续训
"""
import sys
import os
import csv
import json
from typing import Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from training.framework import TrainingConfig, Callback
from training.run_tag import resolve_run_tag, save_config_snapshot, write_latest_pointer
from training.metrics import BestTracker, safe_checkpoint_name
from models.mf.dataset import MFDataLoader, KEY_NUM_USERS, KEY_NUM_ITEMS, KEY_DATASET
from models.mf.model import MatrixFactorization
from evaluation.metrics import compute_metrics


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
EVAL_LOG_PATH = os.path.join(OUTPUT_DIR, "eval_log.csv")
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pt")


class FullRankingCallback(Callback):
    """定期全量排序评估，记录到 eval_log.csv"""

    def __init__(self, loader: MFDataLoader, model: MatrixFactorization,
                 config, tag_dir: str):
        self.loader = loader
        self.model = model
        self.config = config
        self.tag_dir = tag_dir
        self.tag_checkpoint_dir = os.path.join(tag_dir, "checkpoints")
        self.tag_eval_log = os.path.join(tag_dir, "eval_log.csv")
        self.eval_every = config.get("eval_every", 10)
        self.eval_k = config.get("k", 20)
        self.tracker = BestTracker(
            config.get("metrics"),
            config.get("checkpoint_mode", "per_metric"),
        )

        self.test_pos = {}
        for u, i in loader.test_pairs:
            self.test_pos.setdefault(u, set()).add(i)
        self.train_pos = loader.user_items

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.tag_checkpoint_dir, exist_ok=True)
        if not os.path.exists(self.tag_eval_log):
            with open(self.tag_eval_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", f"recall@{self.eval_k}", f"ndcg@{self.eval_k}"])

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if epoch % self.eval_every != 0 and epoch != 1:
            return

        print(f"  [eval] epoch {epoch}: computing full ranking...")
        self.model.set_eval()
        with torch.no_grad():
            user_emb = self.model.get_user_embeddings()
            item_emb = self.model.get_item_embeddings()
            test_users = sorted(self.test_pos.keys())
            test_user_ids = torch.LongTensor(test_users).to(user_emb.device)
            scores_list = []
            for i in range(0, len(test_user_ids), 1024):
                batch = test_user_ids[i:i + 1024]
                scores_list.append((user_emb[batch] @ item_emb.T).cpu())
            scores = torch.cat(scores_list, dim=0)

        result = compute_metrics(scores, self.train_pos, self.test_pos, k=self.eval_k)
        recall_key = f"recall@{self.eval_k}"
        ndcg_key = f"ndcg@{self.eval_k}"
        with open(self.tag_eval_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, result[recall_key], result[ndcg_key]])

        print(f"  [eval] epoch {epoch}: {recall_key}={result[recall_key]:.4f}, "
              f"{ndcg_key}={result[ndcg_key]:.4f}")

        improved = self.tracker.update(result, epoch)
        for name in improved:
            ckpt_path = os.path.join(
                self.tag_checkpoint_dir,
                f"{safe_checkpoint_name(name)}-best-model.pt",
            )
            payload = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.model._optimizer.state_dict(),
                "metrics": dict(result),
            }
            payload.update(result)
            torch.save(payload, ckpt_path)
            print(f"  [ckpt] best -> {ckpt_path} ({name}={result[name]:.4f})")

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        print("\n[best]")
        for name, entry in self.tracker.best_results().items():
            print(f"  {name}={entry['value']:.4f} at epoch "
                  f"{entry['epoch']} -> {entry['checkpoint']}")


def save_checkpoint(model, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": model._optimizer.state_dict(),
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=model._device)
    model.load_state_dict(ckpt["model_state_dict"])
    model._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]


def main(tag: str | None = None, resume: bool = False):

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        config = TrainingConfig()
        config["dataset"] = "ml100k"
    else:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        flat = {}
        for section in ["data", "model", "training", "evaluation"]:
            if section in raw:
                flat.update(raw[section])
        if "run_tag" in raw:
            flat["run_tag"] = raw["run_tag"]
        config = TrainingConfig(overrides=flat)

    run_tag = resolve_run_tag(config, cli_tag=tag)
    tag_dir = os.path.join(OUTPUT_DIR, run_tag)
    tag_checkpoint_dir = os.path.join(tag_dir, "checkpoints")
    tag_latest_ckpt = os.path.join(tag_checkpoint_dir, "latest.pt")
    tag_history_path = os.path.join(tag_dir, "history.json")
    tag_eval_log_path = os.path.join(tag_dir, "eval_log.csv")
    os.makedirs(tag_checkpoint_dir, exist_ok=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(tag_history_path, "w") as f:
        json.dump({"history": [], "best": {}}, f)

    loader = MFDataLoader(config)
    params = loader.get_init_params()

    model = MatrixFactorization(config, params[KEY_NUM_USERS], params[KEY_NUM_ITEMS])
    start_epoch = 0
    if resume and os.path.exists(LATEST_CKPT):
        start_epoch = load_checkpoint(model, LATEST_CKPT)
        print(f"[train] 从 epoch {start_epoch} 恢复训练")

    full_rank = FullRankingCallback(loader, model, config, tag_dir)

    print(f"[train] dataset={config.get(KEY_DATASET)}, epochs={config.epochs}, "
          f"lr={config.lr}, batch={config.batch_size}, "
          f"resume={resume}, start_epoch={start_epoch}, run_tag={run_tag}")

    history = []
    try:
        for epoch in range(start_epoch + 1, config.epochs + 1):
            model.set_train()
            epoch_losses = []
            for batch in loader.train_loader():
                m = model.train_step(batch)
                epoch_losses.append(m["loss"])

            model.set_eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch in loader.val_loader():
                    m = model.eval_step(batch)
                    epoch_val_losses.append(m["val_loss"])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val = sum(epoch_val_losses) / len(epoch_val_losses)

            if epoch <= 2:
                grad_norm = model.embedding.weight.grad.norm().item() if model.embedding.weight.grad is not None else 0
                emb_norm = model.embedding.weight.norm(p=2).item()
                print(f"  [diag] grad_norm={grad_norm:.6f} emb_norm={emb_norm:.1f} "
                      f"lr={config.lr} wd={config.get('weight_decay', 0)}")

            print(f"[epoch {epoch}/{config.epochs}] train_loss={avg_loss:.4f} "
                  f"val_loss={avg_val:.4f}")

            history.append({
                "epoch": epoch, "train_loss": avg_loss, "val_loss": avg_val,
            })

            if epoch % config.save_every_n_epochs == 0:
                save_checkpoint(model, epoch, tag_latest_ckpt)

            full_rank.on_epoch_end(epoch, {})

            with open(tag_history_path, "w") as f:
                json.dump({
                    "history": history,
                    "best": full_rank.tracker.best_results(),
                }, f, indent=2)

    except KeyboardInterrupt:
        print("\n[train] 训练中断，保存 checkpoint...")
        save_checkpoint(model, epoch, tag_latest_ckpt)

    save_checkpoint(model, config.epochs, tag_latest_ckpt)

    # 实验归档完成：把本次实验的产物复制到稳定指针路径（供攻击流程使用）
    import shutil
    shutil.copyfile(tag_latest_ckpt, LATEST_CKPT)
    shutil.copyfile(tag_history_path, HISTORY_PATH)
    shutil.copyfile(tag_eval_log_path, EVAL_LOG_PATH)
    from pathlib import Path
    write_latest_pointer(Path(OUTPUT_DIR), run_tag)
    save_config_snapshot(config.as_dict(), Path(tag_dir))

    print(f"[train] 完成。history -> {tag_history_path}")
    print(f"[train] eval_log -> {tag_eval_log_path}")
    print(f"[train] 实验归档 -> {tag_dir}")
    print(f"[train] 稳定指针 -> {LATEST_CKPT}（run_tag={run_tag}）")


if __name__ == "__main__":
    main()
