"""WMF 训练组装车间（步骤⑤）

ALS 每轮交替最小二乘 = 一个 epoch；训练循环 + 全量排序评估回调
（rank̄ + recall@K / ndcg@K），产物写入 outputs/{run_tag}/。
"""
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import yaml

from evaluation.attack_eval import build_attack_eval_metrics, ranking_scores
from evaluation.metrics import compute_metrics, expected_percentile_rank
from models.wmf.config_keys import (
    KEY_ALPHA,
    KEY_CHECKPOINT_MODE,
    KEY_CONFIDENCE_SCHEME,
    KEY_DATASET,
    KEY_EPOCHS,
    KEY_EPSILON,
    KEY_EVAL_EVERY,
    KEY_K,
    KEY_METRICS,
    KEY_SAVE_EVERY_N_EPOCHS,
)
from models.wmf.dataset import (
    KEY_NUM_ITEMS,
    KEY_NUM_USERS,
    WMFDataLoader,
    WMFDataset,
)
from models.wmf.model import WMFModel
from training.framework import TrainingConfig
from training.metrics import (
    BestTracker,
    eval_ks_from_metrics,
    match_metric_values,
    safe_checkpoint_name,
)
from training.run_tag import (
    resolve_run_tag,
    save_config_snapshot,
    write_latest_pointer,
)
from training.config_utils import apply_k
from training.timing import section_enter, section_exit


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
EVAL_LOG_PATH = os.path.join(OUTPUT_DIR, "eval_log.csv")
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pt")


def _split_train_val(pairs: List[Tuple[int, int]], seed: int = 42
                     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """95/5 随机划分，与攻击模板 fit 的 BPR 路径口径一致。"""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.95)
    return shuffled[:split], shuffled[split:]


def train_wmf_from_meta(cfg: TrainingConfig, meta: Dict[str, Any],
                        out_dir, model_cls, metrics_cfg,
                        checkpoint_mode: str = "per_metric",
                        targets: List[int] | None = None,
                        clean_user_items: Dict[int, set] | None = None,
                        warm_start: bool = False,
                        ) -> Tuple[Any, List[Dict[str, Any]]]:
    """在任意 meta（干净或中毒）上训练 WMF（ALS），返回 (model, history)。

    中毒数据对 ALS 而言就是普通数据：把注入假用户后的 train_pairs 直接
    构造成全量训练矩阵，每个 epoch 一次完整 sweep（论文 Eq.4/5），
    不需要 BPR 负采样。评估复用攻击模板的全量排序指标（recall@K / ndcg@K
    / 目标 HR@K / NDCG@K），产出一致的 checkpoint 与 history.json。
    WMF 无 embedding 属性，warm_start=True 时打印警告并回退随机初始化。
    """
    num_users = meta["num_users"]
    num_items = meta["num_items"]
    user_items = meta["user_items"]
    train_pairs, val_pairs = _split_train_val(meta["train_pairs"])
    alpha = float(cfg.get(KEY_ALPHA, 40.0))
    epsilon = float(cfg.get(KEY_EPSILON, 1e-8))
    scheme = cfg.get(KEY_CONFIDENCE_SCHEME, "minimal")
    train_ds = WMFDataset(train_pairs, alpha=alpha, epsilon=epsilon,
                          scheme=scheme)
    val_ds = WMFDataset(val_pairs, alpha=alpha, epsilon=epsilon,
                        scheme=scheme)
    train_batch = (train_ds.users, train_ds.items, train_ds.conf,
                   train_ds.p, train_ds.user_obs, train_ds.item_obs)
    val_batch = (val_ds.users, val_ds.items, val_ds.conf, val_ds.p)

    edge_index = torch.LongTensor(
        [[u, i] for u, i in meta["train_pairs"]]).T
    model = model_cls(cfg, num_users, num_items, edge_index)
    if warm_start:
        print("[fit] [!] WMF 无 embedding 属性，warm-start 已跳过"
              "（随机初始化训练）")

    k = int(cfg.get(KEY_K, 20))
    eval_every = int(cfg.get(KEY_EVAL_EVERY, 5))
    epochs = int(cfg.get(KEY_EPOCHS, 15))
    tracker = BestTracker(metrics_cfg, checkpoint_mode)

    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    print(f"[fit] WMF(ALS) 训练: users={num_users}, items={num_items}, "
          f"train={len(train_pairs)}, val={len(val_pairs)}, epochs={epochs}")

    for epoch in range(1, epochs + 1):
        _t_epoch = section_enter(f"Epoch {epoch}/{epochs}")
        model.set_train()
        m = model.train_step(train_batch)
        model.set_eval()
        with torch.no_grad():
            v = model.eval_step(val_batch)
        entry = {"epoch": epoch, "train_loss": m["loss"],
                 "val_loss": v["val_loss"]}
        print(f"  [epoch {epoch}/{epochs}] train_loss={m['loss']:.4f} "
              f"val_loss={v['val_loss']:.4f}")

        if epoch % eval_every == 0 or epoch == 1:
            scores, users, test_pos_local = ranking_scores(
                model, meta["test_pairs"])
            ks = eval_ks_from_metrics(metrics_cfg, k)
            res, target_details = build_attack_eval_metrics(
                scores, users, user_items, test_pos_local,
                clean_user_items or {}, targets or [], ks,
                list(tracker.directions), device=model._device,
            )
            entry.update(res)
            if target_details:
                entry["targets"] = target_details
            eval_str = ", ".join(f"{n}={res[n]:.4f}" for n in res)
            print(f"    [eval] {eval_str}")
            improved = tracker.update(res, epoch)
            for name in improved:
                ckpt_path = ckpt_dir / \
                    f"{safe_checkpoint_name(name)}-best-model.pt"
                payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": dict(res),
                }
                payload.update(res)
                torch.save(payload, ckpt_path)
                print(f"    [ckpt] best → {ckpt_path} "
                      f"({name}={res[name]:.4f})")

        section_exit(f"Epoch {epoch}/{epochs}", _t_epoch)
        history.append(entry)

    torch.save({"epoch": epochs, "model_state_dict": model.state_dict()},
               ckpt_dir / "latest.pt")
    (Path(out_dir) / "history.json").write_text(
        json.dumps({"history": history, "best": tracker.best_results()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fit] 训练完成 → {out_dir}")
    return model, history


class FullRankingCallback:
    """定期全量排序评估：论文主指标 rank̄ + recall@K / ndcg@K。

    全量评分走独立的 predict_full_ranking（不经 eval_step），
    满足框架的 eval_step 标量契约。
    """

    def __init__(self, loader: WMFDataLoader, model: WMFModel,
                 config: TrainingConfig, tag_dir: str):
        self.loader = loader
        self.model = model
        self.config = config
        self.tag_dir = tag_dir
        self.tag_checkpoint_dir = os.path.join(tag_dir, "checkpoints")
        self.tag_eval_log = os.path.join(tag_dir, "eval_log.csv")
        self.eval_every = int(config.get(KEY_EVAL_EVERY, 1))
        self.eval_k = int(config.get(KEY_K, 20))
        self.tracker = BestTracker(
            config.get(KEY_METRICS),
            config.get(KEY_CHECKPOINT_MODE, "per_metric"),
        )
        self.metric_names = list(self.tracker.directions)

        self.all_user_ids = torch.arange(loader.num_users)
        os.makedirs(self.tag_checkpoint_dir, exist_ok=True)
        if not os.path.exists(self.tag_eval_log):
            with open(self.tag_eval_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch"] + self.metric_names)

    def on_epoch_end(self, epoch: int) -> Optional[Dict[str, float]]:
        if epoch % self.eval_every != 0 and epoch != 1:
            return None

        print(f"  [eval] epoch {epoch}: computing full ranking...")
        self.model.set_eval()
        with torch.no_grad():
            scores = self.model.predict_full_ranking(self.all_user_ids)

        ks = eval_ks_from_metrics(self.config.get(KEY_METRICS), self.eval_k)
        res_by_k = {
            K: compute_metrics(scores, self.loader.user_items,
                               self.loader.test_user_items, k=K)
            for K in ks
        }
        result = match_metric_values(self.metric_names, res_by_k)
        result["rank"] = expected_percentile_rank(
            scores, self.loader.user_items, self.loader.test_user_items
        )

        with open(self.tag_eval_log, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch] + [result.get(n, float("nan"))
                           for n in self.metric_names])

        eval_str = ", ".join(
            f"{n}={result.get(n, float('nan')):.4f}" for n in self.metric_names)
        print(f"  [eval] epoch {epoch}: {eval_str}")

        improved = self.tracker.update(result, epoch)
        for name in improved:
            ckpt_path = os.path.join(
                self.tag_checkpoint_dir,
                f"{safe_checkpoint_name(name)}-best-model.pt",
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": dict(result),
            }, ckpt_path)
            print(f"  [ckpt] best -> {ckpt_path} "
                  f"({name}={result[name]:.4f})")
        return result

    def on_train_end(self):
        print("\n[best]")
        for name, entry in self.tracker.best_results().items():
            print(f"  {name}={entry['value']:.4f} at epoch "
                  f"{entry['epoch']} -> {entry['checkpoint']}")


def save_checkpoint(model, epoch, path, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }, path)
    print(f"[ckpt] epoch {epoch} -> {path}")


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=model._device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt["epoch"]


def main(tag: Optional[str] = None, resume: bool = False,
         config_path: Optional[str] = None,
         epochs_override: Optional[int] = None):
    config_path = config_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in ["data", "model", "training", "evaluation"]:
        if section in raw:
            flat.update(raw[section])
    if "run_tag" in raw:
        flat["run_tag"] = raw["run_tag"]
    if epochs_override is not None:
        flat[KEY_EPOCHS] = epochs_override
    flat = apply_k(flat)
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

    loader = WMFDataLoader(config)
    params = loader.get_init_params()
    model = WMFModel(config, params[KEY_NUM_USERS], params[KEY_NUM_ITEMS])

    start_epoch = 0
    if resume and os.path.exists(LATEST_CKPT):
        start_epoch = load_checkpoint(model, LATEST_CKPT)
        print(f"[train] 从 epoch {start_epoch} 恢复训练")

    full_rank = FullRankingCallback(loader, model, config, tag_dir)
    epochs = int(config.get(KEY_EPOCHS, 15))
    print(f"[train] dataset={config.get(KEY_DATASET)}, epochs={epochs}, "
          f"resume={resume}, start_epoch={start_epoch}, run_tag={run_tag}")

    history = []
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            model.set_train()
            train_batch = next(iter(loader.train_loader()))
            train_metrics = model.train_step(train_batch)

            model.set_eval()
            with torch.no_grad():
                val_batch = next(iter(loader.val_loader()))
                val_metrics = model.eval_step(val_batch)

            print(f"[epoch {epoch}/{epochs}] "
                  f"train_loss={train_metrics['loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f}")

            entry = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["val_loss"],
            }
            history.append(entry)

            if epoch % int(config.get(KEY_SAVE_EVERY_N_EPOCHS, 5)) == 0:
                save_checkpoint(model, epoch, tag_latest_ckpt)

            eval_result = full_rank.on_epoch_end(epoch)
            if eval_result:
                entry.update(eval_result)

            with open(tag_history_path, "w") as f:
                json.dump({
                    "history": history,
                    "best": full_rank.tracker.best_results(),
                }, f, indent=2)

    except KeyboardInterrupt:
        print("\n[train] 训练中断，保存 checkpoint...")
        save_checkpoint(model, epoch, tag_latest_ckpt)

    save_checkpoint(model, epochs, tag_latest_ckpt)

    # 实验归档：复制到稳定指针路径（供攻击流程引用）
    shutil.copyfile(tag_latest_ckpt, LATEST_CKPT)
    shutil.copyfile(tag_history_path, HISTORY_PATH)
    shutil.copyfile(tag_eval_log_path, EVAL_LOG_PATH)
    write_latest_pointer(Path(OUTPUT_DIR), run_tag)
    save_config_snapshot(config.as_dict(), Path(tag_dir))
    full_rank.on_train_end()

    from models.wmf.report import write_report
    write_report(OUTPUT_DIR, HISTORY_PATH, EVAL_LOG_PATH,
                 full_rank.tracker.best_results(), tag=run_tag,
                 model=model, loader=loader)

    print(f"[train] 完成。history -> {tag_history_path}")
    print(f"[train] eval_log -> {tag_eval_log_path}")
    print(f"[train] 实验归档 -> {tag_dir}")
    print(f"[train] 稳定指针 -> {LATEST_CKPT}（run_tag={run_tag}）")


if __name__ == "__main__":
    main()
