"""LightGCN 训练组装车间
用法:
  python models/lightgcn/main.py              # 新训练
  python models/lightgcn/main.py --resume      # 断点续训
"""
import sys
import os
import csv
import pickle
import json
from typing import Dict
import torch
import numpy as np

from training.framework import (
    TrainingConfig, Experiment, Trainer,
    LoggerCallback, CheckpointCallback, MetricAccumulator, Callback
)
from models.lightgcn.dataset import LightGCNDataLoader, KEY_NUM_USERS, KEY_NUM_ITEMS, KEY_DATASET
from models.lightgcn.model import LightGCN
from evaluation.metrics import compute_metrics


# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
EVAL_LOG_PATH = os.path.join(OUTPUT_DIR, "eval_log.csv")
LATEST_CKPT = os.path.join(CHECKPOINT_DIR, "latest.pt")


class FullRankingCallback(Callback):
    """定期全量排名评估，记录到 eval_log.csv"""

    def __init__(self, loader: LightGCNDataLoader, model: LightGCN,
                 config):
        self.loader = loader
        self.model = model
        self.config = config
        self.eval_every = config.get("eval_every", 10)
        self.best_recall = 0.0
        self.best_epoch = 0

        # 测试集正样本
        self.test_pos = {}
        for u, i in loader.test_pairs:
            self.test_pos.setdefault(u, set()).add(i)
        # 训练集（过滤用）
        self.train_pos = loader.user_items

        # 创建 csv 表头
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(EVAL_LOG_PATH):
            with open(EVAL_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "recall@20", "ndcg@20"])

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

        result = compute_metrics(scores, self.train_pos, self.test_pos, k=20)
        with open(EVAL_LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, result["recall@20"], result["ndcg@20"]])

        print(f"  [eval] epoch {epoch}: recall@20={result['recall@20']:.4f}, "
              f"ndcg@20={result['ndcg@20']:.4f}")

        if result["recall@20"] > self.best_recall:
            self.best_recall = result["recall@20"]
            self.best_epoch = epoch
            best_path = os.path.join(CHECKPOINT_DIR, f"best_epoch{epoch:04d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model._optimizer.state_dict(),
                'recall': result["recall@20"],
                'ndcg': result["ndcg@20"],
            }, best_path)
            print(f"  [ckpt] best → {best_path}")

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        print(f"\n[best] recall@20={self.best_recall:.4f} at epoch {self.best_epoch}")


def save_checkpoint(model, epoch, path):
    """保存完整训练状态，支持断点续训"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model._optimizer.state_dict(),
    }, path)
    print(f"[ckpt] epoch {epoch} → {path}")


def load_checkpoint(model, path):
    """恢复训练状态"""
    ckpt = torch.load(path, map_location=model._device)
    model.load_state_dict(ckpt['model_state_dict'])
    model._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch']


def main():
    resume = "--resume" in sys.argv

    # 加载配置（展平嵌套 YAML）
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"[train] config not found: {config_path}, using defaults")
        config = TrainingConfig()
        config['dataset'] = 'gowalla'
    else:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        # 展平 {data:{dataset:gowalla}, model:{emb_dim:64}, training:{lr:0.001}} → {dataset:gowalla, emb_dim:64, lr:0.001, ...}
        flat = {}
        for section in ['data', 'model', 'training', 'evaluation']:
            if section in raw:
                flat.update(raw[section])
        config = TrainingConfig(overrides=flat)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 数据
    loader = LightGCNDataLoader(config)
    params = loader.get_init_params()

    # 模型
    print("[train] Building adjacency matrix...")
    edge_index = torch.LongTensor([[u, i] for u, i in loader.all_train_pairs]).T
    model = LightGCN(config, params[KEY_NUM_USERS], params[KEY_NUM_ITEMS], edge_index)

    start_epoch = 0
    if resume and os.path.exists(LATEST_CKPT):
        start_epoch = load_checkpoint(model, LATEST_CKPT)
        print(f"[train] 从 epoch {start_epoch} 恢复训练")
    elif resume:
        print("[train] 未找到 latest.pt，从头训练")

    # 实验
    experiment = Experiment(config, model)
    # 替换 dataloader 为我们的实现
    experiment.dataloader = loader

    metric_log = MetricAccumulator()
    full_rank = FullRankingCallback(loader, model, config)

    trainer = Trainer(callbacks=[
        LoggerCallback(),
        metric_log,
        full_rank,
    ])

    print(f"[train] dataset={config.get(KEY_DATASET)}, epochs={config.epochs}, "
          f"lr={config.lr}, batch={config.batch_size}, "
          f"resume={resume}, start_epoch={start_epoch}")

    # 手动训练循环（而非 Trainer.run），支持 epoch 级别的 checkpoint 保存
    try:
        for epoch in range(start_epoch + 1, config.epochs + 1):
            # === Train ===
            model.set_train()
            epoch_losses = []
            for batch in loader.train_loader():
                m = model.train_step(batch)
                epoch_losses.append(m['loss'])

            # === Val ===
            model.set_eval()
            epoch_val_losses = []
            with torch.no_grad():
                for batch in loader.val_loader():
                    m = model.eval_step(batch)
                    epoch_val_losses.append(m['val_loss'])

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val = sum(epoch_val_losses) / len(epoch_val_losses)
            print(f"[epoch {epoch}/{config.epochs}] train_loss={avg_loss:.4f} "
                  f"val_loss={avg_val:.4f}")

            metric_log.history.append({
                "epoch": epoch, "loss": avg_loss, "val_loss": avg_val,
            })

            # 定期保存 checkpoint
            if epoch % config.save_every_n_epochs == 0:
                save_checkpoint(model, epoch, LATEST_CKPT)

            # 全量评估
            full_rank.on_epoch_end(epoch, {})

            # 保存训练历史
            with open(HISTORY_PATH, 'w') as f:
                json.dump(metric_log.history, f, indent=2)

    except KeyboardInterrupt:
        print("\n[train] 训练中断，保存 checkpoint...")
        save_checkpoint(model, epoch, LATEST_CKPT)
        print(f"[train] 已保存至 {LATEST_CKPT}，下次 --resume 从此恢复")

    # 最终保存
    save_checkpoint(model, config.epochs, LATEST_CKPT)
    print(f"[train] 完成。history → {HISTORY_PATH}")
    print(f"[train] eval_log → {EVAL_LOG_PATH}")
    print(f"[train] 运行 python models/lightgcn/outputs/plot_results.py {config.get(KEY_DATASET)} 生成图表")


if __name__ == "__main__":
    main()
