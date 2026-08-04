"""LightGCN 训练组装车间
组装 DataLoader → Model → Trainer → 回调，启动训练流程。
每 epoch 末尾使用全量排名计算 recall@20 / ndcg@20。
"""
import sys
import os
import pickle
from typing import Dict
import torch
import numpy as np

from training.framework import (
    TrainingConfig, ConfigBuilder, Experiment, Trainer,
    LoggerCallback, CheckpointCallback, MetricAccumulator, Callback
)
from models.lightgcn.dataset import LightGCNDataLoader, KEY_NUM_USERS, KEY_NUM_ITEMS, KEY_DATASET
from models.lightgcn.model import LightGCN
from evaluation.metrics import compute_metrics


class FullRankingCallback(Callback):
    """每 N 个 epoch 做全量排名评估（Recall@K / NDCG@K）"""

    def __init__(self, loader: LightGCNDataLoader, model: LightGCN,
                 config: TrainingConfig, eval_every: int = 10):
        self.loader = loader
        self.model = model
        self.config = config
        self.eval_every = eval_every
        self.best_recall = 0.0
        self.best_epoch = 0

        # 加载测试集 ground truth
        dataset_name = config.get(KEY_DATASET, "gowalla")
        meta_path = f"g:/Idea/TPA/models/lightgcn/data/processed/{dataset_name}/meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.test_user_items = meta["user_items_test"] if "user_items_test" in meta else {}

        # 构建测试集 user → pos_items 映射
        test_pairs = meta.get("test_pairs", [])
        self.test_pos = {}
        for u, i in test_pairs:
            if u not in self.test_pos:
                self.test_pos[u] = set()
            self.test_pos[u].add(i)

        # 训练集用于过滤
        self.train_pos = meta["user_items"]

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        if epoch % self.eval_every != 0 and epoch != 1:
            return

        self.model.set_eval()
        with torch.no_grad():
            user_emb = self.model.get_user_embeddings()
            item_emb = self.model.get_item_embeddings()

            test_users = sorted(self.test_pos.keys())
            test_user_ids = torch.LongTensor(test_users).to(user_emb.device)

            # 分批计算全量排名
            batch_size = 1024
            scores_list = []
            for i in range(0, len(test_user_ids), batch_size):
                batch = test_user_ids[i:i + batch_size]
                scores_list.append(user_emb[batch] @ item_emb.T)

            scores = torch.cat(scores_list, dim=0).cpu()

        # 计算指标
        result = compute_metrics(scores, self.train_pos, self.test_pos, k=20)
        print(f"  [eval] epoch {epoch}: {result}")

        if result["recall@20"] > self.best_recall:
            self.best_recall = result["recall@20"]
            self.best_epoch = epoch

    def on_train_end(self, metrics: Dict[str, float]) -> None:
        print(f"\nBest recall@20={self.best_recall:.4f} at epoch {self.best_epoch}")


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"[train] config not found: {config_path}, using defaults")
        config = TrainingConfig()
        config['dataset'] = 'gowalla'
    else:
        config = TrainingConfig.from_yaml(config_path)

    # 数据
    loader = LightGCNDataLoader(config)
    params = loader.get_init_params()

    # 边列表 → 邻接矩阵（使用完整训练集）
    print("[train] Building adjacency matrix...")
    edge_index = torch.LongTensor([[u, i] for u, i in loader.all_train_pairs]).T

    # 模型
    model = LightGCN(config, params[KEY_NUM_USERS], params[KEY_NUM_ITEMS], edge_index)

    # 实验
    experiment = Experiment(config, model)
    experiment.dataloader = loader

    # 回调
    metric_log = MetricAccumulator()
    full_rank = FullRankingCallback(loader, model, config, eval_every=10)

    trainer = Trainer(callbacks=[
        LoggerCallback(),
        CheckpointCallback(model, config),
        metric_log,
        full_rank,
    ])

    print(f"[train] Starting: dataset={config.get(KEY_DATASET)}, "
          f"epochs={config.epochs}, lr={config.lr}, batch_size={config.batch_size}")
    trainer.run(experiment)

    # 保存训练历史
    import json
    hist_path = os.path.join(os.path.dirname(__file__), "outputs", "history.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(metric_log.history, f, indent=2)
    print(f"[train] history saved to {hist_path}")


if __name__ == "__main__":
    main()
