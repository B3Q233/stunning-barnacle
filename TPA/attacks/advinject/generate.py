"""AdvInject data stage: initialize and optimize fake-user interactions."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from attacks.advinject.common import AttackConfig, canonical_config, initialize_fake_data, load_meta, pairs_to_csr, project_fake, resolve_target_items, save_fake_artifacts, set_seed
from attacks.advinject.train_surrogate import compute_adversarial_gradient
from training.run_tag import resolve_run_tag, save_config_snapshot, write_latest_pointer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 为什么这样做：假用户档案不能一拍脑袋生成，需要在 surrogate 上做“对抗式
# 梯度下降”压低外层目标（目标物品在真实用户上的 log-softmax 概率）。
# 功能：选择目标 → 初始化假用户（模板用户行）→ 有限 unroll 梯度更新 →
# 二值投影 → 落盘中毒 meta/档案。
# 参考公式：updated = current − adv_lr·grad/‖grad‖₂（逐行归一化，形状均为
#   (n_fakes, n_items)），随后 project_fake 按 threshold 二值化。
# 使用举例：python -m attacks.advinject.run --config ... --mode data

def resolve_targets(meta, config):
    return resolve_target_items(meta, config, seed=int(config.get("seed", 1)))

def attack_paths(config):
    dataset = config.get("dataset", "ml100k")
    model = config.get("model", {}).get("name", "wmf")
    tag = resolve_run_tag(config)
    base = PROJECT_ROOT / "attacks" / "advinject"
    return base / "data" / "poisoned" / dataset / model / tag, base / "outputs" / dataset / model / tag

def generate(config):
    config = canonical_config(config)
    seed = int(config.get("seed", 1))
    device = config.get("training", {}).get("device", "cpu")
    set_seed(seed, str(device).startswith("cuda"))
    meta, _ = load_meta(config)
    targets = resolve_targets(meta, config)
    train_csr = pairs_to_csr(meta["train_pairs"], meta["num_users"], meta["num_items"])
    attack = AttackConfig.from_dict(config)
    n_fakes = int(attack.n_fakes if attack.n_fakes > 1 else meta["num_users"] * attack.n_fakes)
    initial = initialize_fake_data(train_csr, n_fakes, seed)
    current = initial.toarray().astype(np.float32)
    history = []
    best_current = current.copy()
    best_loss = float("inf")
    for epoch in range(1, attack.adv_epochs + 1):
        from training.epoch_log import log_train_line
        from training.timing import section_enter, section_exit

        _t_epoch = section_enter(f"Epoch {epoch}/{attack.adv_epochs}")
        gradient, record = compute_adversarial_gradient(train_csr, current, meta["num_items"], targets, config)
        # 矩阵形状：current/gradient/updated 均为 (n_fakes, n_items)；
        # 逐行 l2 归一化避免步长被个别大梯度主导（同上游实现）。
        row_norm = np.linalg.norm(gradient, axis=1, keepdims=True).clip(min=1e-12)
        before = current.copy()
        updated = before - attack.adv_lr * gradient / row_norm
        current = project_fake(torch.as_tensor(updated), attack.proj_threshold, targets, attack.click_targets).numpy()
        log_train_line(epoch, attack.adv_epochs, record["loss"])
        entry = {"epoch": epoch,
                 "epoch_seconds": section_exit(
                     f"Epoch {epoch}/{attack.adv_epochs}", _t_epoch),
                 "loss": record["loss"]}
        history.append(entry)
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            best_current = current.copy()
    data_dir, output_dir = attack_paths(config)
    save_config_snapshot(config, output_dir)
    save_fake_artifacts(data_dir, sparse.csr_matrix(best_current, dtype=np.float32), targets, history, meta, config)
    write_latest_pointer(data_dir.parent, resolve_run_tag(config))
    return {"data_dir": str(data_dir), "output_dir": str(output_dir), "targets": targets.tolist(), "num_fakes": n_fakes, "history": history, "best_loss": best_loss}

if __name__ == "__main__":
    from training.config_utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="attacks/advinject/config.yaml")
    args = parser.parse_args()
    print(generate(load_config(args.config)))
