"""AdvInject data stage: initialize and optimize fake-user interactions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from attacks.advinject.common import AttackConfig, canonical_config, initialize_fake_data, load_meta, pairs_to_csr, project_fake, resolve_target_items, save_fake_artifacts, set_seed
from attacks.advinject.train_surrogate import compute_adversarial_gradient
from training.run_tag import resolve_run_tag, save_config_snapshot, write_latest_pointer

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
        gradient, record = compute_adversarial_gradient(train_csr, current, meta["num_items"], targets, config)
        row_norm = np.linalg.norm(gradient, axis=1, keepdims=True).clip(min=1e-12)
        before = current.copy()
        updated = before - attack.adv_lr * gradient / row_norm
        current = project_fake(torch.as_tensor(updated), attack.proj_threshold, targets, attack.click_targets).numpy()
        record.update({"epoch": epoch, "changed": int(np.count_nonzero(current != before))})
        history.append(record)
        if record["loss"] < best_loss:
            best_loss = record["loss"]
            best_current = current.copy()
    data_dir, output_dir = attack_paths(config)
    save_config_snapshot(config, output_dir)
    save_fake_artifacts(data_dir, sparse.csr_matrix(best_current, dtype=np.float32), targets, history, meta, config)
    write_latest_pointer(data_dir.parent, resolve_run_tag(config))
    return {"data_dir": str(data_dir), "output_dir": str(output_dir), "targets": targets.tolist(), "num_fakes": n_fakes, "history": history, "best_loss": best_loss}

if __name__ == "__main__":
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="attacks/advinject/config.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        print(generate(yaml.safe_load(handle)))
