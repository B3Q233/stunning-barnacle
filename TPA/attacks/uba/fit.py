"""UBA 攻击 —— 第 4 步：中毒模型拟合与对比评估（model 模式）

职责：
- 在中毒数据上【新建】一个受害模型实例（模型由 config `model.name` 指定，注册表见
  attacks/uba/registry.py；不覆盖干净模型）
- 可选 warm-start：把干净 checkpoint 中重叠的用户/物品嵌入迁移过来，
  新增的假用户行保持随机初始化，然后继续训练
- 训练完成后与干净模型做对比评估，产出：
  - `uba_comparison.md/json`：模型效用（投毒代价）+ 目标物品 HR@K/NDCG@K
  - `target_user_metrics.md/json`：论文口径的目标用户群 U_t 上的 HR@K/NDCG@K

不修改 models/* 下的任何代码，只通过注册表 import 复用。
用法:
  python attacks/uba/fit.py --config attacks/uba/config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.framework import TrainingConfig
from models.lightgcn.dataset import LightGCNDataset

from attacks.uba.registry import (
    get_dataset_cls,
    get_model_cls,
    load_model_config,
)
from attacks.uba.generate import (
    DEFAULT_OUT_DIR as POISONED_OUT_DIR,
    load_meta,
    load_yaml_config,
    raw_meta_path,
)
from attacks.uba.evaluate import (
    build_attack_eval_metrics,
    compare_models,
    compute_target_user_metrics,
    ranking_scores,
    save_report,
    save_target_user_report,
)
from training.config_utils import DEFAULT_K, resolve_k
from training.run_tag import (
    read_latest_tag,
    resolve_run_tag,
    save_config_snapshot,
)
from training.paths import resolve_from_root
from training.metrics import (
    BestTracker,
    eval_ks_from_metrics,
    safe_checkpoint_name,
)


def build_training_config(config: Dict[str, Any], dataset: str,
                          model_name: str = "lightgcn") -> TrainingConfig:
    """构造受害模型训练配置。

    优先级：模型自身 config.yaml 默认值 < 攻击配置 training 段 < model.overrides。
    这样攻击配置只写"与模型默认不同"的项，模型结构超参仍以模型自身配置为准。
    """
    model_cfg = load_model_config(
        model_name, overrides=config.get("model", {}).get("overrides"))
    model_defaults = model_cfg.get("model", {})
    tr_defaults = model_cfg.get("training", {})
    ev_defaults = model_cfg.get("evaluation", {})

    overrides: Dict[str, Any] = {
        "dataset": dataset,
        "emb_dim": model_defaults.get("emb_dim", 64),
        "n_layers": model_defaults.get("n_layers", 3),
        "init_method": model_defaults.get("init_method", "normal"),
        "factors": model_defaults.get("factors", 100),
        "lr": tr_defaults.get("lr", 0.001),
        "epochs": tr_defaults.get("epochs", 30),
        "batch_size": tr_defaults.get("batch_size", 256),
        "weight_decay": tr_defaults.get("weight_decay", 0.0001),
        "neg_ratio": tr_defaults.get("neg_ratio", 1),
        "device": tr_defaults.get("device", "cuda"),
        # K 优先级（spec 2026-09-17 §4.3）：攻击配置顶层 k > 模型配置已解析的 k > DEFAULT_K
        "k": resolve_k(config, default=ev_defaults.get("k", DEFAULT_K)),
        "eval_every": tr_defaults.get("eval_every", 5),
    }
    tr = config.get("training", {})
    for key in ("lr", "epochs", "batch_size", "weight_decay", "neg_ratio",
                "device", "k", "eval_every", "num_workers", "persistent_workers"):
        if key in tr:
            overrides[key] = tr[key]
    overrides.update(config.get("model", {}).get("overrides", {}))
    return TrainingConfig(overrides=overrides)


def resolve_metrics_cfg(config: Dict[str, Any], model_name: str) -> list:
    """攻击配置 evaluation.metrics 优先；缺省取模型自身 resolved metrics。

    两者都没有时按**已解析的 K** 派生 recall@K/ndcg@K 并打印告警——禁止再写死
    recall@20/ndcg@20（spec 2026-09-17 I3）。
    """
    model_cfg = load_model_config(
        model_name, overrides=config.get("model", {}).get("overrides"))
    metrics = config.get("evaluation", {}).get("metrics")
    if metrics is None:
        metrics = model_cfg.get("evaluation", {}).get("metrics")
    if metrics is None:
        k = resolve_k(config, default=resolve_k(model_cfg, default=DEFAULT_K))
        metrics = [f"recall@{k}", f"ndcg@{k}"]
        print(f"[fit] [!] {model_name} 未声明 evaluation.metrics，"
              f"按 resolved K={k} 派生 {metrics}")
    return metrics


def transfer_clean_embeddings(model, ckpt_path: Path, clean_num_users: int,
                              num_fake_users: int) -> None:
    """warm-start：把干净模型嵌入迁移到中毒模型。

    嵌入行布局: [0, M) 用户 + [M, M+N) 物品
    中毒模型:   [0, M+n_fake) 用户 + [M+n_fake, M+n_fake+N) 物品
    → 原用户行不变，物品行整体偏移 n_fake，假用户行保持随机初始化。
    """
    ckpt = torch.load(ckpt_path, map_location=model._device, weights_only=True)
    clean_emb = ckpt["model_state_dict"]["embedding.weight"]
    assert clean_emb.size(0) == clean_num_users + model.num_items, (
        f"clean checkpoint 嵌入行数 {clean_emb.size(0)} 与 "
        f"{clean_num_users} 用户 + {model.num_items} 物品不匹配"
    )
    assert clean_emb.size(1) == model.emb_dim

    poisoned_emb = model.embedding.weight.data
    poisoned_emb[:clean_num_users] = clean_emb[:clean_num_users]
    poisoned_emb[model.num_users:] = clean_emb[clean_num_users:]
    print(f"[fit] warm-start: 迁移 {clean_num_users} 用户 + {model.num_items} 物品嵌入；"
          f"{num_fake_users} 个假用户行保持随机初始化")


def build_model(cfg: TrainingConfig, meta: Dict[str, Any], model_cls,
                warm_start: bool, warm_ckpt: Path | None,
                clean_num_users: int | None = None):
    """在（中毒）数据上新建受害模型实例。"""
    edge_index = torch.LongTensor([[u, i] for u, i in meta["train_pairs"]]).T
    model = model_cls(cfg, meta["num_users"], meta["num_items"], edge_index)
    if warm_start:
        if warm_ckpt is None or not Path(warm_ckpt).exists():
            print(f"[fit] [!] warm_start=true 但 checkpoint 不存在: {warm_ckpt}，"
                  f"改为随机初始化训练")
            return model
        assert clean_num_users is not None
        transfer_clean_embeddings(model, Path(warm_ckpt), clean_num_users,
                                  meta["num_users"] - clean_num_users)
    return model


def _split_train_val(pairs: List[Tuple[int, int]], seed: int = 42
                     ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """95/5 随机划分，与仓库其它攻击/模型的数据导入逻辑保持一致。"""
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.95)
    return shuffled[:split], shuffled[split:]


def train_poisoned_model(cfg: TrainingConfig, poisoned_meta: Dict[str, Any],
                         out_dir: Path, warm_start: bool,
                         warm_ckpt: Path | None, clean_num_users: int | None,
                         model_cls, dataset_cls, metrics_cfg,
                         checkpoint_mode: str = "per_metric",
                         targets: List[int] | None = None,
                         clean_user_items: Dict[int, set] | None = None,
                         ) -> Tuple[Any, List[Dict[str, Any]]]:
    """训练中毒模型，返回 (model, history)。

    WMF 是全量 ALS 语义（无 mini-batch train_step），走共享 ALS 训练分支；
    其余模型走标准 mini-batch 训练循环 + 每 eval_every 轮全量评估。
    """
    if getattr(model_cls, "__name__", "") == "WMFModel":
        from models.wmf.train import train_wmf_from_meta

        return train_wmf_from_meta(
            cfg, poisoned_meta, out_dir, model_cls, metrics_cfg,
            checkpoint_mode=checkpoint_mode, targets=targets,
            clean_user_items=clean_user_items, warm_start=warm_start,
        )

    from training.epoch_log import log_eval_line, log_train_line, write_history
    from training.timing import section_enter, section_exit

    num_users = int(poisoned_meta["num_users"])
    num_items = int(poisoned_meta["num_items"])
    user_items = poisoned_meta["user_items"]
    neg_ratio = cfg.get("neg_ratio", 1)
    k = cfg.get("k", DEFAULT_K)
    eval_every = cfg.get("eval_every", 5)
    tracker = BestTracker(metrics_cfg, checkpoint_mode)

    model = build_model(cfg, poisoned_meta, model_cls, warm_start, warm_ckpt,
                        clean_num_users)
    train_pairs, val_pairs = _split_train_val(poisoned_meta["train_pairs"])
    train_ds = dataset_cls(train_pairs, num_items, user_items, num_users,
                           mode="train", neg_ratio=neg_ratio)
    val_ds = dataset_cls(val_pairs, num_items, user_items, num_users,
                         mode="train", neg_ratio=neg_ratio)
    num_workers = int(cfg.get("num_workers", 0))
    persistent = num_workers > 0 and bool(cfg.get("persistent_workers", False))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers,
                              persistent_workers=persistent, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=persistent, pin_memory=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    print(f"[fit] 中毒模型训练: users={num_users}, items={num_items}, "
          f"train={len(train_pairs)}, val={len(val_pairs)}, epochs={cfg.epochs}")

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = section_enter(f"Epoch {epoch}/{cfg.epochs}")
        model.set_train()
        epoch_losses: List[float] = []
        for batch in train_loader:
            epoch_losses.append(model.train_step(batch)["loss"])

        model.set_eval()
        epoch_val: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                epoch_val.append(model.eval_step(batch)["val_loss"])

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_val = sum(epoch_val) / len(epoch_val)
        log_train_line(epoch, cfg.epochs, avg_loss, avg_val)
        entry: Dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss,
                                 "val_loss": avg_val}

        if epoch % eval_every == 0 or epoch == 1:
            scores, users, test_pos_local = ranking_scores(
                model, poisoned_meta["test_pairs"])
            ks = eval_ks_from_metrics(metrics_cfg, k)
            res, target_details = build_attack_eval_metrics(
                scores, users, user_items, test_pos_local,
                clean_user_items or {}, targets or [], ks,
                list(tracker.directions), device=model._device,
            )
            entry.update(res)
            if target_details:
                entry["targets"] = target_details
            log_eval_line(res)
            improved = tracker.update(res, epoch)
            for name in improved:
                ckpt_path = ckpt_dir / f"{safe_checkpoint_name(name)}-best-model.pt"
                payload = {"epoch": epoch,
                           "model_state_dict": model.state_dict(),
                           "metrics": dict(res)}
                payload.update(res)
                torch.save(payload, ckpt_path)
                print(f"    [ckpt] best → {ckpt_path} ({name}={res[name]:.4f})")

        entry["epoch_seconds"] = section_exit(f"Epoch {epoch}/{cfg.epochs}", t_epoch)
        history.append(entry)

    torch.save({"epoch": cfg.epochs, "model_state_dict": model.state_dict()},
               ckpt_dir / "latest.pt")
    write_history(out_dir, history, tracker.best_results())
    print(f"[fit] 训练完成 → {out_dir}")
    return model, history


def load_clean_model(cfg: TrainingConfig, clean_meta: Dict[str, Any],
                     ckpt_path: Path, model_cls):
    """加载干净模型（仅用于对比，不被修改）。"""
    model = build_model(cfg, clean_meta, model_cls, warm_start=False,
                        warm_ckpt=None)
    ckpt = torch.load(ckpt_path, map_location=model._device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[fit] 干净模型加载完成: {ckpt_path}")
    return model


def target_user_comparison(clean_model, poisoned_model, clean_meta, poisoned_meta,
                           target_users: List[int], target_item: int, k: int
                           ) -> Dict[str, Any]:
    """论文口径：目标用户群 U_t 上的 clean vs poisoned HR@K / NDCG@K。

    为什么单独算：evaluation/attack_eval.py 的 target_hr@K 统计的是"所有未交互目标
    物品的合格用户"，而论文 Table 1 统计的是被攻击的 50 个目标用户；两者口径不同，
    都保留才能同时对齐论文与仓库基线表。
    """
    scores_c, users_c, _ = ranking_scores(clean_model, clean_meta["test_pairs"])
    scores_p, users_p, _ = ranking_scores(poisoned_model, poisoned_meta["test_pairs"])
    return {
        "k": k,
        "clean": compute_target_user_metrics(
            scores_c, users_c, clean_meta["user_items"], target_users,
            target_item, k, device=clean_model._device),
        "poisoned": compute_target_user_metrics(
            scores_p, users_p, clean_meta["user_items"], target_users,
            target_item, k, device=poisoned_model._device),
    }


from training.timing import timed  # noqa: E402


@timed("中毒模型拟合")
def main(config: Dict[str, Any], skip_train: bool = False,
         tag: str | None = None) -> Dict[str, Any]:
    """model 阶段入口（run.py / batch 统一调用 main(config)）。"""
    dataset = str(config["dataset"])
    attack_name = config["attack"]["name"]
    model_name = config.get("model", {}).get("name", "lightgcn")
    model_cls = get_model_cls(model_name)
    dataset_cls = get_dataset_cls(model_name) or LightGCNDataset
    out_root = Path(config.get("output", {}).get("dir", "attacks/uba/outputs"))
    poisoned_base = Path(str(POISONED_OUT_DIR).format(
        dataset=dataset, model=model_name))

    if tag or config.get("run_tag"):
        run_tag = resolve_run_tag(config, cli_tag=tag)
    else:
        run_tag = read_latest_tag(poisoned_base) or resolve_run_tag(config)

    out_dir = (PROJECT_ROOT / out_root / dataset / model_name / run_tag).resolve()
    poisoned_path = poisoned_base / run_tag / "meta.pkl"
    if not poisoned_path.exists():
        raise FileNotFoundError(
            f"未找到中毒数据 {poisoned_path}\n"
            f"run_tag={run_tag}，请先运行 generate.py 生成该实验的中毒数据，"
            f"或检查 attacks/uba/data/poisoned/{dataset}/{model_name}/ 下已有的 tag"
        )
    save_config_snapshot(config, out_dir)
    poisoned_meta = load_meta(poisoned_path)
    print(f"[fit] run_tag: {run_tag}")

    clean_meta_path = raw_meta_path(config, model_name)
    clean_meta = load_meta(clean_meta_path)

    stats_path = poisoned_path.parent / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    targets = [int(t["item_id"]) for t in stats["targets"]]
    target_users = [int(u) for u in stats["target_users"]["ids"]]
    print(f"[fit] 目标物品 {targets}；目标用户 {len(target_users)} 个"
          f"（分配策略 {stats['allocation']['strategy']}）")

    cfg = build_training_config(config, dataset, model_name)
    k = cfg.get("k", DEFAULT_K)
    metrics_cfg = resolve_metrics_cfg(config, model_name)
    checkpoint_mode = config.get("evaluation", {}).get("checkpoint_mode",
                                                       "per_metric")
    warm_cfg = config.get("warm_start", {})
    warm_start = bool(warm_cfg.get("enabled", True))
    warm_ckpt = (warm_cfg.get("checkpoint")
                 or (config.get("checkpoint") or {}).get("clean"))
    if warm_ckpt:
        warm_ckpt = resolve_from_root(warm_ckpt, PROJECT_ROOT)

    if skip_train:
        model = build_model(cfg, poisoned_meta, model_cls, warm_start=False,
                            warm_ckpt=None)
        primary = BestTracker(metrics_cfg).primary_metric
        ckpt_path = None
        if primary:
            ckpt_path = out_dir / "checkpoints" / \
                f"{safe_checkpoint_name(primary)}-best-model.pt"
        if ckpt_path is None or not ckpt_path.exists():
            ckpt_path = out_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = out_dir / "checkpoints" / "latest.pt"
        ckpt = torch.load(ckpt_path, map_location=model._device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[fit] 从 checkpoint 加载中毒模型: {ckpt_path}")
        history: List[Dict[str, Any]] = []
    else:
        model, history = train_poisoned_model(
            cfg, poisoned_meta, out_dir, warm_start, warm_ckpt,
            clean_num_users=clean_meta["num_users"] if warm_start else None,
            model_cls=model_cls, dataset_cls=dataset_cls,
            metrics_cfg=metrics_cfg, checkpoint_mode=checkpoint_mode,
            targets=targets, clean_user_items=clean_meta["user_items"],
        )

    clean_ckpt_cfg = ((config.get("checkpoint") or {}).get("clean")
                      or warm_cfg.get("checkpoint"))
    clean_ckpt = resolve_from_root(clean_ckpt_cfg, PROJECT_ROOT) if clean_ckpt_cfg else None
    if clean_ckpt is None or not clean_ckpt.exists():
        print(f"[fit] [!] 未配置/不存在干净 checkpoint（{clean_ckpt}），跳过对比评估")
        return {"dataset": dataset, "targets": targets, "history": history}
    clean_model = load_clean_model(cfg, clean_meta, clean_ckpt, model_cls)

    report_utility = bool(
        config.get("evaluation", {}).get("report_model_utility", True))
    report = compare_models(clean_model, model, clean_meta, poisoned_meta,
                            targets, k, report_utility=report_utility)
    report["allocation"] = stats["allocation"]
    report["target_users"] = {"count": len(target_users), "ids": target_users}
    md_path = save_report(report, out_dir, name=attack_name)
    print(f"[fit] 对比报告 → {md_path}")

    tum = target_user_comparison(clean_model, model, clean_meta, poisoned_meta,
                                 target_users, targets[0], k)
    tum_path = save_target_user_report(tum, out_dir)
    print(f"[fit] 目标用户群指标 → {tum_path}")
    report["target_user_metrics"] = tum
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UBA 中毒模型拟合")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "attacks" / "uba" / "config.yaml"))
    parser.add_argument("--skip-train", action="store_true",
                        help="跳过训练，直接加载已有 checkpoint 做对比评估")
    parser.add_argument("--tag", type=str, default=None,
                        help="实验标签 run_tag（优先于 config.run_tag 与 latest.json）")
    args = parser.parse_args()
    main(load_yaml_config(Path(args.config)), skip_train=args.skip_train,
         tag=args.tag)
