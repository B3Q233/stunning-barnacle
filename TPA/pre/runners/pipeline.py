# -*- coding: utf-8 -*-
"""pre 实验编排：把 Phase 1→5 串起来（只编排，不实现算法）。

阶段与产物（v2：原始模型改为"每个模型只训一次"的共享缓存）：
    Phase 1  targets   -> <exp>/targets.json
                          若 strategy=top_exposure，会先训练/复用参考模型再算曝光
    Phase 2  original  -> <exp>/original/<model>/clean/   （全量物品矩阵，参考系 z^0）
    Phase 3  degrade   -> <exp>/degraded/<model>/item_<id>/ratio_<pp>/
                          （nested 删除：D(0.1) ⊆ D(0.2) ⊆ … ⊆ D(0.9)）
    Phase 4  degraded  -> 同上目录下的条件模型（未攻击，测退化了多少）
    Phase 5  attack    -> <exp>/attacks/<model>/item_<id>/<attack>_ratio_<pp>/
    Phase 11/12 analyze-> <exp>/results/*.csv + summary.json

缓存语义：所有阶段可重复执行，已有产物默认复用（pre.cache.reuse）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pre.runners.common import TPA_ROOT, load_meta, read_json, save_json
from pre.runners.degrade import build_and_cache_degraded, ratio_dirname
from pre.runners.run_attack import run_attack
from pre.runners.targets import load_or_build_targets
from pre.runners.train_model import (load_original_record, train_and_cache,
                                     train_original)


def clean_meta_path(model_name: str, dataset: str) -> Path:
    """干净数据的 processed meta（目标选择与原始模型的输入）。

    优先用受害模型自己的 processed 目录，缺失时回退 lightgcn（与攻击模块一致）。
    """
    primary = (TPA_ROOT / "models" / model_name / "data" / "processed"
               / dataset / "meta.pkl")
    if primary.exists():
        return primary
    fallback = (TPA_ROOT / "models" / "lightgcn" / "data" / "processed"
                / dataset / "meta.pkl")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"找不到 {dataset} 的 processed meta：{primary} 与 {fallback} 都不存在")


def _seed(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("pre", {}).get("training", {}).get("seeds", [42])[0])


def _reuse(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("pre", {}).get("cache", {}).get("reuse", True))


def phase_targets(cfg: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    """Phase 1：目标物品集合（缓存，全实验复用）。

    strategy=top_exposure 时：先训练/复用参考模型的共享原始缓存，再在其上算
    全物品原生 HR@K，按曝光降序选目标。
    """
    pre = cfg["pre"]
    tcfg = pre.get("targets", {})
    strategy = str(tcfg.get("strategy", "top_popularity"))
    seed = _seed(cfg)
    if strategy == "top_exposure":
        ref_model = str(tcfg.get("exposure_source_model", "lightgcn"))
        ref_meta = load_meta(clean_meta_path(ref_model, cfg["dataset"]))
        rec = train_original(cfg, exp_dir, ref_model, ref_meta, seed, _reuse(cfg))
        from pre.runners.common import build_model, build_training_config
        import torch
        tcfg_model = build_training_config(cfg, ref_model)
        edge_index = torch.LongTensor(
            [[u, i] for u, i in ref_meta["train_pairs"]]).T
        model = build_model(ref_model, tcfg_model, ref_meta["num_users"],
                            ref_meta["num_items"], edge_index)
        model.load_state_dict(torch.load(rec["model_path"], map_location="cpu",
                                         weights_only=False))
        model.set_eval()
        from pre.analysis.exposure import native_exposure
        exposure = native_exposure(model, ref_meta, k=int(cfg.get("k", 10)))
        targets = load_or_build_targets(cfg, exp_dir, ref_meta, exposure)
    else:
        meta = load_meta(clean_meta_path(pre["models"][0], cfg["dataset"]))
        targets = load_or_build_targets(cfg, exp_dir, meta)
    if not (exp_dir / "targets.json").exists():
        save_json(targets, exp_dir / "targets.json")
    return targets


def phase_original(cfg: Dict[str, Any], exp_dir: Path,
                   targets: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Phase 2：每个模型训练**一份**完整训练集模型（参考嵌入 z^0 的来源）。"""
    out: Dict[str, Dict[str, Any]] = {}
    seed = _seed(cfg)
    for model_name in cfg["pre"]["models"]:
        meta = load_meta(clean_meta_path(model_name, cfg["dataset"]))
        out[model_name] = train_original(cfg, exp_dir, model_name, meta, seed,
                                         _reuse(cfg))
    return out


def phase_degrade(cfg: Dict[str, Any], exp_dir: Path,
                  targets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Phase 3：按删除比例造退化数据（嵌套、确定性、缓存删除结果）。"""
    recs: List[Dict[str, Any]] = []
    seed = _seed(cfg)
    for model_name in cfg["pre"]["models"]:
        clean = load_meta(clean_meta_path(model_name, cfg["dataset"]))
        for t in targets["items"]:
            for ratio in cfg["pre"]["deletion_ratios"]:
                recs.append(build_and_cache_degraded(
                    cfg, exp_dir, model_name, clean, t["item_id"],
                    float(ratio), seed, reuse=_reuse(cfg)))
    return recs


def phase_degraded(cfg: Dict[str, Any], exp_dir: Path,
                   targets: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Phase 4：在退化数据上重训（不攻击），得到 z^r —— 退化基线。"""
    out: Dict[str, Dict[str, Any]] = {}
    seed = _seed(cfg)
    for model_name in cfg["pre"]["models"]:
        clean = load_meta(clean_meta_path(model_name, cfg["dataset"]))
        for t in targets["items"]:
            for ratio in cfg["pre"]["deletion_ratios"]:
                info = build_and_cache_degraded(
                    cfg, exp_dir, model_name, clean, t["item_id"],
                    float(ratio), seed, reuse=_reuse(cfg))
                meta = load_meta(Path(info["meta_path"]))
                rec = train_and_cache(
                    cfg, exp_dir, model_name, meta, t["item_id"],
                    group="degraded", label=ratio_dirname(float(ratio)),
                    seed=seed, reuse=_reuse(cfg),
                    extra_meta={"deletion": {k: v for k, v in info.items()
                                             if k != "deleted_interactions"},
                                "meta_path": info["meta_path"]})
                out[f"{model_name}/{t['item_id']}/{ratio}"] = rec
    return out


def phase_attack(cfg: Dict[str, Any], exp_dir: Path, targets: Dict[str, Any],
                 attacks: Optional[Sequence[str]] = None,
                 ratios: Optional[Sequence[float]] = None,
                 models: Optional[Sequence[str]] = None
                 ) -> List[Dict[str, Any]]:
    """Phase 5：对退化数据跑全部攻击，并在中毒数据上重训得到 z^{A,r}。"""
    seed = _seed(cfg)
    attacks = list(attacks or cfg["pre"]["attacks"])
    ratios = [float(r) for r in (ratios or cfg["pre"]["deletion_ratios"])]
    models = list(models or cfg["pre"]["models"])
    records: List[Dict[str, Any]] = []
    for model_name in models:
        clean = load_meta(clean_meta_path(model_name, cfg["dataset"]))
        original = load_original_record(exp_dir, model_name)
        if original is None:
            # 分批推进时的常见情形：忘了先跑 --mode original。
            # 这里跳过并记录，不让整批崩掉（已完成条件不受影响）。
            print(f"[pre] [!] 跳过 {model_name}：缺少共享原始模型，"
                  f"请先跑 --mode original")
            records.append({"status": "skipped_no_original", "model_name": model_name})
            continue
        for t in targets["items"]:
            item_id = t["item_id"]
            for ratio in ratios:
                info = build_and_cache_degraded(cfg, exp_dir, model_name, clean,
                                                item_id, ratio, seed,
                                                reuse=_reuse(cfg))
                for attack_name in attacks:
                    arec = run_attack(
                        cfg, exp_dir, attack_name, model_name, item_id, ratio,
                        Path(info["meta_path"]),
                        clean_checkpoint=original["checkpoint_path"],
                        seed=seed, reuse=_reuse(cfg))
                    if arec.get("status") != "ok":
                        records.append(arec)
                        continue
                    meta = load_meta(Path(arec["poisoned_meta"]))
                    mrec = train_and_cache(
                        cfg, exp_dir, model_name, meta, item_id,
                        group="attacks",
                        label=f"{attack_name}_{ratio_dirname(ratio)}",
                        seed=seed, reuse=_reuse(cfg),
                        extra_meta={"attack": attack_name,
                                    "deletion_ratio": float(ratio),
                                    "degraded_meta": info["meta_path"],
                                    "poisoned_meta": arec["poisoned_meta"],
                                    "fake_users_added": int(meta["num_users"]) -
                                                       int(clean["num_users"])})
                    arec["attacked_record"] = {
                        "embedding_path": mrec["embedding_path"],
                        "model_path": mrec["model_path"]}
                    records.append(arec)
                    save_json(arec, exp_dir / "attacks" / model_name /
                              f"item_{item_id:05d}" /
                              f"{attack_name}_{ratio_dirname(ratio)}" /
                              "attack.json")
    return records
