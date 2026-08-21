"""批量投毒：调度器（公共分类缓存 + 原子实验执行）。"""
from __future__ import annotations

import sys
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.batch.generator import build_atomic_base
from attacks.batch.generator import generate_configs, write_configs
from attacks.batch.registry import get as get_attack
from attacks.batch.utils import (
    effective_dataset, public_cache_dir, read_json, resolution_k, write_json)


def attack_cache_path(cfg: Dict[str, Any]) -> Path:
    """攻击自带的分类缓存路径（由各攻击 classify 生成）。"""
    return (PROJECT_ROOT / "attacks" / cfg["attack"]["name"]
            / "data" / "rec_freq" / effective_dataset(cfg)
            / f"{cfg['model']['name']}_top{resolution_k(cfg)}.json")


def normalize_cache(cfg: Dict[str, Any], attack_cache: Dict[str, Any],
                    cache_dir: Path) -> None:
    """把攻击侧缓存归一化为公共缓存（ordinary → normal），并写 meta.json。"""
    categories = attack_cache["categories"]
    write_json({
        "popular": categories["popular"],
        "normal": categories["ordinary"],
        "cold": categories["cold"],
    }, cache_dir / "rec_freq.json")
    write_json({
        "dataset": effective_dataset(cfg),
        "model": cfg["model"]["name"],
        "topk": resolution_k(cfg),
        "checkpoint": cfg.get("classification", {}).get("checkpoint"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, cache_dir / "meta.json")


def ensure_classify_cache(cfg: Dict[str, Any],
                          cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """公共缓存存在则直接读；否则调攻击 classify 生成一次并归一化。"""
    base = build_atomic_base(cfg)
    cache_dir = cache_dir or public_cache_dir(base)
    target = cache_dir / "rec_freq.json"
    if target.exists():
        return read_json(target)
    spec = get_attack(base["attack"]["name"])
    classify_cfg = dict(base)
    classify_cfg["mode"] = "classify"
    spec.classify(classify_cfg)
    normalize_cache(base, read_json(attack_cache_path(base)), cache_dir)
    return read_json(target)


def plan_runs(cfg, categories, batch_tag):
    """生成原子配置计划：[(相对路径, 原子配置), ...]。"""
    return generate_configs(cfg, categories, batch_tag)


def write_meta(cfg, batch_tag, entries, meta_path) -> None:
    write_json({
        "batch_tag": batch_tag,
        "attack": cfg["attack"]["name"],
        "dataset": effective_dataset(cfg),
        "model": cfg["model"]["name"],
        "topk": resolution_k(cfg),
        "tiers": list(cfg["batch"]["tiers"]),
        "per_tier": cfg["batch"]["per_tier"],
        "total_runs": len(entries),
        "seed": cfg["batch"].get("seed", 42),
    }, meta_path)


def staging_dir(runs_root: Path, atomic_cfg: Dict[str, Any]) -> Path:
    """fit.py 固定拼接 {dataset}/{model}/{run_tag} 的 staging 目录。"""
    return (runs_root / atomic_cfg["dataset"]
            / atomic_cfg["model"]["name"] / atomic_cfg["run_tag"])


def _cleanup_staging(runs_root: Path, src: Path) -> None:
    """移动后清理空的 staging 父目录（如 runs/ml100k/lightgcn），保留 runs_root。"""
    parent = src.parent
    while parent != runs_root and parent != runs_root.parent:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def run_atomic(atomic_cfg: Dict[str, Any], stage: str) -> None:
    """执行单个原子实验的 data（generate）或 model（fit）阶段。"""
    spec = get_attack(atomic_cfg["attack"]["name"])
    fn = {"data": spec.generate, "model": spec.fit}[stage]
    fn(atomic_cfg)


def _file_logger(out_root: Path):
    """创建 runner.log 的 FileHandler；调用方用完后 removeHandler + close。"""
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"batch.{out_root.name}")
    handler = logging.FileHandler(log_dir / "runner.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger, handler


def run_batch(cfg, batch_tag, out_root, cache,
              max_targets=None, dry_run=False) -> None:
    """生成原子配置并逐个执行（data → model），整理到分层目录。"""
    configs_dir = out_root / "configs"
    runs_root = out_root / "runs"
    entries = plan_runs(cfg, cache, batch_tag)
    if max_targets is not None:
        entries = entries[:max_targets]
    write_configs(entries, configs_dir)
    write_meta(cfg, batch_tag, entries, out_root / "meta.json")
    logger, handler = _file_logger(out_root)
    try:
        logger.info("batch_tag=%s total_runs=%d dry_run=%s",
                    batch_tag, len(entries), dry_run)
        print(f"[batch] 原子配置 {len(entries)} 个 -> {configs_dir}")
        if dry_run:
            return
        for rel, atomic in entries:
            print(f"[batch] {atomic['run_tag']}")
            logger.info("run %s", atomic["run_tag"])
            run_atomic(atomic, "data")
            run_atomic(atomic, "model")
            src = staging_dir(runs_root, atomic)
            dst = runs_root / rel[:-len(".yaml")]
            if src != dst and src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(src), str(dst))
                _cleanup_staging(runs_root, src)
            logger.info("done %s", atomic["run_tag"])
    finally:
        logger.removeHandler(handler)
        handler.close()
