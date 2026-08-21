"""批量投毒：调度器（公共分类缓存 + 原子实验执行）。"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.batch.generator import build_atomic_base
from attacks.batch.registry import get as get_attack
from attacks.batch.utils import (
    public_cache_dir, read_json, resolution_k, write_json)


def attack_cache_path(cfg: Dict[str, Any]) -> Path:
    """攻击自带的分类缓存路径（由各攻击 classify 生成）。"""
    return (PROJECT_ROOT / "attacks" / cfg["attack"]["name"]
            / "data" / "rec_freq" / cfg["experiment"]["dataset"]
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
        "dataset": cfg["experiment"]["dataset"],
        "model": cfg["model"]["name"],
        "topk": resolution_k(cfg),
        "checkpoint": cfg.get("classification", {}).get("checkpoint"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }, cache_dir / "meta.json")


def ensure_classify_cache(cfg: Dict[str, Any],
                          cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """公共缓存存在则直接读；否则调攻击 classify 生成一次并归一化。"""
    cache_dir = cache_dir or public_cache_dir(cfg)
    target = cache_dir / "rec_freq.json"
    if target.exists():
        return read_json(target)
    spec = get_attack(cfg["attack"]["name"])
    base = build_atomic_base(cfg)
    base["mode"] = "classify"
    spec.classify(base)
    normalize_cache(cfg, read_json(attack_cache_path(cfg)), cache_dir)
    return read_json(target)
