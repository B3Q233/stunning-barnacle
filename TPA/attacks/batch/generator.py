"""批量投毒攻击：配置生成（四层 Deep Merge + 分层采样）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TIER_NAMES = ("popular", "normal", "cold")


def validate_batch_config(cfg: Dict[str, Any]) -> None:
    exp = cfg.get("experiment")
    if not isinstance(exp, dict) or not exp.get("dataset"):
        raise ValueError("缺少 experiment.dataset")
    batch = cfg.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("缺少 batch 段")
    per_tier = batch.get("per_tier")
    if not isinstance(per_tier, int) or per_tier <= 0:
        raise ValueError("batch.per_tier 必须为正整数")
    tiers = batch.get("tiers")
    if not tiers:
        raise ValueError("batch.tiers 不能为空")
    for tier in tiers:
        if tier not in TIER_NAMES:
            raise ValueError(f"未知分层 {tier!r}，可选 {TIER_NAMES}")
    if batch.get("strategy", "random") not in ("random", "first"):
        raise ValueError("batch.strategy 仅支持 random|first")
    if not cfg.get("model", {}).get("name"):
        raise ValueError("缺少 model.name")
    if not cfg.get("attack", {}).get("name"):
        raise ValueError("缺少 attack.name")


def load_batch_config(path: Path) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_batch_config(cfg)
    return cfg
