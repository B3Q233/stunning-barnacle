"""批量投毒攻击：配置生成（四层 Deep Merge + 分层采样）。"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from attacks.batch.registry import get as get_attack
from attacks.batch.utils import deep_merge, flatten_experiment, group_name
from training.config_utils import apply_k
from training.run_tag import sanitize_run_tag


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
    return apply_k(cfg)


def load_attack_default(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """加载攻击插件自带的默认配置（P4）。"""
    import yaml
    spec = get_attack(cfg["attack"]["name"])
    with open(PROJECT_ROOT / spec.config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_atomic_base(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """四层合并（P4 攻击默认 ← P3 Batch ← P2 override），剔除扩展段。"""
    merged = deep_merge(load_attack_default(cfg), flatten_experiment(cfg))
    if isinstance(cfg.get("override"), dict):
        merged = deep_merge(merged, cfg["override"])
    merged.pop("override", None)
    return apply_k(merged)


def sample_targets(categories, tiers, per_tier, strategy="random",
                   seed=42):
    """每层采样 K 个目标物品；random 用固定 seed，first 取层内前 K。"""
    rng = random.Random(seed)
    out = {}
    for tier in tiers:
        pool = list(categories.get(tier, []))
        if not pool:
            print(f"[batch] 层 {tier} 为空，跳过")
            out[tier] = []
            continue
        out[tier] = (pool[:per_tier] if strategy == "first"
                     else rng.sample(pool, min(per_tier, len(pool))))
    return out


def atomic_run_tag(cfg, tier, item_id, batch_tag) -> str:
    return sanitize_run_tag(f"{batch_tag}-{tier}-item{item_id}")


def build_atomic_config(cfg, item_id, tier, batch_tag) -> dict:
    """生成单个原子配置：四层合并 + 运行时字段（P1）。"""
    atomic = copy.deepcopy(build_atomic_base(cfg))
    atomic["attack"]["target_items"] = {
        "strategy": "specified", "ids": [int(item_id)]}
    atomic["run_tag"] = atomic_run_tag(cfg, tier, item_id, batch_tag)
    atomic["output"] = {"dir": f"attacks/batch/output/{batch_tag}/runs"}
    return atomic


def config_rel_path(cfg, tier, item_id) -> str:
    return f"{group_name(build_atomic_base(cfg))}/{tier}/item{item_id}.yaml"


def generate_configs(cfg, categories, batch_tag) -> List[Tuple[str, dict]]:
    """返回 [(相对路径, 原子配置), ...]，相对路径为 {group}/{tier}/item{id}.yaml。"""
    sampling = cfg["batch"]
    base = build_atomic_base(cfg)
    group = group_name(base)
    targets = sample_targets(
        categories, sampling["tiers"], sampling["per_tier"],
        sampling.get("strategy", "random"), sampling.get("seed", 42))
    entries = []
    for tier, items in targets.items():
        for item in items:
            atomic = copy.deepcopy(base)
            atomic["attack"]["target_items"] = {
                "strategy": "specified", "ids": [int(item)]}
            atomic["run_tag"] = atomic_run_tag(cfg, tier, item, batch_tag)
            atomic["output"] = {
                "dir": f"attacks/batch/output/{batch_tag}/runs"}
            entries.append((f"{group}/{tier}/item{item}.yaml", atomic))
    return entries


def write_configs(entries, configs_dir) -> List[Path]:
    import yaml
    paths = []
    for rel, atomic in entries:
        p = configs_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(atomic, allow_unicode=True, sort_keys=False),
                     encoding="utf-8")
        paths.append(p)
    return paths
