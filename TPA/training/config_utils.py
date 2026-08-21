"""配置 k 统一解析与指标模板展开。

目标：k 只在一处定义（最外层 `k`，或回退 evaluation/classification/training.k），
各段与指标名自动绑定，避免"改 k 要改多处"。

约定：
- 指标名支持 `{k}` 模板，如 `target_ndcg@{k}` → `target_ndcg@10`；
- `apply_k` 返回深拷贝，不修改入参，可重复调用（幂等）。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List


def resolve_k(cfg: Dict[str, Any], default: int = 20) -> int:
    """解析统一 k：顶层 `k` > evaluation.k > classification.k > training.k > default。"""
    if cfg.get("k") is not None:
        return int(cfg["k"])
    for section in ("evaluation", "classification", "training"):
        sec = cfg.get(section)
        if isinstance(sec, dict) and sec.get("k") is not None:
            return int(sec["k"])
    return int(default)


def _expand_name(name: str, k: int) -> str:
    return str(name).replace("{k}", str(k))


def expand_metrics(metrics: Any, k: int) -> Any:
    """把指标列表中的 `{k}` 模板展开为实际 K。"""
    if not metrics:
        return metrics
    out: List[Any] = []
    for item in metrics:
        if isinstance(item, dict):
            out.append({_expand_name(name, k): direction
                        for name, direction in item.items()})
        elif isinstance(item, str):
            parts = item.strip().split()
            name = _expand_name(parts[0], k)
            out.append(" ".join([name] + parts[1:]) if len(parts) > 1 else name)
        else:
            out.append(item)
    return out


def apply_k(cfg: Dict[str, Any], default: int = 20) -> Dict[str, Any]:
    """返回副本：解析统一 k，注入各段 k，展开 evaluation.metrics / 顶层 metrics。"""
    out = copy.deepcopy(cfg)
    k = resolve_k(out, default)
    for section in ("classification", "training", "evaluation"):
        if isinstance(out.get(section), dict):
            out[section]["k"] = k
    ev = out.get("evaluation")
    if isinstance(ev, dict) and "metrics" in ev:
        ev["metrics"] = expand_metrics(ev["metrics"], k)
    if "metrics" in out:
        out["metrics"] = expand_metrics(out["metrics"], k)
    return out
