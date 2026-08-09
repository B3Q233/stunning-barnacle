"""多指标方向解析与最优跟踪（所有实验统一使用）

- `parse_metrics`：解析 evaluation.metrics 的方向标注
  （YAML 字典 `recall@20: upper` / 字符串 `"ndcg@20 lower"` / 裸字符串）
- `BestTracker`：按方向独立跟踪每个指标的最优 epoch，
  per_metric 模式每个指标一份 `{指标}-best-model.pt`（同 epoch 多指标各存一份）
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


VALID_DIRECTIONS = ("upper", "lower")

# 内置默认方向：按指标名前缀匹配（显式标注优先于本表）
DEFAULT_DIRECTIONS: Dict[str, str] = {
    # 越高越好
    "recall": "upper",
    "ndcg": "upper",
    "precision": "upper",
    "hit": "upper",
    "hr": "upper",
    "map": "upper",
    "auc": "upper",
    "acc": "upper",
    "f1": "upper",
    # 越低越好
    "loss": "lower",
    "rmse": "lower",
    "mae": "lower",
    "mse": "lower",
    "err": "lower",
}


def default_direction(name: str) -> str:
    """按内置表返回指标默认方向；未命中默认 upper。"""
    for key, direction in DEFAULT_DIRECTIONS.items():
        if name == key or name.startswith(key):
            return direction
    return "upper"


def parse_metrics(metrics_cfg: Optional[List[Any]]) -> Dict[str, str]:
    """解析 evaluation.metrics 列表为 {指标名: 方向}。

    支持三种元素：
    - {"recall@20": "upper"}：YAML 字典（显式标注）
    - "ndcg@20 lower"：字符串后跟方向
    - "hit@10"：裸字符串（按默认方向）
    """
    result: Dict[str, str] = {}
    if not metrics_cfg:
        return result
    for item in metrics_cfg:
        if isinstance(item, dict):
            for name, direction in item.items():
                direction = str(direction).strip().lower()
                if direction not in VALID_DIRECTIONS:
                    raise ValueError(
                        f"指标 {name!r} 的方向 {direction!r} 非法，"
                        f"可选 {VALID_DIRECTIONS}"
                    )
                result[str(name)] = direction
        elif isinstance(item, str):
            parts = item.strip().split()
            name = parts[0]
            if len(parts) > 1:
                direction = parts[1].lower()
                if direction not in VALID_DIRECTIONS:
                    raise ValueError(
                        f"指标 {name!r} 的方向 {direction!r} 非法，"
                        f"可选 {VALID_DIRECTIONS}"
                    )
                result[name] = direction
            else:
                result[name] = default_direction(name)
        else:
            raise TypeError(f"metrics 列表元素类型不支持: {type(item)}")
    return result


def safe_checkpoint_name(metric: str) -> str:
    """把指标名安全化为文件名片段（保留 @，替换路径非法字符/空白）。"""
    return re.sub(r'[\\/:*?"<>|\s]', "_", metric).strip("._")


class BestTracker:
    """按方向独立跟踪每个指标的最优结果。"""

    def __init__(self, metrics_cfg: Optional[List[Any]],
                 checkpoint_mode: str = "per_metric"):
        self.directions = parse_metrics(metrics_cfg)
        self.checkpoint_mode = checkpoint_mode
        if self.checkpoint_mode not in ("per_metric", "single"):
            raise ValueError(
                f"checkpoint_mode 非法: {self.checkpoint_mode!r}，"
                f"可选 per_metric | single"
            )
        self._best: Dict[str, Dict[str, Any]] = {}

    @property
    def primary_metric(self) -> Optional[str]:
        """第一个配置的指标名（single 模式与 --skip-train 用）。"""
        names = list(self.directions)
        return names[0] if names else None

    def update(self, metrics: Dict[str, float], epoch: int) -> List[str]:
        """用本 epoch 的指标快照更新各指标最优；返回本次刷新的指标名列表。"""
        improved: List[str] = []
        for name, direction in self.directions.items():
            if name not in metrics:
                continue
            value = float(metrics[name])
            best = self._best.get(name)
            if best is None:
                is_best = True
            elif direction == "upper":
                is_best = value > best["value"]
            else:
                is_best = value < best["value"]
            if is_best:
                self._best[name] = {
                    "epoch": epoch,
                    "value": value,
                    "metrics": dict(metrics),
                }
                improved.append(name)
        return improved

    def best_checkpoints(self) -> List[Tuple[str, str]]:
        """返回需要保存的 (指标, 文件名) 列表。
        per_metric：每个指标一份；single：仅第一个指标。"""
        names = list(self._best)
        if self.checkpoint_mode == "single":
            primary = self.primary_metric
            names = [primary] if primary in self._best else []
        return [(name, f"{safe_checkpoint_name(name)}-best-model.pt")
                for name in names]

    def best_results(self) -> Dict[str, Dict[str, Any]]:
        """history.json 的 best 段：每个指标最优的 epoch/value/全量快照/文件名。"""
        results: Dict[str, Dict[str, Any]] = {}
        for name, _ in self.best_checkpoints():
            entry = dict(self._best[name])
            entry["checkpoint"] = f"{safe_checkpoint_name(name)}-best-model.pt"
            results[name] = entry
        return results
