"""评测指标注册表（单一事实来源）。

新增指标：实现 f(scores, **kwargs) -> float，调用 register_metric 登记，
并在配置 evaluation.metrics 中加入指标名；epoch 输出/history/csv 自动同步。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List


METRICS: Dict[str, Callable[..., float]] = {}


def register_metric(name: str, fn: Callable[..., float]) -> None:
    if name in METRICS:
        raise ValueError(f"指标 {name!r} 已注册")
    METRICS[name] = fn


def get_metric(name: str) -> Callable[..., float]:
    if name not in METRICS:
        raise KeyError(f"未注册指标 {name!r}，可用: {sorted(METRICS)}")
    return METRICS[name]


def compute_metrics(names: List[str], scores: Any, **kwargs) -> Dict[str, float]:
    """按名字依次计算，返回 {指标名: 数值}（动态键）。"""
    return {name: float(get_metric(name)(scores, **kwargs))
            for name in names}
