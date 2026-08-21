"""环节计时工具：打印 【xx开始】 / [xx结束 耗时X分Y秒]。"""
from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar


_F = TypeVar("_F", bound=Callable)


def format_duration(seconds: float) -> str:
    """把秒格式化为 X分Y秒（Y 保留 1 位小数）。"""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}分{secs:.1f}秒"


def section_enter(name: str) -> float:
    """打印 【name开始】 并返回起始时间戳。"""
    print(f"【{name}开始】")
    return time.perf_counter()


def section_exit(name: str, start: float) -> float:
    """打印 [name结束 耗时X分Y秒]，返回耗时秒数。"""
    elapsed = time.perf_counter() - start
    print(f"[{name}结束 耗时{format_duration(elapsed)}]")
    return elapsed


class SectionTimer:
    """上下文管理器：with SectionTimer("数据注入"): ..."""

    def __init__(self, name: str):
        self.name = name
        self._start = 0.0

    def __enter__(self) -> "SectionTimer":
        self._start = section_enter(self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        section_exit(self.name, self._start)


def timed(name: str) -> Callable[[_F], _F]:
    """装饰器：函数开始/结束打印计时。"""
    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = section_enter(name)
            try:
                return func(*args, **kwargs)
            finally:
                section_exit(name, start)
        return wrapper  # type: ignore[return-value]
    return decorator
