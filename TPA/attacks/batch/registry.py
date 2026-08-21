"""Batch 攻击插件注册器（Plugin Registry）。"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class AttackSpec:
    name: str
    config_path: str
    classify: Callable
    generate: Callable
    fit: Callable


_REGISTRY: Dict[str, AttackSpec] = {}


def register(name: str, config_path: str, classify: Callable,
             generate: Callable, fit: Callable) -> None:
    if name in _REGISTRY:
        raise ValueError(f"攻击 {name} 已注册")
    _REGISTRY[name] = AttackSpec(
        name=name, config_path=config_path,
        classify=classify, generate=generate, fit=fit)


def get(name: str) -> AttackSpec:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的攻击 {name!r}，可用：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_names() -> List[str]:
    return sorted(_REGISTRY)


def _register_builtin() -> None:
    for name in ("bandwagon", "random", "pgd", "tpa"):
        register(
            name,
            f"attacks/{name}/config.yaml",
            classify=importlib.import_module(f"attacks.{name}.classify").main,
            generate=importlib.import_module(f"attacks.{name}.generate").main,
            fit=importlib.import_module(f"attacks.{name}.fit").main,
        )


_register_builtin()
