"""TPA 攻击模型注册表（薄壳）

模型注册统一收口到公有注册表 models/registry.py（单一事实来源），
本文件仅保留路径兼容；active_model_name 为 TPA 专属逻辑，保留在本文件。
"""
from typing import Any, Dict

from models.registry import (  # noqa: F401
    AVAILABLE_MODELS,
    get_dataset_cls,
    get_model_cls,
    get_model_entry,
    load_model_config,
)


def active_model_name(config: Dict[str, Any]) -> str:
    """路径构造 / 频次分类使用的模型名。

    代理模式（surrogate.enabled=true）用 surrogate.model_name（攻击者自己的模型），
    否则用受害模型 model.name（白盒对照）。受害模型本身只用于评估与可选 warm-start。
    """
    sur = config.get("surrogate", {})
    if sur.get("enabled", False):
        name = sur.get("model_name")
        if not name:
            raise ValueError("surrogate.enabled=true 但未配置 surrogate.model_name")
        return str(name)
    return config.get("model", {}).get("name", "lightgcn")
