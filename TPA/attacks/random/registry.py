"""Random 攻击模型注册表（薄壳）

模型注册统一收口到公有注册表 models/registry.py（单一事实来源），
本文件仅保留路径兼容，既有 import 行为不变。
"""
from models.registry import (  # noqa: F401
    AVAILABLE_MODELS,
    get_dataset_cls,
    get_model_cls,
    get_model_entry,
    load_model_config,
)
