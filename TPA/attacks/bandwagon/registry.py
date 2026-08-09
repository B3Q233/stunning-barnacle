"""Bandwagon 攻击 —— 可攻击模型注册表

满足"选择模型进行投毒"的需求：攻击配置 ``model.name`` 指定受害推荐模型，
本模块负责把名字解析为模型类 + 默认超参配置。

新增模型时，只需在 AVAILABLE_MODELS 中登记一个条目：:

    "模型名": {
        "model_cls": "包.模块:类名",          # 必须
        "dataset_cls": "包.模块:类名",        # 可选，训练数据载入器
        "config_path": "模型自身 config.yaml 相对 TPA 根的路径",
        "description": "一句话说明",
    }

约定：被登记模型需实现 LightGCN 同款接口，攻击模块依赖：
- ``get_user_embeddings()`` / ``get_item_embeddings()``（评分矩阵）
- ``embedding`` 属性（warm-start 嵌入迁移）
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "lightgcn": {
        "model_cls": "models.lightgcn.model:LightGCN",
        "dataset_cls": "models.lightgcn.dataset:LightGCNDataset",
        "config_path": "models/lightgcn/config.yaml",
        "description": "LightGCN (SIGIR 2020)：3 层 LGC + 层组合 + BPR",
    },
}


def _import(qualname: str) -> Any:
    module_name, _, attr = qualname.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def get_model_entry(name: str) -> Dict[str, Any]:
    """解析模型名，未知名字直接报错并列出可用模型。"""
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"未知模型 '{name}'，当前可投毒/打分的模型: {list(AVAILABLE_MODELS)}。"
            f"如需新增，请在 attacks/bandwagon/registry.py 的 AVAILABLE_MODELS 中登记。"
        )
    return AVAILABLE_MODELS[name]


def get_model_cls(name: str):
    return _import(get_model_entry(name)["model_cls"])


def get_dataset_cls(name: str):
    entry = get_model_entry(name)
    if "dataset_cls" not in entry:
        return None
    return _import(entry["dataset_cls"])


def load_model_config(name: str,
                      overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """加载模型自身 config.yaml（默认超参），再用 overrides 覆盖 model 段。"""
    entry = get_model_entry(name)
    cfg_path = PROJECT_ROOT / entry["config_path"]
    if not cfg_path.exists():
        raise FileNotFoundError(f"模型配置不存在: {cfg_path}")

    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if overrides:
        if "model" in cfg and isinstance(cfg["model"], dict):
            cfg["model"].update(overrides)
        else:
            cfg.update(overrides)
    return cfg
