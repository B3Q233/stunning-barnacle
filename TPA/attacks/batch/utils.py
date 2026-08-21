"""Batch 公共工具：deep_merge / 路径 / 命名 / JSON。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # G:\Idea\TPA


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：嵌套 dict 深合并，list/标量整体覆盖；不修改入参。"""
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolution_k(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("classification", {}).get("k")
               or cfg.get("training", {}).get("k") or 20)


def group_name(cfg: Dict[str, Any]) -> str:
    return (f"{cfg['attack']['name']}_{cfg['experiment']['dataset']}_"
            f"{cfg['model']['name']}_top{resolution_k(cfg)}")


def flatten_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把 experiment.* 展开到顶层，删除 batch 段（override 由生成器另行处理）。"""
    out = dict(cfg)
    out.update(out.pop("experiment"))
    out.pop("batch", None)
    return out


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def public_cache_dir(cfg: Dict[str, Any]) -> Path:
    return (PROJECT_ROOT / "attacks" / "batch" / "cache" / "classification"
            / cfg["experiment"]["dataset"] / cfg["model"]["name"]
            / f"top{resolution_k(cfg)}")


def public_rec_freq_path(cfg: Dict[str, Any]) -> Path:
    return public_cache_dir(cfg) / "rec_freq.json"
