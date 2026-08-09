"""run_tag 实验隔离工具

约定（对所有实验统一采用）：
- 每次实验有一个 run_tag；默认取当前时间，格式 ``%Y-%m-%d-%H:%M``
  （例：2026-08-07-14:20）。
- Windows 文件系统不允许 ``:`` 等字符出现在目录名中，因此写入路径时
  ``:`` 会被替换为 ``-``（如 ``2026-08-07-14-20``）；日志/元数据中的
  run_tag 与目录名保持一致（即使用替换后的安全形式）。
- 优先级：CLI ``--tag`` > config ``run_tag`` > ``latest.json`` 指针
  （fit 阶段用，指向最近一次 data 生成） > 自动当前时间。
- 每次实验的产物写入独立目录，并在目录内保存 config.yaml 快照，
  互不覆盖；共享缓存（classify 的 rec_freq、tpa 的 path 缓存）不隔离。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RUN_TAG_FORMAT = "%Y-%m-%d-%H:%M"  # 例：2026-08-07-14:20


def default_run_tag() -> str:
    """默认实验标签：当前年月日时分。"""
    return datetime.now().strftime(DEFAULT_RUN_TAG_FORMAT)


def sanitize_run_tag(tag: str) -> str:
    """把 run_tag 变成可安全用作目录名的形式（替换 Windows 非法字符）。"""
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "-", tag.strip())
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    if not safe:
        raise ValueError(f"run_tag 非法（处理后为空）: {tag!r}")
    return safe


def resolve_run_tag(config: Optional[Dict[str, Any]] = None,
                    cli_tag: Optional[str] = None) -> str:
    """解析 run_tag：CLI --tag > config.run_tag > 自动当前时间。"""
    raw = cli_tag or (config or {}).get("run_tag") or default_run_tag()
    return sanitize_run_tag(raw)


def save_config_snapshot(config: Dict[str, Any], out_dir: Path) -> Path:
    """把本次实验使用的 config.yaml 快照保存到实验目录。"""
    import yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def write_latest_pointer(pointer_dir: Path, tag: str) -> Path:
    """在数据/模型基础目录写 latest.json，记录最近一次实验的 run_tag。
    供后续阶段（如 fit 单独运行时）解析出同一个 tag。"""
    pointer_dir.mkdir(parents=True, exist_ok=True)
    path = pointer_dir / "latest.json"
    path.write_text(
        json.dumps({"run_tag": tag}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_latest_tag(pointer_dir: Path) -> Optional[str]:
    """读取基础目录下的 latest.json 中的 run_tag；不存在返回 None。"""
    path = pointer_dir / "latest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("run_tag")
