"""跨平台路径解析工具。

约定：配置中的文件/目录路径统一相对 TPA 项目根书写（如
``models/lightgcn/outputs/...``），运行时代码用 :func:`resolve_from_root`
解析到项目根下；若用户显式给出绝对路径则原样使用，避免在 Linux 上被误拼接，
或把 ``g:/Idea/...`` 当作相对路径在 CWD 下建出目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def resolve_from_root(path: PathLike, root: Path) -> Path:
    """把相对路径解析到 ``root`` 下；绝对路径保持原样。"""
    p = Path(path)
    return p if p.is_absolute() else root / p
