"""跨平台路径解析工具。

约定：配置中的文件/目录路径统一相对 TPA 项目根书写（如
``models/lightgcn/outputs/...``），运行时代码用 :func:`resolve_from_root`
解析到项目根下；若用户显式给出绝对路径则原样使用，避免在 Linux 上被误拼接，
或把"盘符 + 冒号 + 仓库目录"形式的绝对路径当作相对路径，在 CWD 下建出同名目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "implicit" / "raw"
EXPLICIT_RAW_DIR = PROJECT_ROOT / "data" / "explicit" / "raw"

DATASET_INTERACTION = {
    "gowalla": "implicit",
    "amazon-book": "implicit",
    "yelp2018": "implicit",
    "ml100k": "implicit",
}


def resolve_from_root(path: PathLike, root: Path) -> Path:
    """把相对路径解析到 ``root`` 下；绝对路径保持原样。"""
    p = Path(path)
    return p if p.is_absolute() else root / p


def raw_data_root(interaction: str) -> Path:
    """返回 implicit/explicit 的 raw 根目录；未知类型报错。"""
    if interaction == "implicit":
        return IMPLICIT_RAW_DIR
    if interaction == "explicit":
        return EXPLICIT_RAW_DIR
    raise ValueError(
        f"未知交互类型 {interaction!r}，可选: implicit | explicit"
    )


def raw_data_dir(dataset: str) -> Path:
    """返回 {dataset} 的原始数据目录；未知数据集报错并列出可用项。"""
    if dataset not in DATASET_INTERACTION:
        raise ValueError(
            f"未知数据集 {dataset!r}，已登记: {sorted(DATASET_INTERACTION)}"
        )
    return raw_data_root(DATASET_INTERACTION[dataset]) / dataset
