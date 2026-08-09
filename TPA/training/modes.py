"""run.py 三阶段编排模式解析（所有攻击统一使用）

模式含义：
- classify：只跑推荐频次分类
- data：只跑画像生成/数据注入
- model：只跑中毒模型拟合与评估
- both：data + model
- all：classify + data + model
"""
from __future__ import annotations

from typing import Tuple


VALID_MODES = ("classify", "data", "model", "both", "all")


def stages_for_mode(mode: str) -> Tuple[bool, bool, bool]:
    """返回 (run_classify, run_data, run_model)。"""
    if mode not in VALID_MODES:
        raise ValueError(
            f"未知 mode {mode!r}，可选 {VALID_MODES}"
        )
    return (
        mode in ("classify", "all"),
        mode in ("data", "both", "all"),
        mode in ("model", "both", "all"),
    )
