"""每 epoch 输出与产物统一组件。

规范见 docs/superpowers/specs/2026-09-02-epoch-output-standard-design.md：
- 控制台每 epoch 行 + 耗时；
- history.json 顶层 {history, best}，history 元素必含 epoch/epoch_seconds；
- eval_log.csv 表头 epoch + 动态指标名；
- 指标名单不在此层硬编码，以评测返回 dict 键为准。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from training.timing import SectionTimer


def epoch_section(epoch: int, total: int) -> SectionTimer:
    """每个 epoch 的计时上下文（打印 【Epoch i/N开始/结束 耗时】）。"""
    return SectionTimer(f"Epoch {epoch}/{total}")


def log_train_line(epoch: int, total: int, train_loss: float,
                   val_loss: float | None = None) -> None:
    """打印标准 epoch 训练行。"""
    line = f"  [epoch {epoch}/{total}] train_loss={train_loss:.4f}"
    if val_loss is not None:
        line += f" val_loss={val_loss:.4f}"
    print(line)


def log_eval_line(metrics: Dict[str, Any]) -> None:
    """打印该轮评测返回的全部指标（动态键，不筛选）。"""
    parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            parts.append(f"{name}={value:.4f}")
        else:
            parts.append(f"{name}={value}")
    print(f"    [eval] {', '.join(parts)}")


def write_history(out_dir, history: List[Dict[str, Any]],
                  best: Dict[str, Any]) -> Path:
    """写标准 history.json：{history: [...], best: {...}}。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "history.json"
    path.write_text(
        json.dumps({"history": history, "best": best},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_eval_log(out_dir, rows: List[Dict[str, Any]],
                   metric_names: List[str]) -> Path:
    """写 eval_log.csv；表头由传入指标名动态生成。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_log.csv"
    header = ["epoch"] + list(metric_names)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [row.get("epoch", "")] +
                [row.get(name, float("nan")) for name in metric_names])
    return path
