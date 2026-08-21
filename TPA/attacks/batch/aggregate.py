"""批量投毒：结果整合（results.csv + 按层 mean±std + clean 基线）。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_best_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("best")


def scan_runs(runs_root: Path, group: str) -> List[Tuple[str, int, Dict[str, Any]]]:
    """扫描分层 runs 目录，返回 [(tier, item_id, best_metrics), ...]。"""
    base = runs_root / group
    if not base.exists():
        return []
    out = []
    for tier_dir in sorted(base.iterdir()):
        if not tier_dir.is_dir():
            continue
        for item_dir in sorted(tier_dir.iterdir()):
            if not item_dir.is_dir() or not item_dir.name.startswith("item"):
                continue
            best = load_best_metrics(item_dir)
            if best is None:
                continue
            out.append((tier_dir.name, int(item_dir.name[len("item"):]), best))
    return out


def build_results_rows(runs_root: Path, group: str, cfg: Dict[str, Any],
                       k: int) -> List[Dict[str, Any]]:
    rows = []
    for tier, item, best in scan_runs(runs_root, group):
        row = {
            "attack": cfg["attack"]["name"],
            "dataset": cfg["experiment"]["dataset"],
            "model": cfg["model"]["name"],
            "tier": tier,
            "item": item,
        }
        for key in (f"target_hr@{k}", f"target_ndcg@{k}",
                    f"recall@{k}", f"ndcg@{k}"):
            row[key] = best.get(key)
        rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, Any]], k: int, path: Path) -> None:
    fieldnames = ["attack", "dataset", "model", "tier", "item",
                  f"target_hr@{k}", f"target_ndcg@{k}",
                  f"recall@{k}", f"ndcg@{k}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})


def tier_summary(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """按层统计 target_hr@{k} / target_ndcg@{k} 的 mean/std/n。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tier"], []).append(row)
    out = {}
    for tier, group in sorted(grouped.items()):
        out[tier] = {}
        for metric in (f"target_hr@{k}", f"target_ndcg@{k}"):
            vals = [r[metric] for r in group
                    if isinstance(r.get(metric), (int, float))]
            if not vals:
                out[tier][metric] = {"mean": None, "std": None, "n": 0}
                continue
            out[tier][metric] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return out


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:.4f}"


def write_summary_md(batch_tag, summary, clean_baseline, k, path) -> None:
    lines = [
        f"# 批量投毒攻击结果汇总（batch_tag={batch_tag}）",
        "",
        f"| Tier | HR@{k} | NDCG@{k} |",
        "|---|---|---|",
    ]
    for tier in sorted(summary):
        hr = summary[tier][f"target_hr@{k}"]
        nd = summary[tier][f"target_ndcg@{k}"]
        lines.append(f"| {tier} | {_fmt(hr['mean'])} ± {_fmt(hr['std'])} "
                     f"| {_fmt(nd['mean'])} ± {_fmt(nd['std'])} |")
    if clean_baseline:
        r = clean_baseline.get(f"recall@{k}", float("nan"))
        n = clean_baseline.get(f"ndcg@{k}", float("nan"))
        lines += ["", "## Clean Model Utility",
                  "", f"Recall@{k} : {r:.4f}", f"NDCG@{k}   : {n:.4f}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
