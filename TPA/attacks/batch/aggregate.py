"""批量投毒：结果整合（results.csv + 按层 mean±std + clean 基线）。"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from attacks.batch.utils import effective_dataset
from training.timing import timed


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


def _metric_value(best: Dict[str, Any], key: str):
    """BestTracker 格式为 {指标: {epoch, value, metrics, checkpoint}}，
    提取 value；兼容旧式扁平 {指标: 数值}。"""
    entry = best.get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


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
            "dataset": effective_dataset(cfg),
            "model": cfg["model"]["name"],
            "tier": tier,
            "item": item,
        }
        for key in (f"target_hr@{k}", f"target_ndcg@{k}",
                    f"recall@{k}", f"ndcg@{k}"):
            row[key] = _metric_value(best, key)
        rows.append(row)
    return rows


def write_results_csv(rows: List[Dict[str, Any]], k: int, path: Path,
                      summary: Optional[Dict[str, Any]] = None) -> None:
    fieldnames = ["attack", "dataset", "model", "tier", "item",
                  f"target_hr@{k}", f"target_ndcg@{k}",
                  f"recall@{k}", f"ndcg@{k}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fn: row.get(fn, "") for fn in fieldnames})

        if summary:
            base = {
                "attack": rows[0]["attack"] if rows else "",
                "dataset": rows[0]["dataset"] if rows else "",
                "model": rows[0]["model"] if rows else "",
            }
            for tier, metrics in sorted(summary.items()):
                row = dict(base)
                row["tier"] = tier
                row["item"] = "avg"
                for metric in (f"target_hr@{k}", f"target_ndcg@{k}",
                               f"recall@{k}", f"ndcg@{k}"):
                    row[metric] = metrics[metric]["mean"]
                writer.writerow({fn: row.get(fn, "") for fn in fieldnames})


def tier_summary(rows: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """按层统计四项指标（target_hr/target_ndcg/recall/ndcg）的 mean/std/n。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["tier"], []).append(row)
    out = {}
    for tier, group in sorted(grouped.items()):
        out[tier] = {}
        for metric in (f"target_hr@{k}", f"target_ndcg@{k}",
                       f"recall@{k}", f"ndcg@{k}"):
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


def write_tier_stats_json(batch_tag: str, summary: Dict[str, Any], k: int,
                          path: Path) -> None:
    """把按层统计（mean/std/n）写入 run_tag 目录下的 tier_stats.json。"""
    import json
    payload = {"batch_tag": batch_tag, "k": k, "tiers": {}}
    for tier, metrics in sorted(summary.items()):
        payload["tiers"][tier] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


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


@timed("Clean 基线计算")
def compute_clean_baseline(cfg: Dict[str, Any], k: int) -> Dict[str, float]:
    """用 w_clean 在 clean 数据上算 recall@k / ndcg@k（投毒代价基线）。"""
    import torch

    from attacks.batch.generator import build_atomic_base
    from attacks.batch.registry import get as get_attack
    from evaluation.attack_eval import ranking_scores
    from evaluation.metrics import build_train_mask_indices, compute_metrics
    from training.paths import resolve_from_root

    attack = cfg["attack"]["name"]
    model_name = cfg["model"]["name"]
    get_attack(attack)
    gen_mod = __import__(f"attacks.{attack}.generate", fromlist=["load_meta"])
    fit_mod = __import__(f"attacks.{attack}.fit",
                         fromlist=["build_training_config"])
    reg_mod = __import__(f"attacks.{attack}.registry",
                         fromlist=["get_model_cls"])

    base = build_atomic_base(cfg)
    dataset = base["dataset"]
    meta_path = (PROJECT_ROOT / "models" / model_name / "data"
                 / "processed" / dataset / "meta.pkl")
    if not meta_path.exists():
        # 兼容各攻击自定义的干净 meta 路径接口（bandwagon: DEFAULT_RAW_META，
        # random: clean_meta_path）
        if hasattr(gen_mod, "DEFAULT_RAW_META"):
            meta_path = Path(
                str(gen_mod.DEFAULT_RAW_META).format(dataset=dataset))
        elif hasattr(gen_mod, "clean_meta_path"):
            meta_path = Path(gen_mod.clean_meta_path(model_name, dataset))
    meta = gen_mod.load_meta(meta_path)
    train_cfg = fit_mod.build_training_config(base, dataset, model_name)
    edge_index = torch.LongTensor(
        [[u, i] for u, i in meta["train_pairs"]]).T
    model = reg_mod.get_model_cls(model_name)(
        train_cfg, meta["num_users"], meta["num_items"], edge_index)
    ckpt = resolve_from_root(cfg["classification"]["checkpoint"], PROJECT_ROOT)
    model.load_state_dict(torch.load(
        ckpt, map_location=model._device, weights_only=True)["model_state_dict"])
    scores, users, test_pos = ranking_scores(model, meta["test_pairs"])
    rows, cols = build_train_mask_indices(meta["user_items"], users)
    return compute_metrics(
        scores, meta["user_items"], test_pos, k=k,
        mask_indices=(rows, cols),
        topk_device=model._device)
