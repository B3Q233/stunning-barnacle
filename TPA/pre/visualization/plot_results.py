# -*- coding: utf-8 -*-
"""Phase 13：可视化（独立于实验执行，只读标准化 CSV）。

为什么与实验解耦（生产计划要求）：
    实验代码只负责产出 results/embedding_distances.csv。画图随时可重跑，
    不需要重新训练任何模型。改配色、改分组、加图，都不会影响实验产物。

四张图（与生产计划一致）：
    图1  删除比例 → 攻击后到原始嵌入的距离（按攻击方法分线）
    图2  删除比例 → 恢复率 RR（按攻击方法分线）
    图3  不同攻击方法对比（同一骨干、同一比例下的柱状/箱线）
    图4  LightGCN vs MF（同一攻击、同一比例）

用法示例：
    python pre/visualization/plot_results.py \
        --csv pre/outputs/<tag>/results/embedding_distances.csv \
        --out pre/outputs/<tag>/results/figures --alignment procrustes
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path, alignment: str, attack: str | None = None
              ) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("alignment") == alignment]
    if attack:
        rows = [r for r in rows if r.get("attack") == attack]
    return rows


def mean_by(rows: List[Dict[str, str]], key: str, value: str
            ) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        acc[r[key]].append(float(r[value]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--alignment", type=str, default="procrustes",
                    choices=["raw", "procrustes"])
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(Path(args.csv), args.alignment)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[plot] 没有 alignment={args.alignment} 的记录，跳过")
        return 0

    attacks = sorted({r["attack"] for r in rows})
    models = sorted({r["model"] for r in rows})

    # 图1 + 图2：删除比例 -> 攻击后距离 / 恢复率
    for value, fname, title in (
            ("attacked_distance_cos", "fig1_attacked_distance.png",
             "删除比例 vs 攻击后到原始嵌入的余弦距离"),
            ("recovery_rate_cos", "fig2_recovery_rate.png",
             "删除比例 vs 恢复率 RR")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in models:
            for attack in attacks:
                sub = [r for r in rows
                       if r["model"] == model and r["attack"] == attack]
                if not sub:
                    continue
                by_ratio = mean_by(sub, "deletion_ratio", value)
                xs = sorted(by_ratio, key=float)
                ax.plot([float(x) for x in xs], [by_ratio[x] for x in xs],
                        marker="o", label=f"{model}/{attack}")
        ax.set_xlabel("deletion ratio")
        ax.set_ylabel(value)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # 图3：同一骨干下不同攻击的对比（按比例分组的柱状）
    for model in models:
        sub = [r for r in rows if r["model"] == model]
        ratios = sorted({float(r["deletion_ratio"]) for r in sub})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.8 / max(1, len(attacks))
        for i, attack in enumerate(attacks):
            ys = []
            for ratio in ratios:
                cell = [float(r["recovery_rate_cos"]) for r in sub
                        if r["attack"] == attack
                        and abs(float(r["deletion_ratio"]) - ratio) < 1e-9]
                ys.append(sum(cell) / len(cell) if cell else 0.0)
            xs = [j + i * width for j in range(len(ratios))]
            ax.bar(xs, ys, width=width, label=attack)
        ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ratios))])
        ax.set_xticklabels([f"{r:g}" for r in ratios])
        ax.set_xlabel("deletion ratio")
        ax.set_ylabel("recovery_rate_cos")
        ax.set_title(f"{model}：不同攻击方法的恢复率")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig3_attack_compare_{model}.png", dpi=150)
        plt.close(fig)

    # 图4：LightGCN vs MF（同一攻击）
    for attack in attacks:
        sub = [r for r in rows if r["attack"] == attack]
        if len({r["model"] for r in sub}) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in models:
            s2 = [r for r in sub if r["model"] == model]
            by_ratio = mean_by(s2, "deletion_ratio", "recovery_rate_cos")
            xs = sorted(by_ratio, key=float)
            ax.plot([float(x) for x in xs], [by_ratio[x] for x in xs],
                    marker="s", label=model)
        ax.set_xlabel("deletion ratio")
        ax.set_ylabel("recovery_rate_cos")
        ax.set_title(f"{attack}：LightGCN vs MF")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig4_model_compare_{attack}.png", dpi=150)
        plt.close(fig)

    print(f"[plot] 图已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
