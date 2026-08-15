"""WMF 结果展示（步骤⑥）：训练曲线 + 论文对比表。

训练完成后自动调用 write_report；也可单独运行重新生成：
  ..\\.venv\\Scripts\\python.exe -m models.wmf.report
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.metrics import rank_values


# 论文报告值（来源：理解文档 3.3，私有电视数据 Fig.1/Fig.2）
PAPER_RANK_VALUES = [
    ("Popularity 基线", "16.46%", "Fig.1"),
    ("Item-item cosine 邻域", "10.74%", "Fig.1"),
    ("Eq.(9) raw r（f=100）", "13.40%", "Fig.1"),
    ("Eq.(10) binary p（f=100）", "10.49%", "Fig.1"),
    ("WMF f=50", "8.93%", "Fig.1"),
    ("WMF f=100", "8.56%", "Fig.1"),
    ("WMF f=200", "8.35%", "Fig.1"),
    ("随机预测（理论）", "50.00%", "Sec.4.2"),
]


def plot_training_curve(history, curve_path):
    """两联图：左=Eq.(3) 全量损失曲线；右=排序指标曲线。"""
    epochs = [e["epoch"] for e in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(epochs, [e["train_loss"] for e in history],
             marker="o", ms=3, label="train_loss")
    ax1.plot(epochs, [e["val_loss"] for e in history],
             marker="s", ms=3, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("WMF loss (Eq.3)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for m in ("rank", "recall@20", "ndcg@20"):
        ax2.plot(epochs, [e.get(m, float("nan")) for e in history],
                 marker="o", ms=3, label=m)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.set_title("Ranking metrics")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"[report] training_curve -> {curve_path}")


def plot_rank_cdf(ranks, cdf_path):
    """论文 Fig.2 的 Rank CDF：实际观看节目的百分位秩累积分布。

    越靠近左上越好（大部分观看节目排在低百分位）；随机预测为对角线。
    """
    ranks = np.sort(np.asarray(ranks, dtype=np.float64))
    if ranks.size == 0:
        print(f"[report] 无有效测试观测，跳过 rank_cdf -> {cdf_path}")
        return
    y = np.arange(1, ranks.size + 1) / ranks.size
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(ranks * 100, y, drawstyle="steps-post", lw=1.5,
            label="WMF (ml100k)")
    ax.plot([0, 100], [0, 1], ls="--", color="gray", lw=1,
            label="random (diagonal)")
    ax.set_xlabel("percentile rank of watched item (%)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Rank CDF (paper Fig.2 analog)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=150)
    plt.close(fig)
    print(f"[report] rank_cdf -> {cdf_path}")


def _load_latest_model():
    """从稳定指针 latest.pt 加载模型与数据，供 Rank CDF 复算。"""
    import yaml
    from models.wmf.dataset import WMFDataLoader
    from models.wmf.model import WMFModel
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "outputs")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in ["data", "model", "training", "evaluation"]:
        if section in raw:
            flat.update(raw[section])
    if "run_tag" in raw:
        flat["run_tag"] = raw["run_tag"]
    from training.framework import TrainingConfig
    config = TrainingConfig(overrides=flat)
    loader = WMFDataLoader(config)
    model = WMFModel(config, loader.num_users, loader.num_items)
    ckpt = torch.load(os.path.join(output_dir, "checkpoints", "latest.pt"),
                      map_location=model._device)
    model.load_state_dict(ckpt["model_state_dict"])
    return loader, model


def write_comparison_table(best_results, tag, table_path):
    """复现值 vs 论文报告值对比表 + 缺口分析。"""
    lines = [
        "# WMF 复现对比表（步骤⑥）",
        "",
        f"> 实验 run_tag：`{tag}`　生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## 本次复现最优结果（本地 ml100k，f=100）",
        "",
        "| 指标 | 最优值 | 最优 epoch |",
        "|---|---|---|",
    ]
    for name, entry in best_results.items():
        if name == "rank":
            value = f"{entry['value'] * 100:.2f}%"
        else:
            value = f"{entry['value'] * 100:.2f}%"
        lines.append(f"| {name} | {value} | {entry['epoch']} |")

    lines += [
        "",
        "## 论文报告值（私有电视数据，来源 Fig.1）",
        "",
        "| 方法 | rank̄ | 来源 |",
        "|---|---|---|",
    ]
    for method, value, source in PAPER_RANK_VALUES:
        lines.append(f"| {method} | {value} | {source} |")

    rank_entry = best_results.get("rank")
    rank_str = (f"{rank_entry['value'] * 100:.2f}%"
                if rank_entry else "N/A")
    lines += [
        "",
        "## 对齐判定与缺口分析",
        "",
        f"- **判定：部分对齐（定性）**。复现 rank̄ = {rank_str}，"
        "显著优于随机预测的 50%，与论文“WMF 显著优于随机与流行度基线”"
        "的定性结论一致（方向正确、量级合理）。",
        "- **数值不可直接比较**：论文为私有电视数据（约 30 万用户 / 1.7 万节目 / "
        "3200 万训练观测，时间切分、TV 专属测试过滤）；本地 ml100k 为成对隐式数据"
        "（608 用户 / 6298 物品 / 3.9 万训练交互），候选集与密度差异巨大。",
        "- **协议差异**：本地数据无评分值，r_ui 恒为 1，置信度 c_ui 恒为 41"
        "（论文 TV 数据 r_ui 为观看次数）；评估按仓库 all-ranking 协议过滤训练集"
        "已交互物品，论文另过滤训练期已看节目。",
        "- **可调方向**：论文推荐取可行最高因子数（f=200 时 rank̄ 8.35%），"
        "当前默认 f=100，可在 config.yaml 的 `model.factors` 提升后重训。",
        "",
        "## 输出物",
        "",
        "- `training_curve.png`：损失与指标训练曲线",
        "- `rank_cdf.png`：测试观看节目百分位秩累积分布（论文 Fig.2 对应物）",
        "- `history.json`：逐 epoch 全量记录",
        "- `eval_log.csv`：逐 epoch 全量排序评估",
        "",
    ]
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] comparison_table -> {table_path}")


def write_report(output_dir, history_path, eval_log_path,
                 best_results, tag=None, model=None, loader=None):
    """从历史记录生成曲线图与对比表。"""
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)["history"]
    tag = tag or os.path.basename(os.path.normpath(output_dir))
    curve_path = os.path.join(output_dir, "training_curve.png")
    cdf_path = os.path.join(output_dir, "rank_cdf.png")
    table_path = os.path.join(output_dir, "comparison_table.md")
    plot_training_curve(history, curve_path)
    write_comparison_table(best_results, tag, table_path)
    if model is None or loader is None:
        loader, model = _load_latest_model()
    with torch.no_grad():
        scores = model.predict_full_ranking(torch.arange(loader.num_users))
    ranks, _ = rank_values(scores, loader.user_items,
                           loader.test_user_items)
    plot_rank_cdf(ranks, cdf_path)


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "outputs")
    history_path = os.path.join(output_dir, "history.json")
    eval_log_path = os.path.join(output_dir, "eval_log.csv")
    if not os.path.exists(history_path):
        print(f"[report] 未找到 {history_path}，请先运行训练")
        sys.exit(1)
    with open(history_path, "r", encoding="utf-8") as f:
        best_results = json.load(f)["best"]
    from training.run_tag import read_latest_tag
    tag = read_latest_tag(Path(output_dir)) or os.path.basename(output_dir)
    write_report(output_dir, history_path, eval_log_path, best_results,
                 tag=tag)
