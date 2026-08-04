"""LightGCN 结果展示：训练曲线绘制 + 论文值对比"""
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history_path: str, output_path: str):
    """绘制训练曲线"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    losses = [h.get('loss', 0) for h in history]
    val_losses = [h.get('val_loss', 0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, losses, label='train_loss', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    ax2.plot(epochs, val_losses, label='val_loss', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {output_path}")


# 论文报告值 (Table 4, 3-layer LightGCN)
PAPER_VALUES = {
    "gowalla":    {"recall@20": 0.1823, "ndcg@20": 0.1555},
    "yelp2018":   {"recall@20": 0.0639, "ndcg@20": 0.0525},
    "amazon-book":{"recall@20": 0.0410, "ndcg@20": 0.0318},
}


def compare_with_paper(reproduced: dict, dataset: str, output_path: str):
    """生成复现值 vs 论文值对比表"""
    paper = PAPER_VALUES.get(dataset, {})
    lines = [
        f"# LightGCN 复现结果对比 ({dataset})",
        "",
        "| 指标 | 论文报告值 | 复现值 | 偏差 | 状态 |",
        "|------|-----------|--------|------|------|",
    ]
    for metric, paper_val in paper.items():
        repro_val = reproduced.get(metric, 0)
        diff_pct = (repro_val - paper_val) / paper_val * 100
        status = "✅ ±2%" if abs(diff_pct) <= 2 else "⚠️ 偏差较大"
        lines.append(f"| {metric} | {paper_val:.4f} | {repro_val:.4f} | {diff_pct:+.1f}% | {status} |")

    content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    print(content)
    print(f"Comparison table saved to {output_path}")
