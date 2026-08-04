"""LightGCN 结果展示
用法:
  python models/lightgcn/outputs/plot_results.py <dataset>
  python models/lightgcn/outputs/plot_results.py gowalla

从 outputs/ 读取训练历史，生成训练曲线图和论文对比表。
"""
import json
import csv
import sys
import os
import matplotlib
matplotlib.use('Agg')  # 非交互后端，无 GUI 也能运行
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAPER_VALUES = {
    "gowalla":     {"recall@20": 0.1823, "ndcg@20": 0.1555},
    "yelp2018":    {"recall@20": 0.0639, "ndcg@20": 0.0525},
    "amazon-book": {"recall@20": 0.0410, "ndcg@20": 0.0318},
}


def plot_training_history(history_path: str, output_path: str):
    """绘制训练/验证 loss 曲线和评估指标曲线"""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) 训练 loss
    ax = axes[0]
    losses = [h.get('loss', 0) for h in history]
    ax.plot(epochs, losses, linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')

    # 2) 验证 loss
    ax = axes[1]
    val_losses = [h.get('val_loss', 0) for h in history]
    ax.plot(epochs, val_losses, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss')
    ax.set_title('Validation Loss')

    # 3) 评估指标 (如果 eval_log.csv 存在)
    ax = axes[2]
    eval_path = os.path.join(OUTPUT_DIR, 'eval_log.csv')
    if os.path.exists(eval_path):
        eval_epochs, recalls, ndcgs = [], [], []
        with open(eval_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                eval_epochs.append(int(row['epoch']))
                recalls.append(float(row['recall@20']))
                ndcgs.append(float(row['ndcg@20']))
        if eval_epochs:
            ax.plot(eval_epochs, recalls, 'o-', label='recall@20', markersize=4)
            ax.plot(eval_epochs, ndcgs, 's-', label='ndcg@20', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')
    ax.set_title('Evaluation Metrics')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[plot] 训练曲线 → {output_path}")


def compare_with_paper(dataset: str, output_path: str):
    """从 eval_log.csv 读取最佳指标，与论文值对比"""
    eval_path = os.path.join(OUTPUT_DIR, 'eval_log.csv')
    reproduced = {}
    if os.path.exists(eval_path):
        best_recall = 0
        best_ndcg = 0
        best_epoch = 0
        with open(eval_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = float(row['recall@20'])
                n = float(row['ndcg@20'])
                if r > best_recall:
                    best_recall = r
                    best_epoch = int(row['epoch'])
                if n > best_ndcg:
                    best_ndcg = n
        reproduced = {"recall@20": best_recall, "ndcg@20": best_ndcg}

    paper = PAPER_VALUES.get(dataset, {})
    lines = [
        f"# LightGCN vs Paper ({dataset})",
        "",
        f"最佳 epoch: {best_epoch}",
        "",
        "| 指标 | 论文值 | 复现值 | 偏差 | 状态 |",
        "|------|--------|--------|------|------|",
    ]
    all_ok = True
    for metric, paper_val in paper.items():
        repro_val = reproduced.get(metric, 0)
        diff_pct = (repro_val - paper_val) / paper_val * 100 if paper_val > 0 else 0
        ok = abs(diff_pct) <= 2
        if not ok:
            all_ok = False
        status = "✅" if ok else "⚠️"
        lines.append(f"| {metric} | {paper_val:.4f} | {repro_val:.4f} | {diff_pct:+.1f}% | {status} |")

    if all_ok and reproduced:
        lines.append(f"\n✅ 全部指标在 ±2% 偏差内，复现成功。")
    elif reproduced:
        lines.append(f"\n⚠️ 部分指标偏差超过 ±2%，需排查。")

    content = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(content)
    print(content)
    print(f"[plot] 对比表 → {output_path}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "gowalla"
    history_path = os.path.join(OUTPUT_DIR, 'history.json')
    curve_path = os.path.join(OUTPUT_DIR, 'training_curve.png')
    table_path = os.path.join(OUTPUT_DIR, 'comparison_table.md')

    if not os.path.exists(history_path):
        print(f"[plot] 未找到 history.json，请先完成训练。路径: {history_path}")
        sys.exit(1)

    plot_training_history(history_path, curve_path)
    compare_with_paper(dataset, table_path)


if __name__ == "__main__":
    main()
