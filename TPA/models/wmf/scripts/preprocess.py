"""WMF 数据处理脚本（步骤①）

依据理解文档「模块一」：
- 本地 ml100k 为成对隐式反馈格式（user item，无评分值），r_ui 恒为 1；
  p_ui = 1 if r_ui > 0 else 0 恒为 1，置信度 c_ui = 1 + α·r_ui 在数据导入
  阶段按 config 的 α 计算（见 dataset.py）。
- 本脚本只做：解析原始文件 -> id 重映射为连续 0..n-1 -> 产出 meta.pkl 与
  train_pairs.txt / test_pairs.txt，供数据导入阶段（步骤②）加载。

用法（在 TPA 目录下）:
  ..\\.venv\\Scripts\\python.exe models\\wmf\\scripts\\preprocess.py --dataset ml100k
"""
import argparse
import os
import pickle
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # TPA 项目根
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "models" / "wmf" / "data" / "processed"


def parse_pairs_file(filepath) -> list:
    """解析 (user, item) 对：兼容单对行（user item）与 NGCF 多物品行
    （user item1 item2 ...）两种格式，空白行忽略。

    Args:
        filepath: 原始数据文件路径
    Returns:
        [(user_id, item_id), ...]
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            for item_str in parts[1:]:
                pairs.append((user, int(item_str)))
    return pairs


def build_meta(train_pairs, test_pairs) -> dict:
    """把原始 id 重映射为 0..n-1 连续空间，并统计数据集元信息。

    Returns:
        {
            "num_users": int,
            "num_items": int,
            "train_pairs": [(u, i), ...],
            "test_pairs": [(u, i), ...],
            "user_items": {u: {i, ...}},   # 训练集用户交互（评估过滤用）
        }
    """
    users = sorted(set(u for u, _ in train_pairs) | set(u for u, _ in test_pairs))
    items = sorted(set(i for _, i in train_pairs) | set(i for _, i in test_pairs))
    user_map = {old: new for new, old in enumerate(users)}
    item_map = {old: new for new, old in enumerate(items)}

    new_train = [(user_map[u], item_map[i]) for u, i in train_pairs]
    new_test = [(user_map[u], item_map[i]) for u, i in test_pairs]

    user_items = {}
    for u, i in new_train:
        user_items.setdefault(u, set()).add(i)

    return {
        "num_users": len(users),
        "num_items": len(items),
        "train_pairs": new_train,
        "test_pairs": new_test,
        "user_items": user_items,
    }


def save_pairs(pairs, outpath):
    """把重映射后的成对数据写回文本（便于人工抽查）。"""
    with open(outpath, "w", encoding="utf-8") as f:
        for u, i in pairs:
            f.write(f"{u} {i}\n")


def run_preprocess(raw_dir, out_dir, dataset="ml100k") -> dict:
    """执行完整预处理，返回统计信息。

    Args:
        raw_dir: 原始数据目录（内含 {dataset}/train.txt 与 test.txt）
        out_dir: 处理后数据输出根目录（写入 {dataset}/ 子目录）
        dataset: 数据集名
    """
    raw_path = os.path.join(raw_dir, dataset)
    out_path = os.path.join(out_dir, dataset)
    os.makedirs(out_path, exist_ok=True)

    train_pairs = parse_pairs_file(os.path.join(raw_path, "train.txt"))
    test_pairs = parse_pairs_file(os.path.join(raw_path, "test.txt"))
    meta = build_meta(train_pairs, test_pairs)

    save_pairs(meta["train_pairs"], os.path.join(out_path, "train_pairs.txt"))
    save_pairs(meta["test_pairs"], os.path.join(out_path, "test_pairs.txt"))
    with open(os.path.join(out_path, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    stats = {
        "num_users": meta["num_users"],
        "num_items": meta["num_items"],
        "train_pairs": len(meta["train_pairs"]),
        "test_pairs": len(meta["test_pairs"]),
    }
    print(f"=== {dataset} 预处理 ===")
    print(f"用户数: {stats['num_users']}")
    print(f"物品数: {stats['num_items']}")
    print(f"训练交互: {stats['train_pairs']}")
    print(f"测试交互: {stats['test_pairs']}")
    print(f"预处理完成 -> {out_path}/")
    return stats


def main():
    parser = argparse.ArgumentParser(description="WMF 数据预处理")
    parser.add_argument("--dataset", type=str, default="ml100k")
    parser.add_argument("--raw_dir", type=str, default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    run_preprocess(args.raw_dir, args.out_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
